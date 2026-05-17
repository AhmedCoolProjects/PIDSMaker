# Edge Engineered Features — Bottleneck Analysis & Fixes

## Context

The `RollingWindowFeatureComputer` in `pidsmaker/featurization/edge_engineering/rolling_window_featurizer.py`
computes rolling-window statistical features for each edge in a graph. It is called from
`pidsmaker/tasks/feat_inference.py` inside a per-graph loop.

When `before_fusion=True`, the featurizer processes the **raw (unfused)** edge stream for each
15-minute window. A single CLEARSCOPE_E3 time window can contain ~50K+ raw edges. With all 7
feature categories enabled, the pipeline stalls for **over 1 hour** on a single iteration.

---

## Pipeline Recap

```
Database (raw events)
  │
  ▼
CONSTRUCTION: 15-min time windows, fuse_edge=True
  │  per-window: raw_edges saved in graph.graph["raw_edges"]
  │              fused edges in graph edges
  │              fusion_idxs maps fused→raw position
  ▼
FEAT_INFERENCE:
  │  per graph: RollingWindowFeatureComputer.compute(raw_edges)
  │             then index by fusion_idxs to get fused-edge features
  ▼
BATCHING → TRAINING
```

Fusion is **per-window** — no cross-window state. `raw_edges` are events strictly within one
15-min window. The featurizer reset per graph is therefore correct.

---

## Bottleneck #1: `_cat6_global` — O(N × |unique_src|) ← THE KILLER

### How it works now

For **every edge**, `_cat6_global` iterates over **all unique source nodes** ever seen in the
current window to compute `avg_fan_out`:

```python
fan_outs = []
for s in self.window_src_set:
    fan_outs.append(len(self.src_unique_dst.get(s, set())))
avg_fan_out = sum(fan_outs) / unique_src
```

### Concrete example

```
Window with 3 edges:
  Edge 1: (src=A, dst=X)  →  window_src_set={A}        → 1 iteration
  Edge 2: (src=B, dst=Y)  →  window_src_set={A, B}     → 2 iterations
  Edge 3: (src=A, dst=Z)  →  window_src_set={A, B}     → 2 iterations
```

For a real window with **50K edges** and **20K unique sources**:
- Edge ~49000 triggers a loop of 20K iterations
- Total iterations ≈ sum_{i=1}^{20000} i ≈ **200M** source-level iterations
- Each iteration does a dict lookup + set membership + len()

### Fix: Running accumulator (O(1) per edge)

Maintain a counter `_total_fan_out` that tracks the sum of all source fan-outs, updated
incrementally in `_update_state`:

```python
# _update_state:
# When a (src, dst) pair is first seen for this src:
if dst not in self.src_unique_dst[src]:   # before adding
    self._total_fan_out += 1

# _cat6_global (was O(|unique_src|)):
avg_fan_out = self._total_fan_out / max(len(self.window_src_set), 1)  # O(1)
```

**Impact:** The ~1B-iteration loop disappears. Each edge does 2 lookups and a division.

---

## Bottleneck #2: `_cat1_pair` IAT statistics — O(k²) per high-frequency pair

### How it works now

For each pair `(src, dst)`, `pair_timestamps[pair]` is a **growing list** (never pruned). On
every occurrence, IAT mean & std are **recomputed from scratch**:

```python
diffs = [ts_list[i] - ts_list[i-1] for i in range(1, len(ts_list))]
iat_mean = sum(diffs) / len(diffs)  # O(k) pass
var = sum((d - mean) ** 2 for d in diffs) / len(diffs)  # 2nd O(k) pass
```

### Concrete example

A netflow pair that exchanges 1000 packets in 15 minutes:

| Edge # | List size | Work for this edge | Cumulative |
|--------|-----------|-------------------|------------|
| 1      | 0         | O(1)              | O(1)       |
| 2      | 1         | O(1)              | O(2)       |
| 3      | 2         | O(2)              | O(4)       |
| ...    | ...       | ...               | ...        |
| 1000   | 999       | O(999)            | O(500K)    |

For 1000 interactions: **~500K list iterations** for just one pair. With 10K such pairs, this
blows up.

### Fix: Welford's online algorithm (O(1) per edge)

Maintain running mean and `M2` (sum of squared differences from mean) for each pair:

```python
# State per pair_key: count=0, mean=0.0, m2=0.0

# On new timestamp t (in _update_state):
count += 1
delta = t - mean
mean += delta / count
delta2 = t - mean
m2 += delta * delta2

# _cat1_pair reads (O(1)):
iat_mean = mean / window_duration_ns
iat_std  = sqrt(m2 / count) / window_duration_ns  # population std
```

No timestamp list needed at all. Every edge is O(1) regardless of pair frequency.

---

## Bottleneck #3: Unbounded `pair_timestamps`

### How it works now

`pair_timestamps[pair].append(ts)` — timestamps accumulate across the entire window with no
eviction. This causes:
- **Memory bloat**: 50K edges × many high-frequency pairs
- **Slower dict operations**: larger lists → more memory traffic
- **Only matters if fix #2 isn't applied** (with online stats, the list is unused)

### Fix: Drop `pair_timestamps` entirely (with online stats)

If Welford's algorithm is used for IAT statistics, `pair_timestamps` is unnecessary for
`_cat1_pair`. The only other user is `_cat4_pair_type` (which reads `len(pair_timestamps[pair])`
for `pair_count`). That can be served by a simple `pair_count` counter instead:

```python
# Replace pair_timestamps with:
pair_count = defaultdict(int)       # replaces len(pair_timestamps[pair])

# _update_state:
pair_count[pair_key] += 1           # was pair_timestamps[pair_key].append(ts)

# _cat1_pair only needs count (from pair_count), mean, m2 — no list
# _cat4_pair_type reads pair_count[pair_key] instead of len(pair_timestamps[pair_key])
```

If some code genuinely needs the raw timestamps (e.g., for `_time_ratio` calculations with
`first_ts`), maintain lightweight rolling stats: `pair_first_ts`, `pair_last_ts`, `pair_count`,
`iat_mean`, `iat_m2` — no list.

---

## Implementation Plan

1. **Add running `_total_fan_out` counter** → `_update_state` increments it when a new
   `(src, dst)` pair appears for a source. `_cat6_global` uses `_total_fan_out / |window_src_set|`.

2. **Replace IAT computation** → per-pair Welford state (`count`, `mean`, `m2`) updated in
   `_update_state`. `_cat1_pair` reads these directly.

3. **Remove `pair_timestamps`** → replace with `pair_count`, `pair_first_ts`, `pair_last_ts`.
   Use these for `_time_ratio` and count lookups in `_cat1_pair` / `_cat4_pair_type`.

4. **Verify correctness** — run on a small test case (e.g., `velox_edge_cat1.yml`) and compare
   outputs before/after.

---

## Expected Impact

| Category | Before | After |
|----------|--------|-------|
| `_cat6_global` avg_fan_out | O(\|unique_src\|) per edge | O(1) per edge |
| `_cat1_pair` IAT stats | O(k²) per high-freq pair | O(1) per edge |
| Memory (pair_timestamps) | O(total edges) | O(1) per pair |
| Total time (50K edges, all cats) | >1 hour | seconds |
