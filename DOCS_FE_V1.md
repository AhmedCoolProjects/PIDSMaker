# Edge Feature Engineering — Version 1

## Context / Problem Statement

Each graph edge has the form `(src, dst, edge_type, timestamp)`. Edges are processed in sorted
timestamp order. For each edge we attach features computed from **prior edges within the same
time-windowed graph** — before including the current edge. This gives each edge a local behavioral
context that APT detection models can use, while guaranteeing **causality**: no feature leaks
information from the current or future edges.

### Key Timing Semantics

- The pipeline constructs **time-windowed graphs** (e.g., 15-minute windows) in the construction
  step. Each graph is an independent slice.
- `RollingWindowFeatureComputer` is instantiated **fresh per graph** — state resets at graph
  boundaries.
- Within a graph, features accumulate **cumulatively** from the first edge up to (but not including)
  the current edge. There is **no timestamp-based eviction**; the effective window is the entire
  graph's time span.
- `window_duration_ns` (from the config's `time_window_size`) is used **only for normalization**
  of time deltas in `_time_ratio()` — it does not bound how far back the state considers edges.
  Any time delta exceeding `window_duration_ns` is capped to 1.0 (i.e., treated as "very old").

### before_fusion Mode

When `cfg.edge_engineering.before_fusion = True` (requires `construction.fuse_edge = True`):

1. The construction step saves `raw_edges` (pre-fusion events) and `fusion_idxs` in
   `graph.graph["raw_edges"]` and `graph.graph["fusion_idxs"]`.
2. `RollingWindowFeatureComputer` processes the **raw (unfused) edge stream**, so the statistical
   features reflect the higher-resolution unfused event sequence.
3. Feature vectors for fused edges are extracted by indexing:
   `eng_feats = all_feats[fusion_idxs]` (one fused edge inherits the features computed at the
   position of its last constituent raw edge).

This is useful when edge fusion collapses multiple identical edges — computing features on the
pre-fusion stream retains the true inter-arrival patterns and counts.

---

## Five Proxy Keys

| Proxy          | Key                     | Captures                            |
| -------------- | ----------------------- | ----------------------------------- |
| Pair           | `(src, dst)`            | Relationship activity               |
| Source node    | `src`                   | Source behavior / fan-out           |
| Edge type      | `edge_type`             | Operation-level activity            |
| Fine pair-type | `(src, dst, edge_type)` | Specific typed interaction          |
| Source-type    | `(src, edge_type)`      | What a node does with a specific op |

---

## Causal Computation Flow

```
for each edge (src, dst, etype, ts) in sorted order:
    feats = _read_features(src, dst, etype, ts)   ← uses state BEFORE this edge
    _update_state(src, dst, etype, ts)             ← now insert this edge
```

All categories follow this pattern. There is **no difference** between categories in terms of
causality — every single feature reflects only past edges within the current graph.

---

## Feature Categories

### Category 1 — Pair-Level Frequency & Recency
**Proxy:** `(src, dst)`

Maintained state: running count, first/last timestamps, Welford IAT statistics (no list of
timestamps stored).

| Feature                   | Description                                                 | Calculation                                                          |
| ------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| `pair_count`              | How many times this pair has appeared so far in this graph  | `pair_count[pair_key]`                                               |
| `pair_time_since_last`    | Elapsed time since the previous occurrence of this pair     | `_time_ratio(ts - pair_last_ts)`, or `1.0` if first occurrence       |
| `pair_first_seen_offset`  | How long ago the pair was first seen in this graph           | `_time_ratio(ts - pair_first_ts)`, or `0.0` if first occurrence       |
| `pair_activity_span`      | Temporal spread of pair activity so far                     | `_time_ratio(pair_last_ts - pair_first_ts)`, or `0.0` if count ≤ 1   |
| `pair_inter_arrival_mean` | Mean inter-arrival time for this pair (Welford)              | `iat_mean[ pair_key] / window_duration_ns`, or `0.0` if count ≤ 1    |
| `pair_inter_arrival_std`  | Std dev of inter-arrival times (Welford)                     | `sqrt(iat_m2 / iat_n) / window_duration_ns`, or `0.0` if count ≤ 2  |
| `pair_rate`               | Edges per second for this pair so far in this graph          | `pair_count / window_duration_s`                                     |

**APT relevance:** Repeated `(proc → file)` writes or `(proc → net_socket)` connects with
increasing frequency are hallmarks of data exfiltration or beaconing.

**Implementation note:** IAT statistics use **Welford's online algorithm** (O(1) per edge, no
timestamp list). Running state per pair: `count`, `mean`, `m2` (sum of squared deviations from
mean). Updated once per edge in `_update_state`.

---

### Category 2 — Source Node Fan-out & Activity
**Proxy:** `src`

Maintained state: per-src, a total count, a set of unique dst nodes, a set of unique edge types,
and the last-seen timestamp. Sets grow monotonically within the graph (no eviction).

| Feature                 | Description                                      | Calculation                                              |
| ----------------------- | ------------------------------------------------ | -------------------------------------------------------- |
| `src_total_edges`       | Total edges from this src so far in this graph   | `src_count[src]`                                         |
| `src_unique_dst`        | Unique destination nodes reached by this src     | `len(src_unique_dst[src])`                               |
| `src_unique_edge_types` | Unique operation types used by this src          | `len(src_unique_types[src])`                             |
| `src_time_since_last`   | Time since any previous edge from this src       | `_time_ratio(ts - src_last_seen[src])`, or `1.0` if first |
| `src_rate`              | Edge rate for this src so far in this graph      | `src_total_edges / window_duration_s`                    |
| `src_dst_diversity`     | Ratio of unique dst to total src edges           | `src_unique_dst / max(src_total_edges, 1)`               |

**APT relevance:** A process suddenly connecting to many unique dst nodes (high `src_unique_dst`)
is a lateral movement signal. A process using only one edge type repeatedly (low
`src_unique_edge_types`, high `src_total_edges`) suggests scripted behavior.

---

### Category 3 — Edge-Type Level Distribution
**Proxy:** `edge_type`

Maintained state: per-type count, last-seen timestamp; plus global graph edge count.

| Feature                | Description                                        | Calculation                                             |
| ---------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| `type_count`           | Occurrences of this edge type so far in this graph | `type_count[etype]`                                     |
| `type_ratio`           | Fraction of graph-wide edges that are this type     | `type_count[etype] / max(window_total, 1)`              |
| `type_time_since_last` | Time since last occurrence of this edge type       | `_time_ratio(ts - type_last_seen[etype])`, or `1.0` if first |
| `type_rate`            | This edge type's rate per second so far            | `type_count[etype] / window_duration_s`                 |

**APT relevance:** An anomalous surge in `WRITE` or `EXECUTE` events, or a sudden appearance of
an edge type that is normally rare (low `type_ratio` baseline), indicates unusual activity.

---

### Category 4 — Fine-Grained Pair-Type Features
**Proxy:** `(src, dst, edge_type)`

Maintained state: per-triple count and last-seen timestamp. Pair-level count from Category 1
state is reused.

| Feature                     | Description                                                     | Calculation                                                         |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| `pair_type_count`           | Count of this exact triple so far in this graph                  | `triple_count[triple_key]`                                          |
| `pair_type_time_since_last` | Time since last occurrence of this triple                       | `_time_ratio(ts - triple_last_seen[triple_key])`, or `1.0` if first |
| `pair_type_ratio_in_pair`   | Fraction of all pair edges that are this type                   | `triple_count / max(pair_count, 1)`                                 |
| `pair_type_ratio_in_type`   | Fraction of this edge type that involve this pair               | `triple_count / max(type_count, 1)`                                 |

**APT relevance:** A process repeatedly executing the same file (`EXECUTE` on the same binary) or
writing to the same network socket repeatedly is highly suspicious. `pair_type_ratio_in_pair`
near 1.0 means the pair is locked to a single operation type — potentially scripted C2
communication.

---

### Category 5 — Source-EdgeType Joint Features
**Proxy:** `(src, edge_type)`

Maintained state: per `(src, edge_type)` count and last-seen timestamp. Source-level total from
Category 2 is reused.

| Feature                    | Description                                                 | Calculation                                                            |
| -------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| `src_type_count`           | Count of edges from this src with this edge type so far     | `src_type_count[src_type_key]`                                         |
| `src_type_time_since_last` | Time since this src last used this edge type                | `_time_ratio(ts - src_type_last_seen[src_type_key])`, or `1.0` if first |
| `src_type_ratio`           | Fraction of src's edges that use this edge type             | `src_type_count / max(src_count[src], 1)`                              |

**APT relevance:** A process with `src_type_ratio` near 1.0 for `WRITE` is doing almost
exclusively writes — suspicious if that process normally doesn't write. Combines who is acting
(`src`) with what they are doing (`edge_type`).

---

### Category 6 — Global Graph Activity Features
**Proxy:** entire graph (one value per edge, reflecting graph-wide state at that moment)

Maintained state: global counters, unique-node sets, unique-pair sets, and a running
`_total_fan_out` counter (O(1) — avoids looping over all source nodes).

| Feature                    | Description                                             | Calculation                                                       |
| -------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `window_total_edges`       | Total edges seen so far in this graph                   | `window_total`                                                    |
| `window_unique_src`        | Unique source nodes in graph so far                     | `len(window_src_set)`                                             |
| `window_unique_dst`        | Unique destination nodes in graph so far                | `len(window_dst_set)`                                             |
| `window_unique_pairs`      | Unique `(src, dst)` pairs in graph so far               | `len(window_pair_set)`                                            |
| `window_unique_edge_types` | Number of distinct edge types in graph so far           | `len(window_type_set)`                                            |
| `window_edge_type_entropy` | Normalized Shannon entropy of edge-type distribution    | `-sum(p * log(p)) / log(unique_types)`, or `0.0` if total ≤ 1    |
| `window_avg_src_fan_out`   | Average unique dst per src among all active sources     | `_total_fan_out / max(len(window_src_set), 1)`                    |

**APT relevance:** `window_edge_type_entropy` drops during scripted attacks (few operation types
dominate). `window_unique_pairs` grows rapidly during scanning or lateral movement.

**Implementation note:** `_total_fan_out` is a running counter incremented in `_update_state`
whenever a `(src, dst)` pair is **first** observed for that source. This avoids an O(|unique_src|)
loop per edge.

---

### Category 7 — Burstiness / Dominance Features
**Proxy:** cross-proxy ratios (computed from states already maintained above)

| Feature               | Description                             | Calculation                                     |
| --------------------- | --------------------------------------- | ----------------------------------------------- |
| `pair_burst_ratio`    | How dominant this pair is in the graph  | `pair_count / max(window_total, 1)`             |
| `src_burst_ratio`     | How dominant this src is in the graph   | `src_count[src] / max(window_total, 1)`         |
| `pair_type_dominance` | This triple's share of the whole graph  | `triple_count / max(window_total, 1)`           |

**APT relevance:** High `src_burst_ratio` means one actor is responsible for most activity — a
hallmark of automated attack tooling. High `pair_burst_ratio` means one communication channel
monopolizes the graph — typical of a C2 beacon.

---

## Implementation Notes

### State Required Per Proxy

```python
state = {
    # Proxy: (src, dst) — no deque, running stats only
    'pair_count':       defaultdict(int),     # replaces len(pair_timestamps[pair])
    'pair_first_ts':    {},                   # first occurrence timestamp
    'pair_last_ts':     {},                   # last occurrence timestamp
    'pair_iat_mean':    defaultdict(float),   # Welford running mean
    'pair_iat_m2':      defaultdict(float),   # Welford sum squared deviations
    'pair_iat_n':       defaultdict(int),     # Welford sample count

    # Proxy: src
    'src_count':         defaultdict(int),
    'src_last_seen':     {},
    'src_unique_dst':    defaultdict(set),
    'src_unique_types':  defaultdict(set),

    # Proxy: edge_type
    'type_count':        defaultdict(int),
    'type_last_seen':    {},

    # Proxy: (src, dst, edge_type)
    'triple_count':      defaultdict(int),
    'triple_last_seen':  {},

    # Proxy: (src, edge_type)
    'src_type_count':    defaultdict(int),
    'src_type_last_seen': {},

    # Global graph
    'window_total':      0,
    'window_src_set':    set(),
    'window_dst_set':    set(),
    'window_pair_set':   set(),
    'window_type_set':   set(),
    '_total_fan_out':    0,     # running sum of src fan-outs (O(1) for cat6)
}
```

### No Timestamp Eviction

Unlike a true sliding window, there is **no eviction of old entries** within a graph. State
accumulates from the first edge to the current edge. The `window_duration_ns` parameter is used
**only** for normalizing time differences in `_time_ratio()`:

```python
def _time_ratio(self, delta_ns):
    if delta_ns is None:
        return 1.0
    return min(delta_ns / self.window_duration_ns, 1.0)
```

Time deltas that exceed `window_duration_ns` are capped to 1.0 ("very old"), but the edge that
produced them is never removed from the state.

### Processing Order (Guaranteed Causal)

1. **Read features** from current state (before this edge is inserted).
2. **Update state** with the current edge (increment counters, update sets, update Welford stats).

Features for edge at position *i* reflect edges *0..i-1* within the same graph. Edge *i*'s own
information is never included in its feature vector.

### before_fusion Processing

When `before_fusion = True`:

```
raw_edges:  [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9]
fusion:     [e0, e2, e3, e7, e9]   (some edges collapsed)
fusion_idxs: [0,  2,  3,  7,  9]

Processing:
  featurizer.compute(raw_edges)    → all_feats [f0, f1, f2, ..., f9]
  eng_feats = all_feats[fusion_idxs]  → [f0, f2, f3, f7, f9]
```

The fused edge inherits the feature vector computed at the position of its last constituent raw
edge in the pre-fusion stream.

---

## Feature Summary Table

| Feature                     | Proxy          | Type         | APT Signal                    |
| --------------------------- | -------------- | ------------ | ----------------------------- |
| `pair_count`                | (src,dst)      | count        | Repeated access               |
| `pair_time_since_last`      | (src,dst)      | recency      | Periodicity / beaconing       |
| `pair_first_seen_offset`    | (src,dst)      | recency      | Persistence in graph          |
| `pair_activity_span`        | (src,dst)      | temporal     | Burst vs sustained            |
| `pair_inter_arrival_mean`   | (src,dst)      | temporal     | C2 timing regularity          |
| `pair_inter_arrival_std`    | (src,dst)      | temporal     | C2 timing regularity          |
| `pair_rate`                 | (src,dst)      | rate         | High-speed exfil              |
| `src_total_edges`           | src            | count        | Volume of activity            |
| `src_unique_dst`            | src            | diversity    | Lateral movement / scanning   |
| `src_unique_edge_types`     | src            | diversity    | Scripted vs organic           |
| `src_time_since_last`       | src            | recency      | Dormant process waking        |
| `src_rate`                  | src            | rate         | Automation detection          |
| `src_dst_diversity`         | src            | ratio        | Fan-out anomaly               |
| `type_count`                | edge_type      | count        | Op-type surge                 |
| `type_ratio`                | edge_type      | ratio        | Dominance of one op           |
| `type_time_since_last`      | edge_type      | recency      | Rare op reactivation          |
| `type_rate`                 | edge_type      | rate         | Op-type burst                 |
| `pair_type_count`           | (src,dst,type) | count        | Specific repeated interaction |
| `pair_type_time_since_last` | (src,dst,type) | recency      | Fine-grained recency          |
| `pair_type_ratio_in_pair`   | (src,dst,type) | ratio        | Pair locked to one op         |
| `pair_type_ratio_in_type`   | (src,dst,type) | ratio        | Pair monopolizes this op      |
| `src_type_count`            | (src,type)     | count        | How often src uses this op    |
| `src_type_time_since_last`  | (src,type)     | recency      | Sudden new op type for src    |
| `src_type_ratio`            | (src,type)     | ratio        | Op-type concentration for src |
| `window_total_edges`        | global         | count        | Overall activity level        |
| `window_unique_src`         | global         | diversity    | Number of active actors       |
| `window_unique_dst`         | global         | diversity    | Attack surface breadth        |
| `window_unique_pairs`       | global         | diversity    | Interaction graph density     |
| `window_unique_edge_types`  | global         | diversity    | Behavioral variety            |
| `window_edge_type_entropy`  | global         | distribution | Scripted vs organic behavior  |
| `window_avg_src_fan_out`    | global         | distribution | Lateral movement pressure     |
| `pair_burst_ratio`          | (src,dst)      | dominance    | One channel dominates         |
| `src_burst_ratio`           | src            | dominance    | One actor dominates           |
| `pair_type_dominance`       | (src,dst,type) | dominance    | Single triple dominates       |

---

## Performance Characteristics

| Category | Data structure | Time per edge | Memory per proxy |
|----------|---------------|---------------|------------------|
| Cat 1    | Running counters + Welford | O(1) | O(1) per pair |
| Cat 2    | Counters + sets | O(1) | O(\|unique_dst\|) per src |
| Cat 3    | Counters | O(1) | O(1) per type |
| Cat 4    | Counters | O(1) | O(1) per triple |
| Cat 5    | Counters | O(1) | O(1) per (src,type) |
| Cat 6    | Global counters + running `_total_fan_out` | O(1) | O(\|unique\|) per set |
| Cat 7    | Read-only from cat 1/2/4 counters | O(1) | None (reuses existing) |

Total per-edge cost: **O(1)** for all categories with online algorithms.
