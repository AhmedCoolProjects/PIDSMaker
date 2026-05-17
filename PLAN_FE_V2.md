# FE V2 — Full-Window Mode + Temporal Position Features

## Motivation

The current `RollingWindowFeatureComputer` operates in **causal mode**: for each edge, features
are read from state accumulated by previous edges only (prefix-state). This is correct for
streaming/online detection, but our pipeline works differently:

1. Construction builds a complete time-windowed graph (e.g., 15 minutes of edges)
2. Featurization processes the **entire graph** at once
3. Detection is applied to the **full window**

Since we always have the complete window available, there is no temporal leakage concern. We can
compute features that reflect the **full window's statistics** — giving every edge a richer,
more stable feature vector.

---

## Architecture: Two Modes

```
mode: "causal"        (default, existing behavior)
mode: "full_window"   (new — two-pass, final-state read)
```

### Causal (existing)

```
For each edge in sorted order:
  1. Read features from state (prefix: edges 0..i-1)
  2. Update state with current edge (now includes edge i)
```

### Full-window (new)

```
Pass 1 — Build state from ALL edges in the graph
  For each edge:
    _update_state(edge)  # no reads

Pass 2 — Read features from final state
  For each edge:
    feats = _read_features(edge)  # reads from complete window state
```

This means:
- `pair_count` for (src,dst) = total raw occurrences in the full window
- `type_count` for WRITE = total raw WRITE events in the full window
- `window_total_edges` = total raw edges in the graph (same value for all edges)
- `src_burst_ratio` = true dominance ratio of this source across the whole window
- Recency features (`time_since_last`) = gap from fused edge's timestamp to the **last** raw
  occurrence of that entity in the window

---

## Categories in Full-Window Mode

| Cat | Name | Full-window semantic | Value |
|-----|------|---------------------|-------|
| 1 | Pair-level | Total pair activity, IAT across all raw edges | High |
| 2 | Source activity | True fan-out across full window | High |
| 3 | Type distribution | True type share of all edges | High |
| 4 | Pair-type | True triple counts across window | High |
| 5 | Source-type | True (src,type) concentration | High |
| 6 | Global | **True global descriptors** — identical for all edges | **Best** |
| 7 | Burst/dominance | **True dominance ratios** — stable across window | **Best** |
| 8 ✨ | Temporal position | Where this edge falls in the window timeline | **New** |

Categories 6 and 7 are especially valuable in full-window mode — in causal mode they change with
every edge (monotonically growing), but in full-window mode they give the complete picture.

---

## New: Category 8 — Temporal Position (6 dims)

Only meaningful with full-window knowledge. Describes where each fused edge sits in the temporal
landscape of the graph's time window.

| # | Feature | Calculation | APT relevance |
|---|---------|-------------|---------------|
| 1 | `time_position` | `(ts - min_ts) / (max_ts - min_ts)` in [0,1] | Attack steps often cluster at specific window phases |
| 2 | `time_from_start` | `(ts - min_ts) / window_duration` | How far into the window |
| 3 | `time_to_end` | `(max_ts - ts) / window_duration` | Time remaining — small near exfiltration phase |
| 4 | `local_edge_density` | `count(raw edges within ±window/100 of ts) / total` | Burst detection — many edges in a tight time bucket |
| 5 | `is_first_quarter` | `1 if time_position < 0.25 else 0` | Initial compromise often in first quarter |
| 6 | `is_last_quarter` | `1 if time_position > 0.75 else 0` | Exfiltration often in last quarter |

All computed from the full raw edge list during Pass 2.

---

## Interaction with `before_fusion`

The full-window mode integrates naturally with `before_fusion`:

```python
# Before fusion: process raw edges, index for fused
featurizer = RollingWindowFeatureComputer(..., mode="full_window")

# eng_edge_data = list of (src, dst, etype_idx, ts) for ALL raw edges
# Pass 1: build state from all raw edges
# Pass 2: read features for all raw edges from final state
all_feats = featurizer.compute_full_window(eng_edge_data)

# Pick only features for fused edges
eng_feats = all_feats[fusion_idxs]  # shape: (num_fused_edges, feat_dim)
```

Each fused edge inherits the features computed at its position in the raw stream — but those
features now reflect the **complete window's statistics** rather than just the prefix.

---

## Config Changes

### New fields in `edge_engineering` block:

```yaml
edge_engineering:
  enabled: True
  mode: "full_window"              # "causal" (default) or "full_window"
  before_fusion: True
  category_1_pair: True
  category_2_source: True
  category_3_type: True
  category_4_pair_type: True
  category_5_source_type: True
  category_6_global: True
  category_7_burst: True
  category_8_temporal: True        # new — only meaningful in full_window mode
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `pidsmaker/featurization/edge_engineering/rolling_window_featurizer.py` | Add `mode` parameter, `compute_full_window()`, `_cat8_temporal()`, update `CATEGORY_DIMS` |
| `pidsmaker/config/config.py` | Add `mode` and `category_8_temporal` to `edge_engineering` arg definitions |
| `pidsmaker/tasks/feat_inference.py` | Dispatch to `compute()` or `compute_full_window()` based on mode |
| `pidsmaker/featurization/edge_engineering/__init__.py` | No changes needed (exports are already dynamic) |
| New configs | Add `velox_edge_full_window.yml`, `velox_edge_fw_cat8.yml`, etc. |
