# Update: velox-edge-fe — Edge Feature Engineering for VELOX

## Purpose
VELOX currently has zero behavioral context per edge. It only sees node types,
node embeddings, and the edge type — nothing about whether this (src, dst) pair
has been seen before, whether this source is scanning many destinations, or
whether this operation is rare in the current window. The goal of this update
is to give VELOX rolling-window statistical edge features so it can detect
anomalies that depend on *behavioral* context, not just structural typing.

## Goals
- Design edge features **justified by attack patterns** in the DARPA/OPTC
  datasets — not random categories. Use `DARPA_PIDSMAKER_ANALYSIS.md` as the
  starting point for feature rationale.
- Compute features **causally**: each edge `i` uses only edges `0..i-1` within
  the same time window.
- Keep computation **O(1) per edge** via incremental state (counters, EMAs,
  reservoirs, hash sketches if needed) — no loops over growing sets.
- Integrate cleanly into the existing pipeline through the `engineered_feats`
  tensor path already half-supported in `data_utils.py`.
- Make the whole thing **toggleable from config** so we can ablate.

## Non-goals
- Changing the GNN/decoder architecture beyond what's needed to consume the new features.
- Touching the database schema or preprocessing of node embeddings.
- Replacing existing edge-type prediction objective. We're augmenting inputs, not changing the task.
- Retraining node embeddings.

## Acceptance criteria
- [ ] A new `engineered_feats` tensor of shape `(E, feat_dim)` is produced per
      graph during `feat_inference.py`, computed causally and in O(1) per edge.
- [ ] `extract_msg_from_data()` in `data_utils.py` carries it through to
      `fields["engineered_feats"]`.
- [ ] `build_edge_feats()` includes it when `batching.edge_features` contains
      `"engineered"`.
- [ ] A new encoder (e.g. `LinearEdgeFeatEncoder`) consumes `(x_src, x_dst, edge_feats)`
      and returns per-edge embeddings compatible with the existing
      `edge_linear_decoder` + `predict_edge_type` objective.
- [ ] New config section `edge_engineering` with toggleable feature categories,
      wired into `config/config.py` (`TASK_ARGS`) and `config/velox.yml`.
- [ ] Fusion path works: when `fuse_edge=True`, features are computed on the
      pre-fusion stream and indexed by `fusion_idxs` for fused edges.
- [ ] Runs end-to-end on at least one DARPA dataset locally and one on HPC.
- [ ] A short ablation in `versions/v0.1.md` showing baseline VELOX vs.
      VELOX + engineered features on the same dataset/seed.

## Constraints
- Do not modify code unrelated to this feature path.
- Must handle varying operation vocabularies across datasets (cadets has 25
  ops, clearscope has 8, optc has 10) — feature dimensions must adapt or be
  dataset-aware via config, not hard-coded.
- Backward compatible: if `edge_features` does not contain `"engineered"`,
  VELOX must behave exactly as before.
- Memory budget: incremental state per window must not exceed `O(N + E)`
  where N = nodes, E = edges in window.

## Pipeline integration points
Reference paths in the repo:
- `pidsmaker/tasks/feat_inference.py` — build `engineered_feats` per graph here.
- `pidsmaker/utils/data_utils.py` — `extract_msg_from_data()` (~L178–348) and
  `build_edge_feats()` are the carry-through points.
- `pidsmaker/encoders/linear_encoder.py` — pattern to mirror for the new encoder.
- `pidsmaker/decoders/edge_linear_decoder.py` — keep output shape compatible.
- `pidsmaker/objectives/predict_edge_type.py` — unchanged.
- `pidsmaker/factory.py` — register the new encoder; update `get_edge_dim`.
- `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py` —
  window creation and fusion logic, including `raw_edges` and `fusion_idxs`.
- `pidsmaker/config/config.py` — `TASK_ARGS` schema for `edge_engineering`.
- `config/velox.yml` — enable the new section.

## Expected feature families (starting point — refine after data analysis)
- **Pair recurrence:** how often has this (src, dst) appeared in this window?
- **Source fan-out / dst fan-in:** unique destinations touched by src so far / unique sources touching dst.
- **Operation rarity:** frequency of this edge type for this src (and globally in window).
- **Temporal:** time since src's last edge, time since (src, dst)'s last edge, inter-arrival stats.
- **Burstiness:** short-window edge rate for src, EMA of activity.
- **Type-mix entropy:** Shannon entropy of operation types issued by src so far.

The agent should justify each retained family with a sentence tying it to an
attack pattern from `DARPA_PIDSMAKER_ANALYSIS.md`.

## How to run
- Local: `python -m pidsmaker.main --config config/velox.yml` (adapt as in the existing README).
- HPC: existing SLURM script in `scripts/` — confirm name with me before submitting.

## Definition of done
- All acceptance criteria checked.
- `versions/v0.1.md` written with the ablation result.
- PR opened from your agent branch into `fe_edge`.