# Project Context: PIDSMaker

## What this project is
PIDSMaker is a research framework for **provenance-based intrusion detection**
on DARPA TC / OPTC datasets. It ingests system-call provenance (events between
processes, files, and network flows), builds temporal graphs from them, and
trains GNN-based detectors (VELOX is one of them) to flag malicious edges.

## Repo layout (root)
- `pidsmaker/` — main Python package (models, tasks, utils, factories).
- `config/` — YAML configs per model/dataset (e.g. `velox.yml`).
- `dataset_preprocessing/` — scripts to ingest raw DARPA data into Postgres.
- `postgres/` — DB setup (schema, init scripts).
- `Ground_Truth/` — ground-truth attack labels.
- `scripts/` — run / batch / HPC submission scripts.
- `tests/` — pytest tests.
- `docs/` — design docs.
- `Dockerfile`, `compose-pidsmaker.yml`, `compose-postgres.yml` — containerized run.
- `DARPA_PIDSMAKER_ANALYSIS.md` — **read this**: statistics across the 9 DARPA/OPTC databases.
- `RUNS.md`, `CHANGES.md` — historical run notes.

## How VELOX currently works (data flow)

PostgreSQL (event_table)
→ [construction] NetworkX MultiDiGraphs, edges sorted by time
→ [feat_inference] Build msg tensor per edge:
msg = [src_type_oh(3d), src_emb(64d), edge_type_oh(10d),
dst_type_oh(3d), dst_emb(64d)]
→ [batching] CollatableTemporalData with src, dst, t, msg, y, edge_index
→ [extract_msg_from_data] Split msg into:
x_src      = [src_type_oh, src_emb]
x_dst      = [dst_type_oh, dst_emb]
edge_feats = edge_type_oh
→ [reindexing] scatter (E, d) → (N, d) for the GNN
→ [encoder] LinearEncoder(x_src, x_dst) → (h_src, h_dst)
→ [decoder] h_src[edge_index], h_dst[edge_index] → lin_src + lin_dst → ReLU
→ predict edge type
→ [objective] CrossEntropy(predicted_type, true_type)
→ loss doubles as anomaly score

**Critical gap:** VELOX has no behavioral context per edge. It does not know
if a (src, dst) pair has appeared before, if a source is fanning out across
many destinations (scanning), or if this op type is rare for this process.
That is what this update addresses.

## Dataset facts that drive design
From `DARPA_PIDSMAKER_ANALYSIS.md`:
- **9 databases**, 4 tables each (event_table + 3 node tables).
- Event counts range from **18M to 1B** per database.
- **8–25 operation types** depending on dataset (READ, WRITE, OPEN, EXECUTE, …).
- **3 node types**: subject (process), file, netflow.
- Attack patterns vary: Nginx backdoors (CADETS), Firefox exploitation
  (CLEARSCOPE / THEIA), Ransomware / C2 (OPTC).
- **CLEARSCOPE** has only 1 malicious process but **40 malicious files** — destination features matter.
- Attacks are **temporally concentrated** (bursts), not spread evenly.
- Some datasets are almost entirely external traffic (CLEARSCOPE); others mostly internal.

## Key files & their roles
- `pidsmaker/tasks/feat_inference.py` — where per-edge feature tensors are built.
- `pidsmaker/utils/data_utils.py` — `extract_msg_from_data()` (~L178–348), `build_edge_feats()`. The carry-through layer for engineered features.
- `pidsmaker/encoders/linear_encoder.py` — reference encoder pattern.
- `pidsmaker/decoders/edge_linear_decoder.py` — per-edge prediction head.
- `pidsmaker/objectives/predict_edge_type.py` — training objective.
- `pidsmaker/factory.py` — `encoder_factory`, `get_edge_dim`. New components register here.
- `pidsmaker/preprocessing/build_graph_methods/build_default_graphs.py` — window creation; produces `raw_edges` and `fusion_idxs` when `fuse_edge=True`.
- `pidsmaker/config/config.py` — `TASK_ARGS` schema (validate new config sections here).
- `config/velox.yml` — VELOX config; gets the new `edge_engineering` block.

## Environment
- Python via conda; env name is project-local (see HPC alias).
- Postgres for event storage (started via `compose-postgres.yml`).
- GPU recommended for training; CPU works for small windows / smoke tests.
- HPC: SLURM cluster, GPU partition, project path:
  `/srv/lustre01/project/nlp_team-um6p-st-sccs-id7fz1zvotk/IDS/ahmed.bargady/xp/PIDSMaker/`

## Where things live
- Raw + ingested data: in Postgres (see `compose-postgres.yml`).
- Configs: `config/*.yml`.
- Run outputs / checkpoints: per the path set in the chosen config.
- HPC scratch / project root: see path above.

## Known quirks
- Edge fusion (`fuse_edge=True`) merges duplicate edges in a window; the
  pre-fusion stream is preserved as `raw_edges` and reverse-mapped via
  `fusion_idxs`. Any per-edge feature must respect this mapping.
- Operation vocabulary size differs per dataset — anything that one-hots over
  ops must read the dataset's op list, not a hard-coded constant.
- `msg` tensor layout is positional; if you change widths upstream, every
  downstream slice in `extract_msg_from_data` must be updated.