# Prompt Log

Format for each entry:

---
### [YYYY-MM-DD HH:MM] — <one-line summary>
**Prompt:**
> The exact prompt the user sent.

**Agent action (1–3 lines):** what was done in response.
---

(Agents: append new entries at the bottom. Never edit past entries.)

---
### [2026-05-18 00:00] — Kickoff: scope edge feature engineering for VELOX
**Prompt:**
> You are on branch fe_edge-claude of PIDSMaker. Before writing any code:
>
> Read AGENTS.md, INSTRUCTIONS.md, context.md.
> List skills/ and read anything relevant.
> Read DARPA_PIDSMAKER_ANALYSIS.md and the files listed under "Key files & their roles" in context.md.
>
> Then reply with: (a) a 3-line restatement of the goal, (b) the edge feature families you'd keep and why (one sentence per family tying it to an attack pattern), (c) a 4–6 step implementation plan, (d) any clarifying questions.
> Do not write code yet. Append this prompt to prompts.md.

**Agent action (1–3 lines):** Read AGENTS.md, INSTRUCTIONS.md, context.md, skills/ + versions/ READMEs, DARPA_PIDSMAKER_ANALYSIS.md, and the key pipeline files (feat_inference.py, data_utils.py extract_msg_from_data/build_edge_feats, linear_encoder.py, edge_linear_decoder.py, predict_edge_type.py, factory.py get_edge_dim, build_default_graphs.py fusion logic, config/velox.yml). Returned scope/plan/clarifications below; no code written.
---

---
### [2026-05-18 00:10] — Design accepted; build v0.1 end-to-end, no commits
**Prompt:**
> Design accepted. Updated decisions:
>
> 1. Datasets for v0.1: focus on BOTH cadets_e3 and clearscope_e3. cadets_e3 stresses pair recurrence + op rarity + temporal features; clearscope_e3 stresses destination fan-in (1 malicious process, 40 malicious files). Together they cover most of the feature design.
> 2. State scope: reset per window. No sliding state.
> 3. Fusion aggregation: take the last raw edge's engineered feature vector per fused edge.
> 4. Fusion path: first verify whether raw_edges and fusion_idxs are actually produced in build_default_graphs.py. If yes, implement the mapping. If no, run with fuse_edge=False and defer fusion to v0.2 — do not scope-creep into building the fusion mapping.
> 5. Sketches: plain Python set for fan-out / fan-in for v0.1. HLL noted as v0.2 option.
> 6. Normalization: log1p on count features, optional per-window standardization on temporal Δt. Both togglable via config from day one.
> 7. SLURM script: grep scripts/ for the VELOX submission script and confirm the filename in your reply before submitting anything.
>
> Workflow change:
> - DO NOT COMMIT anything. Leave all changes uncommitted in the working tree. I will review diffs, run experiments, and commit myself.
> - Append this prompt to prompts.md (without committing).
>
> Now your job:
> 1. Produce a concrete implementation plan tailored to BEATING baseline VELOX on cadets_e3 and clearscope_e3 — call out which feature families you expect to drive the gain on each dataset and why, and any design choices (encoder fusion strategy, projection dims, normalization defaults) that you think give the best shot at outperforming baseline. This is your design pitch.
> 2. Then start implementing your full design end-to-end — your best thinking, your best shot at outperforming baseline VELOX. Build all the pieces (state builder, feat_inference wiring, data_utils carry-through, encoder + factory, config + velox.yml).
> 3. Stop only when the implementation is complete and runnable. Do not commit. Show me the file list you touched and a brief summary of what each change does so I can diff and decide.
>
> Append this prompt to prompts.md before starting.

**Agent action (1–3 lines):** Pitched design + built end-to-end without committing. See response for file list and per-file summary.
---

---
### [2026-05-18 00:25] — Dump full diff + answer 3 quick verification questions
**Prompt:**
> Before I run experiments, dump the full diff so I can review in one read:
>
> git diff --stat
> git diff                            # all modified files
> cat pidsmaker/featurization/edge_engineering.py
> cat pidsmaker/encoders/linear_edge_feat_encoder.py
>
> Then answer briefly:
> (a) Confirm: with edge_engineering.enabled=False AND batching.edge_features="edge_type", behavior is bitwise identical to baseline VELOX — yes/no.
> (b) What is the exact engineered feat_dim with all 7 families on? Walk me through the count.
> (c) Where exactly in feat_inference.py is the per-window state reset? Show the line.
>
> Append this prompt to prompts.md.

**Agent action (1–3 lines):** Dumped diff + cat'd new files, answered (a) bitwise-identical caveat (No, technically — `velox.yml` now also overrides `construction.fuse_edge: False`; with that *and* the two flags reverted, yes), (b) walked through 14 dims, (c) pointed at `builder = EdgeFeatureBuilder(...)` inside the per-path loop.
---