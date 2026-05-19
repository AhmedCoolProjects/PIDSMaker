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
### [2026-05-18 20:40] — Pre-code reading + planning request for edge feature engineering
**Prompt:**
> You are on branch fe_edge-codex of PIDSMaker. Before writing any code:
>
> Read AGENTS.md, INSTRUCTIONS.md, context.md.
> List skills/ and read anything relevant.
> Read DARPA_PIDSMAKER_ANALYSIS.md and the files listed under "Key files & their roles" in context.md.
>
> Then reply with: (a) a 3-line restatement of the goal, (b) the edge feature families you'd keep and why (one sentence per family tying it to an attack pattern), (c) a 4–6 step implementation plan, (d) any clarifying questions.
> Do not write code yet. Append this prompt to prompts.md.

**Agent action (1–3 lines):** Read the requested project docs and key code files, listed and reviewed `skills/`, then prepared a scoped feature-plan response with clarifying questions.
---

---
### [2026-05-18 20:47] — Design decisions finalized + full implementation request (no commits)
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

**Agent action (1–3 lines):** Logged prompt, then prepared a performance-focused implementation plan and started full end-to-end implementation without committing.
---

---
### [2026-05-18 21:01] — Request for full diffs + sanity checks before experiments
**Prompt:**
> Before I run experiments, two things:
>
> 1. Show me the diff for the files you touched. Use:
>    git diff --stat
>    git diff
>    ...for all modified files, and `cat` the new files (edge_engineering.py, linear_edge_feat_encoder.py, find-velox-run-entrypoints.md). Dump everything so I can review in one read.
>
> 2. Three quick sanity questions, answer briefly:
>    (a) In LinearEdgeFeatEncoder, does the gated-fusion path collapse to baseline behavior when engineered_feats is absent or zeroed? I want to be able to ablate by toggling the config alone, no code change.
>    (b) What's the exact engineered feature dimension (feat_dim) with all families enabled, and is it dataset-dependent? get_edge_dim needs to read it correctly for both cadets_e3 (25 ops) and clearscope_e3 (8 ops).
>    (c) Does the per-window state reset trigger on the right boundary in feat_inference.py? Confirm where in the loop you reset.
>
> Once I review the diff and your answers, I'll run the cadets_e3 baseline + engineered comparison on HPC.
>
> Append this prompt to prompts.md.

**Agent action (1–3 lines):** Appended prompt, then collected `git diff --stat`, full `git diff`, and `cat` output for requested new files, followed by brief sanity answers.
---

---
### [2026-05-19 09:21] — User requested commit + push workflow and status confirmation
**Prompt:**
> Commit your changes to YOUR branch only (fe_edge-<your-name>). Do NOT merge or push to fe_edge.
>
> git add -A
> git commit -m "v0.1 engineered edge features (draft for ablation)"
> git push origin HEAD
>
> Then append this prompt to prompts.md and commit+push that too as a second commit:
> git add prompts.md
> git commit -m "log: commit prompt"
> git push origin HEAD
>
> Confirm with `git log --oneline -3` and `git status`.

**Agent action (1–3 lines):** Committed and pushed engineered-feature changes to the current branch, appended this prompt to `prompts.md`, then prepared a second log-only commit and push plus git log/status confirmation.
---