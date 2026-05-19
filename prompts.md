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

### [2026-05-18] — Edge feature engineering initial prompt + design acceptance
**Prompt:**
> Full prompt: Read AGENTS.md, INSTRUCTIONS.md, context.md, skills/README.md, DARPA_PIDSMAKER_ANALYSIS.md and key source files. Then reply with: (a) goal restatement, (b) retained edge feature families with attack pattern rationale, (c) implementation plan, (d) clarifying questions. Do not write code yet.
>
> Followed by design acceptance with decisions on fusion, normalization, reservoir caps, op-vocab, datasets, state scope, and workflow instructions to implement without committing.
>
> Full prompt text in conversation history.

**Agent action:** Read all relevant source files, composed pre-implementation analysis with 6 feature families, design pitch, then implemented full end-to-end: `engineered_feats.py`, `linear_edge_feat_encoder.py`, wiring through `feat_inference.py`, `data_utils.py`, `factory.py`, `config.py`, `velox.yml`. No commits.

---

### [2026-05-18 2] — Commit and push instruction
**Prompt:**
> Commit your changes to YOUR branch only (fe_edge-bigpickle). Do NOT merge or push to fe_edge.
>
> git add -A && git commit -m "v0.1 engineered edge features (draft for ablation)" && git push origin HEAD
>
> Then append this prompt to prompts.md and commit+push that too as a second commit.

**Agent action:** Committed and pushed the implementation, then appended this prompt entry and committed+separately.

---

### [2026-05-18 3] — Fix op2id crash in feat_inference.py
**Prompt:**
> The run failed: `AttributeError: 'str' object has no attribute 'argmax'` on line 57 — `etype2oh` values are strings, not tensors. Fix op2id to use rel2id instead. Show diff, no commit.

**Agent action:** Root cause: `gen_relation_onehot` returns a bidirectional dict mixing tensor and string values. Fixed by passing `rel2id` (which maps op_name→int) as a parameter, deriving zero-based `op2id` from it with proper type filtering and dynamic offset adjustment.
---