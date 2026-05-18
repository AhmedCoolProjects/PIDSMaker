# Agent Rules

You are working on a feature branch of the PIDSMaker project. Before doing ANYTHING:

1. Read `INSTRUCTIONS.md` — the goal of this update.
2. Read `context.md` — project background and architecture.
3. Skim `skills/` — shortcuts and lessons from past work. Use them.
4. Read the latest file in `versions/` if any exist — what was done last.

Then, every time:
- **Log every user prompt** to `prompts.md` (append, never overwrite). Format is in that file.
- **When you learn something reusable** (a command that works, a gotcha, a pattern in this codebase, a data fact), add a new file to `skills/` — one skill per file, named `<short-slug>.md`. Examples: `run-velox-locally.md`, `submit-hpc-job.md`, `darpa-op-types.md`.
- **When a version is finished** (user says "let's tag version X"), create `versions/vX.Y.md` describing what changed, why, and how to verify it.
- **Commit often**, one logical change per commit. Push to YOUR branch only — never to `fe_edge`, `fix_wandb_metrics`, or `main`.
- **Do not edit** `INSTRUCTIONS.md` or `context.md` unless explicitly asked. They are mine.
- **Ask before** installing new dependencies, deleting files, dropping DB tables, or running long HPC jobs.
- **No silent assumptions.** If a path, dataset name, or hyperparameter is unclear, ask.

Branch you are on: see `git branch --show-current`. Stay on it.

## Coding standards for this repo
- Match the existing PyTorch / PyG style in `pidsmaker/`.
- Features must be **causal** (edge `i` only sees edges `0..i-1` in its window) and **O(1) per edge** — no loops over growing sets.
- Add config options under a clearly named section; never hard-code dataset names or paths.
- New components (encoders, feature builders) must be registered via the existing factory pattern in `pidsmaker/factory.py`.