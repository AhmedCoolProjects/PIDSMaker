# Changelog

## 2026-05-15 — wandb Metrics Reporting Fixes

### Bug Fixes

- **Semantic mislabeling of `_max`/`_min` aggregates** (`uncertainty.py`): `max_metrics()` and `min_metrics()` were labeling all metrics from the best-adp-score run as `_max`/`_min`, making e.g. `f1_max` mean "F1 at best adp run" rather than "highest F1 across runs". Replaced with four explicit functions:
  - `true_max_metrics` / `true_min_metrics` — per-metric max/min independently across all iterations (`f1_max`, `fp_min`, …)
  - `best_adp_metrics` / `worst_adp_metrics` — coherent snapshot of all metrics at the best/worst adp_score iteration (`f1_at_best_adp`, `fp_at_best_adp`, …)

- **Silent data loss in `fuse_hyperparameter_metrics`** (`uncertainty.py`): Filter `if "precision" in d` was checking for a literal key named `"precision"`, silently dropping all run data and producing NaN averages. Fixed to `if metric in d`.

- **Crash on missing metrics** (`uncertainty.py`): `avg_std_metrics` used hard dict access `entry[key]` which threw `KeyError` if any iteration was missing a metric, aborting the whole experiment. Now uses `.get(key, np.nan)` with `np.nanmean`/`np.nanstd`, and collects keys from the union of all runs instead of just the first run.

- **Multiple `wandb.log()` calls corrupting step counter** (`main.py`): Three separate `wandb.log()` calls for averaged, min, and max metrics were advancing the wandb step counter 3 times. Merged into a single `wandb.log()` call.

- **Empty `all_test_stats` crash** (`training_loop.py`): `np.max([d["peak_inference_cpu_memory"] for d in all_test_stats])` raised `ValueError` when no epochs produced test stats. Guarded with `if all_test_stats`.

### Improvements

- **Per-iteration metrics logged to wandb** (`main.py`): Each uncertainty iteration now immediately logs its metrics as namespaced keys (`deep_ensemble/iter_0/f1`, `deep_ensemble/iter_1/adp_score`, …) in the same wandb run, enabling per-run inspection and outlier detection without separate runs.
