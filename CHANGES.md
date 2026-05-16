# Changelog

## 2026-05-16 — Edge Engineered Features + before_fusion

### New Features

- **Rolling-window engineered edge features** (new module `pidsmaker/featurization/edge_engineering/`): Adds 7 toggleable categories of causal statistical edge features computed over a rolling time window:
  - Category 1: Pair-level frequency & recency (7 dims)
  - Category 2: Source node fan-out & activity (6 dims)
  - Category 3: Edge-type distribution (4 dims)
  - Category 4: Fine-grained pair-type features (4 dims)
  - Category 5: Source-edge type joint features (3 dims)
  - Category 6: Global window activity (7 dims)
  - Category 7: Burstiness / dominance (3 dims)

- **EdgeEngineeredEncoder** (`pidsmaker/encoders/edge_engineered_encoder.py`): New encoder that consumes engineered edge features alongside node embeddings. Projects edge features, then combines with src/dst node embeddings via a linear layer.

- **Per-category configs** (`config/velox_edge_cat1.yml` through `cat7.yml`): Each enables a single category. `velox_edge_engineered.yml` enables categories 1+2+3+6. `velox_edge_engineered_all.yml` enables all 7.

- **`before_fusion` option** (`edge_engineering.before_fusion`): When `True` (requires `construction.fuse_edge=True`), the `RollingWindowFeatureComputer` processes the **pre-fusion** edge stream so its statistical features reflect unfused events.

### Bug Fixes

- **Feat_inference cache collision** (`pipeline.py`): `feat_inference` task path hash now includes `edge_engineering` config settings, preventing cache collisions between different category configurations.

### Improvements

- **Deep ensemble restart** (`run_n_times.yml`): Changed `restart_from` from `featurization` to `training` to avoid redundant recomputation of featurization and feat_inference across iterations.

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
