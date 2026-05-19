# Find VELOX run entrypoints

**When to use:** when you need the launch/submission script for VELOX runs before scheduling on HPC.

**Steps / commands:**
1. List scripts: `list_directory` on `scripts/` and `scripts/apptainer/`.
2. Search script contents for VELOX/SLURM markers with `grep`:
   - `velox|VELOX|python -m pidsmaker.main`
   - `sbatch|#SBATCH|slurm|partition`
3. If no match is found, confirm there is no dedicated VELOX SLURM script in-repo and use generic run scripts or ask user for external submission file.

**Notes / pitfalls:** this repo currently exposes generic run wrappers (`scripts/run.sh`, `scripts/run_all_datasets.sh`, `scripts/run_serial.sh`) and apptainer helpers, but no explicit VELOX SLURM script under `scripts/`.
