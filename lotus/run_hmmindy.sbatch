#!/bin/bash
# One array task per (W mask, subject) fit of run_model_real.py on Lotus. Each task saves
# its fit to runs/<array job id>/; lotus/aggregate_hmmindy.sbatch then combines them.
#
# Submit both with lotus/submit_hmmindy.sh (from the repo root), which sizes the array
# to match n_masks * n_models in run_model_real.py. To refit only some runs, e.g. ones
# that failed, reusing an earlier run directory:
#     sbatch --array=3,7 --export=ALL,RUN_DIR=runs/<old job id> lotus/run_hmmindy.sbatch
#
# Each fit is a small model (nChan * nPop units), so a few cores each are enough. 12 CPUs
# and 60G lets 4 tasks share one 48-core / 256 GB node.
#
#SBATCH --job-name="hmmindy"
#SBATCH --output="hmmindy_%A_%a.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --export=ALL
#SBATCH -t 48:00:00
#SBATCH --array=0-19
#SBATCH --no-requeue

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
# All tasks of one array job share a directory, so runs from different submissions never mix
RUN_DIR="${RUN_DIR:-runs/$SLURM_ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"

# Python environment created by lotus/setup_env.sh
if command -v module >/dev/null 2>&1; then
    module load python 2>/dev/null || module load python3 2>/dev/null || true
fi
source "$HOME/envs/hmmindy/bin/activate"

# Use every core we were given for PyTorch/NumPy math
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# No display on compute nodes: draw figures to files, make plt.show() a no-op
export MPLBACKEND=Agg
# Print training progress to the log as it happens
export PYTHONUNBUFFERED=1

echo "Job $SLURM_ARRAY_JOB_ID, array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Fitting run $SLURM_ARRAY_TASK_ID with $SLURM_CPUS_PER_TASK CPUs into $RUN_DIR, started $(date)"
python run_model_real.py --run "$SLURM_ARRAY_TASK_ID" --run-dir "$RUN_DIR"
echo "Finished run $SLURM_ARRAY_TASK_ID at $(date)"
