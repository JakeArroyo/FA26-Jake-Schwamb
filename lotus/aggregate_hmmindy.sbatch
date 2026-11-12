#!/bin/bash
# Combines the per-run fits saved by lotus/run_hmmindy.sbatch: picks the best W mask, saves
# HM-MINDy_alpha_fit_<timestamp>.pkl, and draws the summary figures in images/.
#
# lotus/submit_hmmindy.sh submits this automatically once every array task has succeeded.
# To run it by hand (e.g. after refitting failed runs), from the repo root:
#     sbatch --export=ALL,RUN_DIR=runs/<array job id> lotus/aggregate_hmmindy.sbatch
#
#SBATCH --job-name="hmmindy_agg"
#SBATCH --output="hmmindy_agg_%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH -t 02:00:00
#SBATCH --no-requeue

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
: "${RUN_DIR:?Set RUN_DIR, e.g. sbatch --export=ALL,RUN_DIR=runs/<array job id> lotus/aggregate_hmmindy.sbatch}"
mkdir -p images

if command -v module >/dev/null 2>&1; then
    module load python 2>/dev/null || module load python3 2>/dev/null || true
fi
source "$HOME/envs/hmmindy/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname): aggregating $RUN_DIR, started $(date)"
python run_model_real.py --aggregate "$RUN_DIR"
echo "Finished aggregating $RUN_DIR at $(date)"
