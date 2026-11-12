#!/bin/bash
# Submits the HM-MINDy fits as a job array (one task per W mask x subject) plus an
# aggregate job that starts only after every array task succeeds.
#
# Run from the repo root on the Lotus login node:
#     bash lotus/submit_hmmindy.sh
set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module load python 2>/dev/null || module load python3 2>/dev/null || true
fi
source "$HOME/envs/hmmindy/bin/activate"

# Array size comes from n_masks * n_models in run_model_real.py, so the two can't disagree
N_RUNS=$(MPLBACKEND=Agg python run_model_real.py --count | tail -n 1)
echo "Submitting $N_RUNS fits"

ARRAY_ID=$(sbatch --parsable --array=0-$((N_RUNS - 1)) lotus/run_hmmindy.sbatch)
AGG_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" \
    --export=ALL,RUN_DIR="runs/$ARRAY_ID" lotus/aggregate_hmmindy.sbatch)

echo "Array job:     $ARRAY_ID  (logs: hmmindy_${ARRAY_ID}_<task>.*.out, fits: runs/$ARRAY_ID/)"
echo "Aggregate job: $AGG_ID  (log: hmmindy_agg_${AGG_ID}.*.out), starts after all fits succeed"
