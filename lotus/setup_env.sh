#!/bin/bash
# One-time setup of the Python environment for HM-MINDy on Lotus.
# Run this from the login node (lotus-login01) inside the repo directory:
#     bash lotus/setup_env.sh
#
# The Lotus user guide doesn't list a Python module, so this script tries
# Lmod first and, if no Python >= 3.10 is available, installs Miniforge into
# $HOME/miniforge3 (user-local, no admin rights needed).
set -euo pipefail

ENV_DIR="$HOME/envs/hmmindy"

# --- Find a usable Python -----------------------------------------------------
PYTHON=""
if command -v module >/dev/null 2>&1; then
    # `module load` fails if no such module exists; ignore that and fall through
    module load python 2>/dev/null || module load python3 2>/dev/null || true
fi
for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
        PYTHON="$(command -v "$cand")"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No Python >= 3.10 found; installing Miniforge into \$HOME/miniforge3"
    if [ ! -x "$HOME/miniforge3/bin/python3" ]; then
        curl -L -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
        rm -f /tmp/miniforge.sh
    fi
    PYTHON="$HOME/miniforge3/bin/python3"
fi
echo "Using Python: $PYTHON ($("$PYTHON" --version))"

# --- Create the virtual environment ------------------------------------------
"$PYTHON" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# The Lotus user guide lists no GPUs (and the scripts use device="cpu"), so install the CPU build
# of PyTorch (much smaller than the default CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy scipy scikit-learn pandas matplotlib seaborn plotly

# Record which Python the job script should load
echo "$PYTHON" > "$ENV_DIR/.base_python"
echo "Environment ready at $ENV_DIR"
