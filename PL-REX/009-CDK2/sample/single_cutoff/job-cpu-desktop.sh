#!/bin/bash

set -euo pipefail

date
echo "Running on: $HOSTNAME"
echo "PWD: $PWD"

# Activate Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate mace-omol-updated
conda activate mace_omol_updated

# CPU settings
NCPU=12

export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1
export OMP_NUM_THREADS="${NCPU}"
export MKL_NUM_THREADS="${NCPU}"
export OPENBLAS_NUM_THREADS="${NCPU}"
export TORCH_COMPILE_DISABLE=1

echo "Conda env: ${CONDA_PREFIX}"
echo "Python: $(which python)"
python -V

echo
echo "===== MOPAC CHECK ====="

command -v mopac >/dev/null 2>&1 || {
    echo "ERROR: mopac not found in PATH"
    exit 1
}

echo "MOPAC executable: $(which mopac)"

SCRIPT="mopac_mlp_test4.py"
OUT_DIR="mlp/3QQK"

mkdir -p "${OUT_DIR}"

echo
echo "===== STARTING JOB ====="
echo "Script: ${SCRIPT}"
echo "Output: ${OUT_DIR}"

python -u "${SCRIPT}" \
    --pdb-id 3QQK \
    --output-root "${OUT_DIR}" \
    --model mace-off23-medium \
    --auto-model-by-charge \
    --restraint-k 100000.0 \
    --platform CPU \
    --pocket-mode ligand \
    --pocket-cutoff 5.0 \
    --ligand-resname UNL \
    --max-iterations 1000 \
    --continue-on-error

echo
echo "===== JOB FINISHED ====="
date