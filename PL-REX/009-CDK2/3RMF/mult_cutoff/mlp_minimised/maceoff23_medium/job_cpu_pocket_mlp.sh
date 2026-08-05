#!/usr/bin/env bash

#SBATCH --job-name=cdk2_mlp_cpu
#SBATCH --account=rockhpc_dccadd
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2/logs/%x-%j.out
#SBATCH --error=/nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2/logs/%x-%j.err

set -euo pipefail


# ============================================================
# Before submitting
# ============================================================
#
# Make sure the log directory exists:
#
#   mkdir -p /nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2/logs
#
# Submit with:
#
#   sbatch run_3R9D_multiple_cutoffs_cpu_corrected.sh
#
# Cutoff meanings:
#
#   0 Å  = ligand relaxed; entire receptor frozen
#   2 Å  = ligand plus residues within 2 Å relaxed
#   5 Å  = ligand plus residues within 5 Å relaxed
#   10 Å = ligand plus residues within 10 Å relaxed
#   20 Å = ligand plus residues within 20 Å relaxed
#
# Every cutoff starts from the same original complex.
# ============================================================


# ------------------------------------------------------------
# Paths and calculation settings
# ------------------------------------------------------------

WORK_DIR="/nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2"

PYTHON_SCRIPT="${WORK_DIR}/pocket_mlp.py"

PIXI_PROJECT="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp"
PIXI_MANIFEST="${PIXI_PROJECT}/pixi.toml"
PIXI="${HOME}/.pixi/bin/pixi"

OUTPUT_ROOT="${WORK_DIR}/local_results"

PDB_ID="${PDB_ID:-3R9D}"  # Default to 3R9D if not set in the environment       
MODEL="${MODEL:-mace-off23-medium}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1000}"
LIGAND_RESNAME="${LIGAND_RESNAME:-UNL}"


# ------------------------------------------------------------
# Move to the working directory
# ------------------------------------------------------------

cd "${WORK_DIR}"

mkdir -p "${OUTPUT_ROOT}"


# ------------------------------------------------------------
# Compiler and CPU-only environment
# ------------------------------------------------------------

module load GCC

export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1
export TORCH_COMPILE_DISABLE=1
export PYTHONHASHSEED=2026

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"


# ------------------------------------------------------------
# Put Pixi cache on node-local storage
# ------------------------------------------------------------

export PIXI_CACHE_DIR="${TMPDIR:-/tmp}/pixi-cache-${USER}-${SLURM_JOB_ID}"
mkdir -p "${PIXI_CACHE_DIR}"


# ------------------------------------------------------------
# Validate required files
# ------------------------------------------------------------

if [[ ! -x "${PIXI}" ]]; then
    echo "ERROR: Pixi executable was not found: ${PIXI}" >&2
    exit 1
fi

if [[ ! -f "${PIXI_MANIFEST}" ]]; then
    echo "ERROR: Pixi manifest was not found: ${PIXI_MANIFEST}" >&2
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Corrected Python script was not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi


# ------------------------------------------------------------
# Print job information
# ------------------------------------------------------------

echo "============================================================"
echo "CDK2 pocket MLP minimisation"
echo "============================================================"
echo "Date:              $(date)"
echo "Host:              $(hostname)"
echo "Slurm job ID:      ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "Allocated CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "PDB ID:            ${PDB_ID}"
echo "Model:             ${MODEL}"
echo "Maximum iterations:${MAX_ITERATIONS}"
echo "Ligand residue:    ${LIGAND_RESNAME}"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Pixi manifest:     ${PIXI_MANIFEST}"
echo "Output root:       ${OUTPUT_ROOT}"
echo "CUDA devices:      '${CUDA_VISIBLE_DEVICES}'"
echo "============================================================"


# ------------------------------------------------------------
# Check Python, PyTorch, OpenMM and MACE
# ------------------------------------------------------------

echo
echo "Checking the CPU environment..."

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python - <<'PY'
import os
import sys

import openmm
import torch

print("Python:", sys.version)
print("OpenMM:", openmm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))
print("PyTorch CPU threads:", torch.get_num_threads())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
PY


# ------------------------------------------------------------
# Run the corrected Python workflow
# ------------------------------------------------------------

echo
echo "Starting calculations at: $(date)"
echo

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python -u "${PYTHON_SCRIPT}" \
        --pdb-id "${PDB_ID}" \
        --platform CPU \
        --output-root "${OUTPUT_ROOT}" \
        --model "${MODEL}" \
        --no-auto-model-by-charge \
        --pocket-mode "ligand+pocket" \
        --max-iterations "${MAX_ITERATIONS}" \
        --ligand-resname "${LIGAND_RESNAME}" \
        --pocket-cutoff 0.0 \
        --pocket-cutoff 2.0 \
        --pocket-cutoff 5.0 \
        --pocket-cutoff 10.0 \
        --pocket-cutoff 20.0 \
        --nonbonded-mode NoCutoff \
        --skip-ligand-mlp


# ------------------------------------------------------------
# Completion summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Job completed successfully"
echo "============================================================"
echo "Completion time: $(date)"
echo "Output root:     ${OUTPUT_ROOT}"
echo
echo "Expected result directories:"
echo "  ${OUTPUT_ROOT}/${PDB_ID}/results/pocket_cutoff_0A"
echo "  ${OUTPUT_ROOT}/${PDB_ID}/results/pocket_cutoff_2A"
echo "  ${OUTPUT_ROOT}/${PDB_ID}/results/pocket_cutoff_5A"
echo "  ${OUTPUT_ROOT}/${PDB_ID}/results/pocket_cutoff_10A"
echo "  ${OUTPUT_ROOT}/${PDB_ID}/results/pocket_cutoff_20A"
echo "============================================================"

