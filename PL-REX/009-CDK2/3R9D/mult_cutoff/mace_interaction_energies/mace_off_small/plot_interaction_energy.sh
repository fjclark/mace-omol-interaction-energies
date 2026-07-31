#!/usr/bin/env bash

#SBATCH --account=rockhpc_dccadd
#SBATCH --partition=default_free
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --job-name=plot_mace_energy
#SBATCH --output=plot_mace_energy_%j.out
#SBATCH --error=plot_mace_energy_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module purge
module load GCC

# Prevent Matplotlib from trying to open a graphical window.
export MPLBACKEND=Agg

# Use local temporary storage for Pixi cache files.
export PIXI_CACHE_DIR="${TMPDIR:-/tmp}/pixi-cache-${USER}-${SLURM_JOB_ID}"
mkdir -p "${PIXI_CACHE_DIR}"

PIXI="${HOME}/.pixi/bin/pixi"
PIXI_MANIFEST="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp/pixi.toml"

# Save your Python plotting code with this filename.
PYTHON_SCRIPT="${SLURM_SUBMIT_DIR}/plot_interaction_energy.py"

INPUT_CSV="${SLURM_SUBMIT_DIR}/3R9D/results/mace_off_interaction_energies/mace_off_interaction_all_cutoffs.csv"
OUTPUT_PNG="${SLURM_SUBMIT_DIR}/3R9D/results/mace_off_interaction_energies/interaction_energy_vs_cutoff.png"

if [[ ! -x "${PIXI}" ]]; then
    echo "ERROR: Pixi executable not found: ${PIXI}" >&2
    exit 1
fi

if [[ ! -f "${PIXI_MANIFEST}" ]]; then
    echo "ERROR: Pixi manifest not found: ${PIXI_MANIFEST}" >&2
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Plotting script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
    echo "ERROR: Input CSV not found: ${INPUT_CSV}" >&2
    exit 1
fi

echo "============================================================"
echo "MACE interaction-energy plotting job"
echo "============================================================"
echo "Date:              $(date)"
echo "Host:              $(hostname)"
echo "Slurm job ID:      ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Input CSV:         ${INPUT_CSV}"
echo "Output PNG:        ${OUTPUT_PNG}"
echo "Pixi manifest:     ${PIXI_MANIFEST}"
echo "============================================================"

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python -u "${PYTHON_SCRIPT}"

if [[ -f "${OUTPUT_PNG}" ]]; then
    echo
    echo "Plot created successfully:"
    echo "${OUTPUT_PNG}"
else
    echo "ERROR: Python finished, but the expected PNG was not found:" >&2
    echo "${OUTPUT_PNG}" >&2
    exit 1
fi

echo
echo "Job finished: $(date)"
