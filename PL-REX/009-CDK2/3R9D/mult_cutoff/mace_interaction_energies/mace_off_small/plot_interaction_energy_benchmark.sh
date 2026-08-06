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


# ============================================================
# Purpose
# ============================================================
#
# Plot:
#
#   - true 0, 2, 5, 10 and 20 A MLP-relaxed interaction energies
#   - the original unoptimised interaction energy as a dashed benchmark
#
# Default usage:
#
#   sbatch plot_interaction_energy_benchmark.sh
#
# Another system:
#
#   SYSTEM_ID=3QQK sbatch plot_interaction_energy_benchmark.sh
#
# Another model label:
#
#   MODEL_LABEL="MACE-OFF23(small)" \
#       sbatch plot_interaction_energy_benchmark.sh
# ============================================================


# ------------------------------------------------------------
# Working directory and modules
# ------------------------------------------------------------

cd "${SLURM_SUBMIT_DIR}"

module --force purge
module load GCC


# ------------------------------------------------------------
# Headless Matplotlib and Pixi cache
# ------------------------------------------------------------

export MPLBACKEND=Agg

export PIXI_CACHE_DIR="${TMPDIR:-/tmp}/pixi-cache-${USER}-${SLURM_JOB_ID}"

mkdir -p "${PIXI_CACHE_DIR}"


# ------------------------------------------------------------
# Paths and labels
# ------------------------------------------------------------

PIXI="${HOME}/.pixi/bin/pixi"

PIXI_MANIFEST="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp/pixi.toml"

PYTHON_SCRIPT="${SLURM_SUBMIT_DIR}/plot_interaction_energy_benchmark.py"

SYSTEM_ID="${SYSTEM_ID:-3R9D}"

MODEL_LABEL="${MODEL_LABEL:-MACE-OFF23(small)}"

RESULTS_DIR="${SLURM_SUBMIT_DIR}/${SYSTEM_ID}/results/mace_off_interaction_energies"

INPUT_CSV="${RESULTS_DIR}/mace_off_interaction_all_structures.csv"

OUTPUT_PNG="${RESULTS_DIR}/interaction_energy_with_unoptimised_benchmark.png"


# ------------------------------------------------------------
# Validate required files
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

echo "============================================================"
echo "MACE interaction-energy plotting job"
echo "============================================================"
echo "Date:              $(date)"
echo "Host:              $(hostname)"
echo "Slurm job ID:      ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "System ID:         ${SYSTEM_ID}"
echo "Model label:       ${MODEL_LABEL}"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Input CSV:         ${INPUT_CSV}"
echo "Output PNG:        ${OUTPUT_PNG}"
echo "Pixi manifest:     ${PIXI_MANIFEST}"
echo "============================================================"


# ------------------------------------------------------------
# Run plotting script
# ------------------------------------------------------------

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python -u "${PYTHON_SCRIPT}" \
        --input "${INPUT_CSV}" \
        --output "${OUTPUT_PNG}" \
        --model-label "${MODEL_LABEL}"


# ------------------------------------------------------------
# Verify output
# ------------------------------------------------------------

if [[ -f "${OUTPUT_PNG}" ]]; then
    echo
    echo "Plot created successfully:"
    echo "${OUTPUT_PNG}"
else
    echo \
        "ERROR: Python finished, but the expected PNG was not found:" \
        >&2

    echo "${OUTPUT_PNG}" >&2
    exit 1
fi

echo
echo "Job finished: $(date)"