#!/usr/bin/env bash

#SBATCH --job-name=cdk2_mlp_cpu
#SBATCH --account=rockhpc_dccadd
#SBATCH --partition=default_free
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

WORK_DIR="/nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2"

PYTHON_SCRIPT="${WORK_DIR}/mopac_mlp_test4.py"

PIXI_MANIFEST="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp/pixi.toml"

PIXI="${HOME}/.pixi/bin/pixi"

OUTPUT_ROOT="${WORK_DIR}/local_results"

# ------------------------------------------------------------
# Move to working directory
# ------------------------------------------------------------

cd "${WORK_DIR}"

mkdir -p logs
mkdir -p "${OUTPUT_ROOT}"

# ------------------------------------------------------------
# CPU settings
# ------------------------------------------------------------

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Prevent accidental CUDA use in this CPU job.
export CUDA_VISIBLE_DEVICES=""

# Optional: load the same GCC module used by your working jobs.
# module load GCC

# ------------------------------------------------------------
# Check important files
# ------------------------------------------------------------

if [[ ! -x "${PIXI}" ]]; then
    echo "ERROR: Pixi was not found at ${PIXI}"
    exit 1
fi

if [[ ! -f "${PIXI_MANIFEST}" ]]; then
    echo "ERROR: pixi.toml was not found at ${PIXI_MANIFEST}"
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script was not found at ${PYTHON_SCRIPT}"
    exit 1
fi

# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

echo "Date:              $(date)"
echo "Host:              $(hostname)"
echo "Working directory: $(pwd)"
echo "Slurm job ID:      ${SLURM_JOB_ID}"
echo "Allocated CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Python script:     ${PYTHON_SCRIPT}"
echo "Output root:       ${OUTPUT_ROOT}"

# ------------------------------------------------------------
# Run the Python script
# ------------------------------------------------------------

"${PIXI}" run -m "${PIXI_MANIFEST}" \
    python "${PYTHON_SCRIPT}" \
    --pdb-id 3SQQ \
    --platform CPU \
    --output-root "${OUTPUT_ROOT}" \
    --pocket-mode "ligand+pocket" \
    --max-iterations 1000 \
    --ligand-resname UNL \
    --pocket-cutoff 2.0 \
    --pocket-cutoff 5.0 \
    --pocket-cutoff 10.0 \
    --pocket-cutoff 20.0

echo "Job completed at: $(date)"