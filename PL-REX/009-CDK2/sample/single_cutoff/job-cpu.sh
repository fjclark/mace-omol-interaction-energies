#!/bin/bash
#SBATCH --account=rockhpc_dccadd
#SBATCH --partition=default_free
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name=cdk2_mopac_mlp_cpu
#SBATCH --output=cdk2_mopac_mlp_cpu_%j.out
#SBATCH --error=cdk2_mopac_mlp_cpu_%j.err

set -euo pipefail

date
echo "Running on: $HOSTNAME"
echo "PWD: $PWD"

# ------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------
module purge
module load Mamba
module load GCC

# ------------------------------------------------------------------
# Conda environment
# ------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate mace_test_env
conda activate mace-omol-updated

# ------------------------------------------------------------------
# CPU-only settings
# ------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Disable torch.compile (sometimes problematic on clusters)
export TORCH_COMPILE_DISABLE=1

# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------
echo "Conda env: ${CONDA_PREFIX}"
echo "Python: $(which python)"
python -V

echo
echo "===== MOPAC CHECK ====="

# Verify MOPAC exists
command -v mopac >/dev/null 2>&1 || {
    echo "ERROR: mopac not found in PATH"
    exit 1
}

echo "MOPAC executable:"
which mopac

echo "===== ENVIRONMENT ====="
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
# SCRIPT="/nobackup/proj/rockhpc_dccadd/ckn/mlp/009_cdk2/mopac_mlp_test4.py"
SCRIPT = "mopac_mlp_test4.py"
# OUT_DIR="/nobackup/proj/rockhpc_dccadd/ckn/mlp/009cdk2/009-CDK2_batch_results"
OUT_DIR"mlp/3QQK"

mkdir -p "${OUT_DIR}"

echo
echo "===== STARTING JOB ====="
echo "Script: ${SCRIPT}"
echo "Output: ${OUT_DIR}"
echo

# ------------------------------------------------------------------
# Run workflow
# ------------------------------------------------------------------
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

