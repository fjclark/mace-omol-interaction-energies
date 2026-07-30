#!/usr/bin/env bash

#SBATCH --account=rockhpc_dccadd
#SBATCH --partition=default_free
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --job-name=mace_components_cpu
#SBATCH --output=mace_components_cpu_%j.out
#SBATCH --error=mace_components_cpu_%j.err

set -euo pipefail

# Usage:
#   sbatch mace_multiple_cutoffs_comet_cpu.sh [RESULTS_ROOT] [OUTPUT_DIR]
#
# Examples:
#   sbatch mace_multiple_cutoffs_comet_cpu.sh
#   sbatch mace_multiple_cutoffs_comet_cpu.sh 3SQQ/results
#   sbatch mace_multiple_cutoffs_comet_cpu.sh 3QQK/results
#
# Expected input folders:
#   SYSTEM_DIR/input/complex.pdb
#   SYSTEM_DIR/input/ligand.sdf
#   RESULTS_ROOT/pocket_cutoff_2A/
#   RESULTS_ROOT/pocket_cutoff_5A/
#   RESULTS_ROOT/pocket_cutoff_10A/
#   RESULTS_ROOT/pocket_cutoff_20A/

cd "${SLURM_SUBMIT_DIR}"

module load GCC

# Force CPU-only execution.
export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1
export TORCH_COMPILE_DISABLE=1
export PYTHONHASHSEED=2026
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Put Pixi's temporary package metadata cache on node-local storage.
export PIXI_CACHE_DIR="${TMPDIR:-/tmp}/pixi-cache-${USER}-${SLURM_JOB_ID}"
mkdir -p "${PIXI_CACHE_DIR}"

PIXI="${HOME}/.pixi/bin/pixi"
PIXI_PROJECT="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp"
PIXI_MANIFEST="${PIXI_PROJECT}/pixi.toml"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/mace-off.py}"

RESULTS_ROOT="${1:-3SQQ/results}"
SYSTEM_DIR="$(dirname "${RESULTS_ROOT}")"
OUTPUT_DIR="${2:-${RESULTS_ROOT}/mace_off_interaction_energies}"

CUTOFFS=(0A 2A 5A 10A 20A)

MODEL="${MODEL:-medium}"
DEVICE="cpu"
LIGAND_CHARGE="${LIGAND_CHARGE:-0}"
LIGAND_RESNAMES="${LIGAND_RESNAMES:-LIG,UNL}"

if [[ ! -x "${PIXI}" ]]; then
    echo "ERROR: Pixi executable not found: ${PIXI}" >&2
    exit 1
fi

if [[ ! -f "${PIXI_MANIFEST}" ]]; then
    echo "ERROR: Pixi manifest not found: ${PIXI_MANIFEST}" >&2
    exit 1
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Remove old per-cutoff CSV files so stale results are not merged.
rm -f "${OUTPUT_DIR}"/mace_off_interaction_*A.csv \
      "${OUTPUT_DIR}"/mace_off_interaction_all_cutoffs.csv

printf '%s\n' \
    "============================================================" \
    "MACE-OFF component and interaction energies" \
    "============================================================" \
    "Date:              $(date)" \
    "Hostname:          $(hostname)" \
    "Slurm job ID:      ${SLURM_JOB_ID}" \
    "Submission dir:    ${SLURM_SUBMIT_DIR}" \
    "Working directory: $(pwd)" \
    "Allocated CPUs:    ${SLURM_CPUS_PER_TASK}" \
    "Execution device:   ${DEVICE}" \
    "Pixi manifest:     ${PIXI_MANIFEST}" \
    "Python script:     ${PYTHON_SCRIPT}" \
    "Results root:      ${RESULTS_ROOT}" \
    "Output directory:  ${OUTPUT_DIR}" \
    "Model:             ${MODEL}" \
    "============================================================"

echo
echo "Checking CPU PyTorch environment..."
"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python - <<'PY'
import os
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CPU threads:", torch.get_num_threads())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
PY

successful_cutoffs=()
failed_cutoffs=()

for cutoff in "${CUTOFFS[@]}"; do
    if [[ "${cutoff}" == "0A" ]]; then
        complex_pdb="${SYSTEM_DIR}/input/complex.pdb"
        ligand_sdf="${SYSTEM_DIR}/input/ligand.sdf"
    else
        cutoff_dir="${RESULTS_ROOT}/pocket_cutoff_${cutoff}"
        complex_pdb="${cutoff_dir}/complex_pocket_mlp_minimised.pdb"
        ligand_sdf="${cutoff_dir}/complex_pocket_mlp_minimised_ligand.sdf"
    fi

    output_csv="${OUTPUT_DIR}/mace_off_interaction_${cutoff}.csv"

    if [[ ! -f "${complex_pdb}" ]]; then
        echo "WARNING: Missing complex PDB for ${cutoff}: ${complex_pdb}" >&2
        failed_cutoffs+=("${cutoff}")
        continue
    fi

    if [[ ! -f "${ligand_sdf}" ]]; then
        echo "WARNING: Missing ligand SDF for ${cutoff}: ${ligand_sdf}" >&2
        failed_cutoffs+=("${cutoff}")
        continue
    fi

    echo
    echo "============================================================"
    echo "Running cutoff: ${cutoff}"
    echo "Start time:     $(date)"
    echo "Complex:        ${complex_pdb}"
    echo "Ligand:         ${ligand_sdf}"
    echo "Output:         ${output_csv}"
    echo "============================================================"

    if "${PIXI}" run \
        --manifest-path "${PIXI_MANIFEST}" \
        python -u "${PYTHON_SCRIPT}" \
            --complex "${complex_pdb}" \
            --ligand "${ligand_sdf}" \
            --ligand-resnames "${LIGAND_RESNAMES}" \
            --ligand-charge "${LIGAND_CHARGE}" \
            --model "${MODEL}" \
            --device "${DEVICE}" \
            --output "${output_csv}"
    then
        echo "Completed cutoff ${cutoff}: $(date)"
        successful_cutoffs+=("${cutoff}")
    else
        echo "ERROR: Calculation failed for cutoff ${cutoff}." >&2
        failed_cutoffs+=("${cutoff}")
    fi
done

# Combine every successful per-cutoff CSV into one summary file.
"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python - "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
pattern = re.compile(r"_([0-9]+(?:p[0-9]+)?)A\.csv$")

csv_files: list[tuple[float, Path]] = []
for path in output_dir.glob("mace_off_interaction_*A.csv"):
    match = pattern.search(path.name)
    if match is not None:
        cutoff = float(match.group(1).replace("p", "."))
        csv_files.append((cutoff, path))

csv_files.sort(key=lambda item: item[0])
rows: list[dict[str, str]] = []

for cutoff, csv_file in csv_files:
    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["cutoff_angstrom"] = f"{cutoff:g}"
            rows.append(row)

if not rows:
    print("No successful cutoff CSV files were found to combine.")
    raise SystemExit(0)

fieldnames: list[str] = ["cutoff_angstrom"]
for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)

summary = output_dir / "mace_off_interaction_all_cutoffs.csv"
with summary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Combined summary: {summary}")
PY

echo
echo "============================================================"
echo "Job finished"
echo "============================================================"
echo "Finish time: $(date)"
echo "Results:     ${OUTPUT_DIR}"

if (( ${#successful_cutoffs[@]} > 0 )); then
    echo "Successful:  ${successful_cutoffs[*]}"
else
    echo "Successful:  none"
fi

if (( ${#failed_cutoffs[@]} > 0 )); then
    echo "Failed:      ${failed_cutoffs[*]}"
else
    echo "Failed:      none"
fi

echo "============================================================"