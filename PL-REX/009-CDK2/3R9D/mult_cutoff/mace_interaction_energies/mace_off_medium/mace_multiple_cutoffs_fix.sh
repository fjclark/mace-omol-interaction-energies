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
#   sbatch mace_interactions_cpu.sh
#   sbatch mace_interactions_cpu.sh 3R9D/results
#   sbatch mace_interactions_cpu.sh \
#       3R9D/results \
#       3R9D/results/mace_off_interaction_energies
#
# Calculations:
#
#   unoptimised = original structure; no MLP relaxation
#   0A          = ligand relaxed; receptor frozen
#   2A          = ligand + residues within 2 A relaxed
#   5A          = ligand + residues within 5 A relaxed
#   10A         = ligand + residues within 10 A relaxed
#   20A         = ligand + residues within 20 A relaxed

cd "${SLURM_SUBMIT_DIR}"

module load GCC


# ------------------------------------------------------------
# CPU-only settings
# ------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=""
export WARP_DISABLE_CUDA=1
export TORCH_COMPILE_DISABLE=1
export PYTHONHASHSEED=2026

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"


# ------------------------------------------------------------
# Pixi cache
# ------------------------------------------------------------

export PIXI_CACHE_DIR="${TMPDIR:-/tmp}/pixi-cache-${USER}-${SLURM_JOB_ID}"

mkdir -p "${PIXI_CACHE_DIR}"


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PIXI="${HOME}/.pixi/bin/pixi"

PIXI_PROJECT="/nobackup/proj/rockhpc_dccadd/ckn/pocket-mlp"
PIXI_MANIFEST="${PIXI_PROJECT}/pixi.toml"

SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/mace-off.py}"

RESULTS_ROOT="${1:-3R9D/results}"

# For RESULTS_ROOT=3R9D/results:
# SYSTEM_DIR becomes 3R9D
SYSTEM_DIR="$(dirname "${RESULTS_ROOT}")"

OUTPUT_DIR="${2:-${RESULTS_ROOT}/mace_off_interaction_energies}"


# ------------------------------------------------------------
# Structures to evaluate
# ------------------------------------------------------------

STRUCTURES=(
    unoptimised
    0A
    2A
    5A
    10A
    20A
)


# ------------------------------------------------------------
# Calculation settings
# ------------------------------------------------------------

MODEL="${MODEL:-medium}"
DEVICE="cpu"

LIGAND_CHARGE="${LIGAND_CHARGE:-0}"
LIGAND_RESNAMES="${LIGAND_RESNAMES:-LIG,UNL}"


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
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


# ------------------------------------------------------------
# Remove stale results
# ------------------------------------------------------------

rm -f \
    "${OUTPUT_DIR}/mace_off_interaction_unoptimised.csv" \
    "${OUTPUT_DIR}"/mace_off_interaction_*A.csv \
    "${OUTPUT_DIR}/mace_off_interaction_all_structures.csv"


# ------------------------------------------------------------
# Job information
# ------------------------------------------------------------

printf '%s\n' \
    "============================================================" \
    "MACE-OFF interaction-energy calculations" \
    "============================================================" \
    "Date:              $(date)" \
    "Hostname:          $(hostname)" \
    "Slurm job ID:      ${SLURM_JOB_ID}" \
    "Submission dir:    ${SLURM_SUBMIT_DIR}" \
    "Working directory: $(pwd)" \
    "Allocated CPUs:    ${SLURM_CPUS_PER_TASK}" \
    "Execution device:  ${DEVICE}" \
    "Pixi manifest:     ${PIXI_MANIFEST}" \
    "Python script:     ${PYTHON_SCRIPT}" \
    "System directory:  ${SYSTEM_DIR}" \
    "Results root:      ${RESULTS_ROOT}" \
    "Output directory:  ${OUTPUT_DIR}" \
    "MACE model:        ${MODEL}" \
    "============================================================"


# ------------------------------------------------------------
# Check PyTorch CPU environment
# ------------------------------------------------------------

echo
echo "Checking CPU PyTorch environment..."

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python - <<'PY'
import os
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU threads:", torch.get_num_threads())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
PY


# ------------------------------------------------------------
# Run interaction-energy calculations
# ------------------------------------------------------------

successful_structures=()
failed_structures=()

for structure in "${STRUCTURES[@]}"; do

    if [[ "${structure}" == "unoptimised" ]]; then

        # Original coordinates without MLP relaxation.
        complex_pdb="${SYSTEM_DIR}/input/complex.pdb"
        ligand_sdf="${SYSTEM_DIR}/input/ligand.sdf"

    else

        # MLP-relaxed coordinates for this cutoff.
        cutoff_dir="${RESULTS_ROOT}/pocket_cutoff_${structure}"

        complex_pdb="${cutoff_dir}/complex_pocket_mlp_minimised.pdb"

        ligand_sdf="${cutoff_dir}/complex_pocket_mlp_minimised_ligand.sdf"

    fi

    output_csv="${OUTPUT_DIR}/mace_off_interaction_${structure}.csv"

    if [[ ! -f "${complex_pdb}" ]]; then
        echo \
            "WARNING: Missing complex PDB for ${structure}: ${complex_pdb}" \
            >&2

        failed_structures+=("${structure}")
        continue
    fi

    if [[ ! -f "${ligand_sdf}" ]]; then
        echo \
            "WARNING: Missing ligand SDF for ${structure}: ${ligand_sdf}" \
            >&2

        failed_structures+=("${structure}")
        continue
    fi

    echo
    echo "============================================================"
    echo "Running structure: ${structure}"
    echo "Start time:        $(date)"
    echo "Complex:           ${complex_pdb}"
    echo "Ligand:            ${ligand_sdf}"
    echo "Output:            ${output_csv}"
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
        echo "Completed ${structure}: $(date)"
        successful_structures+=("${structure}")
    else
        echo "ERROR: Calculation failed for ${structure}." >&2
        failed_structures+=("${structure}")
    fi

done


# ------------------------------------------------------------
# Combine all results
# ------------------------------------------------------------

"${PIXI}" run \
    --manifest-path "${PIXI_MANIFEST}" \
    python - "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


output_dir = Path(sys.argv[1])

cutoff_pattern = re.compile(
    r"^mace_off_interaction_"
    r"([0-9]+(?:p[0-9]+)?)A\.csv$"
)

entries: list[
    tuple[float, str, float | None, Path]
] = []


# ------------------------------------------------------------
# Add the unoptimised result
# ------------------------------------------------------------

unoptimised_file = (
    output_dir
    / "mace_off_interaction_unoptimised.csv"
)

if unoptimised_file.exists():
    entries.append(
        (
            -1.0,
            "unoptimised",
            None,
            unoptimised_file,
        )
    )


# ------------------------------------------------------------
# Add 0, 2, 5, 10 and 20 A results
# ------------------------------------------------------------

for path in output_dir.glob(
    "mace_off_interaction_*A.csv"
):
    match = cutoff_pattern.match(path.name)

    if match is None:
        continue

    cutoff = float(
        match.group(1).replace("p", ".")
    )

    entries.append(
        (
            cutoff,
            f"{cutoff:g}A",
            cutoff,
            path,
        )
    )


# Unoptimised first, followed by increasing cutoff.
entries.sort(key=lambda item: item[0])

rows: list[dict[str, str]] = []


# ------------------------------------------------------------
# Read the individual CSV files
# ------------------------------------------------------------

for _, structure_label, cutoff, csv_file in entries:

    with csv_file.open(
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            row["structure_label"] = structure_label

            if structure_label == "unoptimised":

                row["structure_state"] = (
                    "unoptimised_original"
                )

                # No cutoff applies to the original structure.
                row["cutoff_angstrom"] = ""

            else:

                row["structure_state"] = (
                    "mlp_relaxed"
                )

                assert cutoff is not None

                row["cutoff_angstrom"] = (
                    f"{cutoff:g}"
                )

            rows.append(row)


if not rows:
    print(
        "No successful interaction-energy "
        "CSV files were found."
    )
    raise SystemExit(0)


# ------------------------------------------------------------
# Calculate changes relative to the unoptimised result
# ------------------------------------------------------------

unoptimised_rows = [
    row
    for row in rows
    if row["structure_label"] == "unoptimised"
]

if len(unoptimised_rows) == 1:

    baseline = float(
        unoptimised_rows[0][
            "interaction_energy_kcal_mol"
        ]
    )

    for row in rows:

        current = float(
            row["interaction_energy_kcal_mol"]
        )

        row[
            "unoptimised_interaction_energy_kcal_mol"
        ] = f"{baseline:.10f}"

        row[
            "interaction_energy_change_from_unoptimised_kcal_mol"
        ] = f"{current - baseline:.10f}"

else:

    print(
        "WARNING: Expected exactly one "
        "unoptimised result, but found "
        f"{len(unoptimised_rows)}."
    )

    for row in rows:

        row[
            "unoptimised_interaction_energy_kcal_mol"
        ] = ""

        row[
            "interaction_energy_change_from_unoptimised_kcal_mol"
        ] = ""


# ------------------------------------------------------------
# Write combined CSV
# ------------------------------------------------------------

preferred_fields = [
    "structure_label",
    "structure_state",
    "cutoff_angstrom",
    "interaction_energy_kcal_mol",
    "unoptimised_interaction_energy_kcal_mol",
    "interaction_energy_change_from_unoptimised_kcal_mol",
]

fieldnames: list[str] = []

for field in preferred_fields:
    if any(field in row for row in rows):
        fieldnames.append(field)

for row in rows:
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)


summary = (
    output_dir
    / "mace_off_interaction_all_structures.csv"
)

with summary.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(rows)


print(f"Combined summary: {summary}")

print()
print("Structure meanings:")
print(
    "  unoptimised = original coordinates; "
    "no MLP relaxation"
)
print(
    "  0A          = ligand relaxed; "
    "receptor frozen"
)
print(
    "  2A-20A      = ligand and selected "
    "pocket residues relaxed"
)
PY


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Job finished"
echo "============================================================"
echo "Finish time: $(date)"
echo "Results:     ${OUTPUT_DIR}"

if (( ${#successful_structures[@]} > 0 )); then
    echo "Successful:  ${successful_structures[*]}"
else
    echo "Successful:  none"
fi

if (( ${#failed_structures[@]} > 0 )); then
    echo "Failed:      ${failed_structures[*]}"
else
    echo "Failed:      none"
fi

echo "============================================================"
