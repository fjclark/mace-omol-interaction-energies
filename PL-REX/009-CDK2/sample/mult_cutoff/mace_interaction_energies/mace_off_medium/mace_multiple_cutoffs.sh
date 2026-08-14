#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash mace_multiple_cutoffs.sh [RESULTS_ROOT] [OUTPUT_DIR]
#
# Example:
#   bash mace_multiple_cutoffs.sh 009-CDK2_batch_results/3QQK/results
#
# Expected input folders:
#   RESULTS_ROOT/pocket_cutoff_2A/
#   RESULTS_ROOT/pocket_cutoff_5A/
#   RESULTS_ROOT/pocket_cutoff_10A/
#   RESULTS_ROOT/pocket_cutoff_20A/

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-${SCRIPT_DIR}/mace-off.py}"

RESULTS_ROOT="${1:-3QQK/results}"
SYSTEM_DIR="$(dirname "${RESULTS_ROOT}")"
OUTPUT_DIR="${2:-${RESULTS_ROOT}/mace_off_interaction_energies}"

CUTOFFS=(0A 2A 5A 10A 20A)

MODEL="${MODEL:-small}"
DEVICE="${DEVICE:-cpu}"
LIGAND_CHARGE="${LIGAND_CHARGE:-0}"
LIGAND_RESNAMES="${LIGAND_RESNAMES:-LIG,UNL}"

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "Error: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 1
fi

for cutoff in "${CUTOFFS[@]}"; do

    if [[ "${cutoff}" == "0A" ]]; then
        # Original, unrelaxed structure
        complex_pdb="${SYSTEM_DIR}/input/complex.pdb"
        ligand_sdf="${SYSTEM_DIR}/input/ligand.sdf"
    else
        # MLP-relaxed structures at different pocket cutoffs
        cutoff_dir="${RESULTS_ROOT}/pocket_cutoff_${cutoff}"
        complex_pdb="${cutoff_dir}/complex_pocket_mlp_minimised.pdb"
        ligand_sdf="${cutoff_dir}/complex_pocket_mlp_minimised_ligand.sdf"
    fi

    output_csv="${OUTPUT_DIR}/mace_off_interaction_${cutoff}.csv"

    if [[ ! -f "${complex_pdb}" ]]; then
        echo "Warning: missing complex PDB for ${cutoff}: ${complex_pdb}" >&2
        continue
    fi

    if [[ ! -f "${ligand_sdf}" ]]; then
        echo "Warning: missing ligand SDF for ${cutoff}: ${ligand_sdf}" >&2
        continue
    fi

    echo
    echo "============================================================"
    echo "Running MACE-OFF interaction energy for cutoff ${cutoff}"
    echo "Complex: ${complex_pdb}"
    echo "Ligand:  ${ligand_sdf}"
    echo "Output:  ${output_csv}"
    echo "============================================================"

    python "${PYTHON_SCRIPT}" \
        --complex "${complex_pdb}" \
        --ligand "${ligand_sdf}" \
        --ligand-resnames "${LIGAND_RESNAMES}" \
        --ligand-charge "${LIGAND_CHARGE}" \
        --model "${MODEL}" \
        --device "${DEVICE}" \
        --output "${output_csv}"
done

python - "${OUTPUT_DIR}" <<'PY'
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
csv_files = sorted(
    output_dir.glob("mace_off_interaction_*A.csv"),
    key=lambda path: float(
        re.search(r"_([0-9]+(?:p[0-9]+)?)A\.csv$", path.name)
        .group(1)
        .replace("p", ".")
    ),
)

rows: list[dict[str, str]] = []

for csv_file in csv_files:
    cutoff_match = re.search(
        r"_([0-9]+(?:p[0-9]+)?)A\.csv$",
        csv_file.name,
    )
    if cutoff_match is None:
        continue

    cutoff = cutoff_match.group(1).replace("p", ".")

    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["cutoff_angstrom"] = cutoff
            rows.append(row)

if rows:
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

    print(f"\nCombined summary: {summary}")
else:
    print("\nNo successful cutoff CSV files were found to combine.")
PY

echo
echo "Finished. Results are under:"
echo "${OUTPUT_DIR}"
