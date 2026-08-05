#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ENERGY_COLUMN = "interaction_energy_kcal_mol"

EXPECTED_RELAXED_STRUCTURES = {
    "0A": 0.0,
    "2A": 2.0,
    "5A": 5.0,
    "10A": 10.0,
    "20A": 20.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MACE-OFF interaction energies for true 0, 2, 5, 10 and "
            "20 angstrom pocket-relaxed structures, using the original "
            "unoptimised structure as a horizontal benchmark."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Combined CSV produced by the interaction-energy Slurm script.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )

    parser.add_argument(
        "--model-label",
        default="MACE-OFF23(small)",
        help="Model name displayed in the plot title.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input.resolve()
    output_png = args.output.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}"
        )

    df = pd.read_csv(input_csv)

    print("CSV columns:")
    print(df.columns.tolist())

    required_columns = {
        "structure_label",
        ENERGY_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df["structure_label"] = (
        df["structure_label"]
        .astype(str)
        .str.strip()
    )

    df[ENERGY_COLUMN] = pd.to_numeric(
        df[ENERGY_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[ENERGY_COLUMN]
    ).copy()

    # --------------------------------------------------------
    # Find the original unoptimised benchmark
    # --------------------------------------------------------

    unoptimised_rows = df[
        df["structure_label"] == "unoptimised"
    ]

    if len(unoptimised_rows) != 1:
        raise ValueError(
            "Expected exactly one 'unoptimised' row, "
            f"but found {len(unoptimised_rows)}."
        )

    unoptimised_energy = float(
        unoptimised_rows.iloc[0][ENERGY_COLUMN]
    )

    # --------------------------------------------------------
    # Select the true MLP-relaxed cutoff results
    # --------------------------------------------------------

    relaxed_df = df[
        df["structure_label"].isin(
            EXPECTED_RELAXED_STRUCTURES
        )
    ].copy()

    if relaxed_df.empty:
        raise ValueError(
            "No relaxed rows were found for "
            "0A, 2A, 5A, 10A or 20A."
        )

    duplicate_labels = relaxed_df[
        relaxed_df.duplicated(
            subset=["structure_label"],
            keep=False,
        )
    ]

    if not duplicate_labels.empty:
        raise ValueError(
            "Duplicate relaxed rows were found:\n"
            + duplicate_labels[
                ["structure_label", ENERGY_COLUMN]
            ].to_string(index=False)
        )

    relaxed_df["cutoff_angstrom"] = (
        relaxed_df["structure_label"]
        .map(EXPECTED_RELAXED_STRUCTURES)
    )

    relaxed_df = relaxed_df.sort_values(
        "cutoff_angstrom"
    ).reset_index(drop=True)

    print()
    print(
        "Unoptimised benchmark: "
        f"{unoptimised_energy:.6f} kcal/mol"
    )

    print()
    print("Relaxed values being plotted:")
    print(
        relaxed_df[
            [
                "structure_label",
                "cutoff_angstrom",
                ENERGY_COLUMN,
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        relaxed_df["cutoff_angstrom"],
        relaxed_df[ENERGY_COLUMN],
        marker="o",
        linewidth=2,
        label="MLP-relaxed interaction energy",
    )

    for cutoff, energy in zip(
        relaxed_df["cutoff_angstrom"],
        relaxed_df[ENERGY_COLUMN],
    ):
        plt.annotate(
            f"{energy:.2f}",
            xy=(cutoff, energy),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    # The original structure is a benchmark, not a 0 A point.
    plt.axhline(
        y=unoptimised_energy,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Unoptimised benchmark "
            f"({unoptimised_energy:.2f} kcal/mol)"
        ),
    )

    plt.xlabel("Pocket-relaxation cutoff (Å)")
    plt.ylabel("Interaction energy (kcal/mol)")

    plt.title(
        f"{args.model_label} interaction energy\n"
        "Unoptimised benchmark vs MLP-relaxed structures"
    )

    plt.xticks(
        relaxed_df["cutoff_angstrom"]
    )

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print(f"Plot saved to: {output_png}")


if __name__ == "__main__":
    main()
