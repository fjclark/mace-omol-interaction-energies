from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------------------------
# Input and output files
# ------------------------------------------------------------------
INPUT_CSV = Path(
    "3SQQ/results/mace_off_interaction_energies/"
    "mace_off_interaction_all_cutoffs.csv"
)

OUTPUT_PNG = INPUT_CSV.parent / "interaction_energy_vs_cutoff.png"


# ------------------------------------------------------------------
# Read results
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

print("CSV columns:")
print(df.columns.tolist())


# Change this if your CSV uses a different column name
ENERGY_COLUMN = "interaction_energy_kcal_mol"

required_columns = {"cutoff_angstrom", ENERGY_COLUMN}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# Convert both columns to numeric values
df["cutoff_angstrom"] = pd.to_numeric(
    df["cutoff_angstrom"],
    errors="coerce",
)

df[ENERGY_COLUMN] = pd.to_numeric(
    df[ENERGY_COLUMN],
    errors="coerce",
)

# Remove invalid rows and order by cutoff
df = (
    df.dropna(subset=["cutoff_angstrom", ENERGY_COLUMN])
    .sort_values("cutoff_angstrom")
)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    df["cutoff_angstrom"],
    df[ENERGY_COLUMN],
    marker="o",
    linewidth=2,
)

# Add the energy value next to each point
for cutoff, energy in zip(
    df["cutoff_angstrom"],
    df[ENERGY_COLUMN],
):
    plt.annotate(
        f"{energy:.2f}",
        xy=(cutoff, energy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
    )

plt.xlabel("Pocket cutoff (Å)")
plt.ylabel("Interaction energy (kcal/mol)")
plt.title("MACE-OFF23(small) interaction energy vs pocket cutoff")

plt.xticks(df["cutoff_angstrom"])
plt.axhline(0, linewidth=1)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"Plot saved to: {OUTPUT_PNG}")