#!/usr/bin/env python3

import csv
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

from rdkit import Chem
from rdkit.Chem import AllChem


# =========================
# Settings
# =========================
PDB_ID = "3QTU"
TARGET_FOLDER = "009-CDK2"
PLREX_BASE = "https://raw.githubusercontent.com/Honza-R/PL-REX/main"

NUM_CONFS = 10
RANDOM_SEED = 2026
PRUNE_RMS = -1

BASE_DIR = Path(PDB_ID)
INPUT_DIR = BASE_DIR / "input"
CONF_DIR = BASE_DIR / "conformers"
MOPAC_VAC_DIR = BASE_DIR / "mopac_vacuum"
MOPAC_WATER_DIR = BASE_DIR / "mopac_water"
RESULTS_DIR = BASE_DIR / "results"

INPUT_SDF = INPUT_DIR / "ligand.sdf"
CONF_SDF = CONF_DIR / "ligand_10confs.sdf"

VAC_RESULTS_CSV = RESULTS_DIR / "vacuum_results.csv"
WATER_RESULTS_CSV = RESULTS_DIR / "water_results.csv"


# =========================
# Setup directories
# =========================
def setup_dirs():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    MOPAC_VAC_DIR.mkdir(parents=True, exist_ok=True)
    MOPAC_WATER_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Download ligand from PL-REX
# =========================
def download_ligand():
    url = f"{PLREX_BASE}/{TARGET_FOLDER}/structures_pl-rex/{PDB_ID}/ligand.sdf"
    print(f"Downloading ligand from:\n  {url}")
    urlretrieve(url, INPUT_SDF)
    print(f"Saved to: {INPUT_SDF}")


# =========================
# Charge
# =========================
def get_charge(mol: Chem.Mol) -> int:
    if mol.HasProp("charge"):
        return int(float(mol.GetProp("charge").strip()))
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


# =========================
# Generate conformers
# =========================
def generate_conformers() -> int:
    suppl = Chem.SDMolSupplier(str(INPUT_SDF), removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"Could not read molecule from {INPUT_SDF}")

    charge = get_charge(mol)
    print(f"Using charge: {charge}")

    mol = Chem.AddHs(mol, addCoords=True)

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.pruneRmsThresh = PRUNE_RMS
    params.useSmallRingTorsions = True
    params.enforceChirality = True

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, params=params))
    if not conf_ids:
        raise RuntimeError("No conformers were generated.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        print("Optimizing conformers with MMFF...")
        for cid in conf_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    else:
        print("MMFF unavailable, optimizing conformers with UFF...")
        for cid in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)

    writer = Chem.SDWriter(str(CONF_SDF))
    for cid in conf_ids:
        writer.write(mol, confId=cid)
    writer.close()

    print(f"Generated {len(conf_ids)} conformers -> {CONF_SDF}")
    return charge


# =========================
# MOPAC keywords
# =========================
def build_water_opt_keywords(charge: int) -> str:
    return f"PM7 CHARGE={charge} EPS=78.4"


def build_vacuum_sp_keywords(charge: int) -> str:
    return f"PM7 CHARGE={charge} 1SCF"


# =========================
# RDKit mol -> MOPAC block
# =========================
def mol_to_mopac_xyz_block(mol: Chem.Mol, conf_id: int = 0) -> str:
    conf = mol.GetConformer(conf_id)
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():2s} {pos.x:12.6f} 1 {pos.y:12.6f} 1 {pos.z:12.6f} 1"
        )
    return "\n".join(lines)


# =========================
# Write water optimisation MOPAC inputs
# =========================
def write_water_optimization_inputs(charge: int) -> list[Path]:
    mols = [m for m in Chem.SDMolSupplier(str(CONF_SDF), removeHs=False) if m is not None]
    keywords = build_water_opt_keywords(charge)

    mop_files = []
    for i, mol in enumerate(mols, start=1):
        coord_block = mol_to_mopac_xyz_block(mol, conf_id=0)

        text = (
            f"{keywords}\n"
            f"\n"
            f"\n"
            f"{coord_block}\n"
            f"\n"
        )

        mop_path = MOPAC_WATER_DIR / f"conf_{i:02d}_water_opt.mop"
        mop_path.write_text(text)
        mop_files.append(mop_path)

    print(f"Wrote {len(mop_files)} water optimisation MOPAC input files -> {MOPAC_WATER_DIR}")
    return mop_files


# =========================
# Find / run MOPAC
# =========================
def find_mopac_executable() -> str:
    for exe in ["mopac", "MOPAC2016.exe", "MOPAC.exe"]:
        path = shutil.which(exe)
        if path:
            return path
    raise FileNotFoundError("Could not find MOPAC executable in PATH.")


def run_mopac_jobs(mop_files: list[Path], workdir: Path):
    mopac_exe = find_mopac_executable()
    print(f"Using MOPAC executable: {mopac_exe}")

    for mop_file in mop_files:
        print(f"Running {mop_file.name} in {workdir}")
        subprocess.run(
            [mopac_exe, mop_file.name],
            cwd=str(workdir),
            check=True
        )


# =========================
# Extract final Cartesian block from MOPAC .out
# =========================
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_final_block(out_file: Path) -> str:
    text = out_file.read_text(errors="ignore")

    blocks = re.split(r"CARTESIAN COORDINATES", text)
    if len(blocks) < 2:
        return ""

    final_block = blocks[-1]
    extracted_lines = []

    for line in final_block.splitlines():
        parts = line.split()

        if (
            len(parts) >= 5
            and parts[0].isdigit()
            and is_float(parts[2])
            and is_float(parts[3])
            and is_float(parts[4])
        ):
            extracted_lines.append(line)
        elif extracted_lines:
            break

    return "\n".join(extracted_lines)


def cartesian_block_to_mopac_geometry(cart_block: str, optimize_flags: int = 0) -> str:
    geometry_lines = []

    for line in cart_block.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            atom = parts[1]
            x, y, z = parts[2:5]

            geometry_lines.append(
                f"{atom:<2} {x} {optimize_flags} {y} {optimize_flags} {z} {optimize_flags}"
            )

    return "\n".join(geometry_lines)


# =========================
# Write vacuum single-point inputs from water outputs
# =========================
def write_vacuum_singlepoint_inputs_from_water_outputs(charge: int) -> list[Path]:
    keywords = build_vacuum_sp_keywords(charge)
    vacuum_mop_files = []

    for out_file in sorted(MOPAC_WATER_DIR.glob("conf_*_water_opt.out")):
        final_cart_block = extract_final_block(out_file)

        if not final_cart_block.strip():
            print(f"Warning: no final Cartesian coordinates found in {out_file.name}")
            continue

        coord_txt = MOPAC_VAC_DIR / f"{out_file.stem}_final_cartesian_coordinates.txt"
        coord_txt.write_text(final_cart_block + "\n")

        # Use fixed coordinates for single-point
        mopac_geometry = cartesian_block_to_mopac_geometry(final_cart_block, optimize_flags=0)

        vacuum_mop = MOPAC_VAC_DIR / f"{out_file.stem}_vacuum_sp.mop"
        with open(vacuum_mop, "w") as f:
            f.write(f"{keywords}\n")
            f.write("\n")
            f.write("\n")
            f.write(mopac_geometry)
            f.write("\n")

        vacuum_mop_files.append(vacuum_mop)
        print(f"Wrote vacuum single-point input: {vacuum_mop.name}")

    print(f"Wrote {len(vacuum_mop_files)} vacuum single-point MOPAC input files -> {MOPAC_VAC_DIR}")
    return vacuum_mop_files


# =========================
# Parse .out files
# =========================
def extract_heat_of_formation(text: str):
    patterns = [
        re.compile(r"FINAL\s+HEAT\s+OF\s+FORMATION\s*=\s*([-+0-9.Ee]+)", re.I),
        re.compile(r"HEAT\s+OF\s+FORMATION\s*=\s*([-+0-9.Ee]+)", re.I),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def extract_total_energy(text: str):
    patterns = [
        re.compile(r"TOTAL\s+ENERGY\s*=\s*([-+0-9.Ee]+)", re.I),
        re.compile(r"ELECTRONIC\s+ENERGY\s*=\s*([-+0-9.Ee]+)", re.I),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def extract_status(text: str) -> str:
    low = text.lower()
    if "job ended normally" in low or "== mopac done ==" in low or "mopac done" in low:
        return "ok"
    if "unable to achieve self-consistence" in low:
        return "scf_failed"
    if "too many cycles" in low:
        return "too_many_cycles"
    if "error" in low:
        return "error"
    if "failed" in low:
        return "failed"
    return "unknown"


def parse_outputs(output_dir: Path, csv_path: Path):
    rows = []

    for out_file in sorted(output_dir.glob("*.out")):
        text = out_file.read_text(errors="ignore")
        status = extract_status(text)
        hof = extract_heat_of_formation(text)
        total_energy = extract_total_energy(text)
        rows.append([out_file.name, status, hof, total_energy])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "heat_of_formation", "total_energy"])
        writer.writerows(rows)

    print(f"Results written to: {csv_path}")
    print("\nSummary:")
    for row in rows:
        print(
            f"{row[0]:35s} status={row[1]:15s} "
            f"heat_of_formation={row[2]} total_energy={row[3]}"
        )


# =========================
# Main
# =========================
def main():
    print(f"Running in current directory: {Path.cwd()}")

    setup_dirs()

    # Step 1: download ligand
    download_ligand()

    # Step 2: generate conformers
    charge = generate_conformers()

    # Step 3: write water optimisation inputs
    water_mop_files = write_water_optimization_inputs(charge)

    # Step 4: run water MOPAC optimisation
    run_mopac_jobs(water_mop_files, MOPAC_WATER_DIR)

    # Step 5: parse water optimisation results
    parse_outputs(MOPAC_WATER_DIR, WATER_RESULTS_CSV)

    # Step 6: write vacuum single-point inputs from water-optimised outputs
    vacuum_mop_files = write_vacuum_singlepoint_inputs_from_water_outputs(charge)

    if not vacuum_mop_files:
        print("No vacuum single-point MOPAC files were generated.")
        return

    # Step 7: run vacuum single-point MOPAC
    run_mopac_jobs(vacuum_mop_files, MOPAC_VAC_DIR)

    # Step 8: parse vacuum single-point results
    parse_outputs(MOPAC_VAC_DIR, VAC_RESULTS_CSV)


if __name__ == "__main__":
    main()
