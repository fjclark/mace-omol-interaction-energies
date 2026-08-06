from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Literal

import MDAnalysis as mda
from ase import Atoms
from ase.io import read as ase_read
from mace.calculators import mace_off
from openff.units import unit
from tqdm import tqdm


# Conversion used by ASE/MACE: eV -> kcal/mol.
EV_TO_KCALMOL = 23.0605
ENERGY_UNIT = unit.kilocalorie / unit.mole


# ============================================================
# PDB utilities
# ============================================================


def export_fixed_pdb(
    pdb_file: Path,
    exclude: set[str] | None = None,
    charge_a: int = 0,
    charge_b: int = 0,
) -> Path:
    """Remove selected residues and add receptor/ligand charge remarks."""
    if exclude is None:
        exclude = {"HOH", "WAT", "NA", "K", "CL", "MG", "CA"}

    universe = mda.Universe(str(pdb_file))

    if exclude:
        exclude_string = " ".join(sorted(exclude))
        selected_atoms = universe.select_atoms(f"not resname {exclude_string}")
    else:
        selected_atoms = universe.atoms

    fixed_pdb = pdb_file.with_name(f"{pdb_file.stem}_fixed.pdb")
    selected_atoms.write(str(fixed_pdb))

    with fixed_pdb.open("r+", encoding="utf-8") as handle:
        content = handle.read()
        handle.seek(0)
        handle.write(f"REMARK charge_a {charge_a}\n")
        handle.write(f"REMARK charge_b {charge_b}\n")
        handle.write(content)

    return fixed_pdb


def compute_pdb_charge(
    pdb_path: Path,
    include_n_term: bool = False,
    include_c_term: bool = True,
) -> int:
    """Estimate the integer protein charge using a simple pH 7 residue model."""
    sidechain_charges = {
        "LYS": +1,
        "ARG": +1,
        "ASP": -1,
        "GLU": -1,
        "HIS": 0,
        "HID": 0,
        "HIE": 0,
        "HIP": +1,
    }
    neutral_caps = {"ACE", "NME", "BNC", "BCC", "BCB"}

    residues: dict[tuple[str, str, str], dict[str, object]] = {}

    with Path(pdb_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            residue_name = line[17:20].strip()
            chain = line[21].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            atom_name = line[12:16].strip()

            key = (chain, residue_number, insertion_code)
            residues.setdefault(key, {"name": residue_name, "atoms": set()})
            atoms = residues[key]["atoms"]
            assert isinstance(atoms, set)
            atoms.add(atom_name)

    def residue_sort_key(key: tuple[str, str, str]) -> tuple[str, int, str]:
        chain, residue_number, insertion_code = key
        try:
            numeric_residue_number = int(residue_number)
        except ValueError:
            numeric_residue_number = 0
        return chain, numeric_residue_number, insertion_code

    keys = sorted(residues, key=residue_sort_key)
    total_charge = sum(
        sidechain_charges.get(str(residues[key]["name"]), 0)
        for key in keys
    )

    if include_n_term:
        for key in keys:
            residue = residues[key]
            atoms = residue["atoms"]
            assert isinstance(atoms, set)
            if residue["name"] not in neutral_caps and "N" in atoms:
                total_charge += 1
                break

    if include_c_term:
        for key in reversed(keys):
            residue = residues[key]
            atoms = residue["atoms"]
            assert isinstance(atoms, set)
            if residue["name"] not in neutral_caps and "OXT" in atoms:
                total_charge -= 1
                break

    return total_charge


def read_remarks(pdb_file: Path) -> dict[str, int | str]:
    """Read REMARK key/value pairs from a PDB file."""
    remarks: dict[str, int | str] = {}

    with Path(pdb_file).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("REMARK"):
                continue

            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue

            key = parts[1]
            value: int | str = parts[2].strip()
            if "charge" in key:
                value = int(value)
            remarks[key] = value

    return remarks


def add_element_symbols(pdb_file: Path) -> Path:
    """Populate element symbols when possible, without requiring Open Babel."""
    new_pdb = pdb_file.with_name(f"{pdb_file.stem}_el.pdb")

    if new_pdb.exists():
        return new_pdb

    obabel = shutil.which("obabel")
    if obabel is not None and os.access(obabel, os.X_OK):
        subprocess.run(
            [obabel, str(pdb_file), "-O", str(new_pdb), "--addelement"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return new_pdb

    # Comet may expose an obabel path that is not executable.  Copy the PDB
    # instead and let mda_to_ase infer any missing element symbols.
    shutil.copy2(pdb_file, new_pdb)
    print(
        "Open Babel is unavailable or not executable; "
        "using MDAnalysis/PDB atom-name element inference instead.",
        flush=True,
    )
    return new_pdb


# ============================================================
# ASE / MACE helpers
# ============================================================


def _infer_element_symbol(atom) -> str:
    """Return an ASE-compatible element symbol for one MDAnalysis atom."""
    element = str(getattr(atom, "element", "") or "").strip()
    if element:
        return element[0].upper() + element[1:].lower()

    atom_name = re.sub(
        r"[^A-Za-z]",
        "",
        str(getattr(atom, "name", "")),
    ).upper()
    resname = str(getattr(atom, "resname", "") or "").upper()

    if not atom_name:
        raise ValueError(f"Cannot infer the element for atom {atom!r}")

    two_letter_ions = {
        "BR", "CL", "NA", "MG", "CA", "ZN", "FE",
        "MN", "CU", "CO", "NI",
    }
    if resname in two_letter_ions and atom_name.startswith(resname):
        return resname[0] + resname[1].lower()

    # Protein atom names such as CA, CB and CD are carbon atoms.
    return atom_name[0]


def mda_to_ase(mda_atoms: mda.core.groups.AtomGroup) -> Atoms:
    """Convert an MDAnalysis AtomGroup into an ASE Atoms object."""
    symbols = [_infer_element_symbol(atom) for atom in mda_atoms]
    return Atoms(symbols=symbols, positions=mda_atoms.positions.copy())


def get_ase_atoms_from_files(
    receptor_pdb: Path,
    ligand_sdf: Path,
) -> dict[Literal["a", "b"], Atoms]:
    """Load receptor and ligand structures as ASE objects."""
    receptor_universe = mda.Universe(str(receptor_pdb))
    receptor_atoms = mda_to_ase(receptor_universe.atoms)

    ligand_atoms = ase_read(str(ligand_sdf), index=0)
    if ligand_atoms is None or len(ligand_atoms) == 0:
        raise ValueError(f"Empty ligand: {ligand_sdf}")

    return {"a": receptor_atoms, "b": ligand_atoms}


def get_mlp_energy(
    atoms: Atoms,
    calc,
    total_charge: int,
) -> unit.Quantity:
    """Calculate one MACE energy and return it in kcal/mol."""
    atoms.info["charge"] = total_charge
    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    energy = energy_ev * EV_TO_KCALMOL * ENERGY_UNIT
    return energy.to(ENERGY_UNIT)


def get_mlp_energies_from_files(
    receptor_pdb: Path,
    ligand_sdf: Path,
    calc,
) -> dict[str, unit.Quantity]:
    """Calculate receptor, ligand, complex and interaction energies."""
    molecules = get_ase_atoms_from_files(receptor_pdb, ligand_sdf)
    remarks = read_remarks(receptor_pdb)

    try:
        receptor_charge = int(remarks["charge_a"])
        ligand_charge = int(remarks["charge_b"])
    except KeyError as exc:
        raise KeyError(
            f"Missing {exc.args[0]!r} REMARK in receptor PDB: {receptor_pdb}"
        ) from exc

    print("Calculating receptor energy...", flush=True)
    receptor_energy = get_mlp_energy(
        molecules["a"],
        calc,
        receptor_charge,
    )
    print(f"Receptor energy: {receptor_energy.m:.6f} kcal/mol", flush=True)

    print("Calculating ligand energy...", flush=True)
    ligand_energy = get_mlp_energy(
        molecules["b"],
        calc,
        ligand_charge,
    )
    print(f"Ligand energy: {ligand_energy.m:.6f} kcal/mol", flush=True)

    print("Calculating complex energy...", flush=True)
    complex_atoms = molecules["a"] + molecules["b"]
    complex_energy = get_mlp_energy(
        complex_atoms,
        calc,
        receptor_charge + ligand_charge,
    )
    print(f"Complex energy: {complex_energy.m:.6f} kcal/mol", flush=True)

    interaction_energy = (
        complex_energy - receptor_energy - ligand_energy
    ).to(ENERGY_UNIT)
    print(
        f"Interaction energy: {interaction_energy.m:.6f} kcal/mol",
        flush=True,
    )

    return {
        "receptor_energy": receptor_energy,
        "ligand_energy": ligand_energy,
        "complex_energy": complex_energy,
        "interaction_energy": interaction_energy,
    }


# ============================================================
# PL-REX batch driver
# ============================================================


def iter_plrex_complex_dirs(target_dir: Path) -> list[Path]:
    base = target_dir / "structures_pl-rex"
    if not base.exists():
        return []

    return [
        directory
        for directory in sorted(base.iterdir())
        if directory.is_dir()
        and (directory / "receptor.pdb").exists()
        and (directory / "ligand.sdf").exists()
    ]


def process_plrex_target(
    target_dir: Path,
    out_csv: Path,
    calc,
    exclude: set[str],
    ligand_charge: int = 0,
) -> None:
    """Calculate component energies for each PL-REX complex."""
    complex_dirs = iter_plrex_complex_dirs(target_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    for complex_dir in tqdm(complex_dirs, desc=target_dir.name):
        complex_id = complex_dir.name
        receptor = complex_dir / "receptor.pdb"
        ligand = complex_dir / "ligand.sdf"

        try:
            receptor_charge = compute_pdb_charge(receptor)
            receptor_with_elements = add_element_symbols(receptor)
            receptor_fixed = export_fixed_pdb(
                receptor_with_elements,
                exclude=exclude,
                charge_a=receptor_charge,
                charge_b=ligand_charge,
            )

            energies = get_mlp_energies_from_files(
                receptor_fixed,
                ligand,
                calc,
            )

            rows.append(
                {
                    "target": target_dir.name,
                    "complex_id": complex_id,
                    "charge_a": receptor_charge,
                    "charge_b": ligand_charge,
                    "receptor_energy_kcal_mol": energies[
                        "receptor_energy"
                    ].m,
                    "ligand_energy_kcal_mol": energies["ligand_energy"].m,
                    "complex_energy_kcal_mol": energies["complex_energy"].m,
                    "interaction_energy_kcal_mol": energies[
                        "interaction_energy"
                    ].m,
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "target": target_dir.name,
                    "complex_id": complex_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )

    if not rows:
        raise ValueError(f"No valid PL-REX complexes found under {target_dir}")

    fieldnames = sorted({key for row in rows for key in row})
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {out_csv}")


# ============================================================
# Single local/HPC complex driver
# ============================================================


def extract_receptor_from_complex(
    complex_pdb: Path,
    receptor_out: Path,
    ligand_resnames: set[str],
    exclude: set[str],
) -> Path:
    """Extract a receptor-only PDB from a protein-ligand complex."""
    universe = mda.Universe(str(complex_pdb))

    excluded_resnames = set(exclude) | set(ligand_resnames)
    if excluded_resnames:
        selection = "not resname " + " ".join(sorted(excluded_resnames))
        receptor_atoms = universe.select_atoms(selection)
    else:
        receptor_atoms = universe.atoms

    if len(receptor_atoms) == 0:
        raise ValueError(
            "The receptor selection is empty. Check --ligand-resnames and "
            "the residue names in the complex PDB."
        )

    receptor_out.parent.mkdir(parents=True, exist_ok=True)
    receptor_atoms.write(str(receptor_out))
    return receptor_out


def process_local_complex(
    complex_pdb: Path,
    ligand_sdf: Path,
    out_csv: Path,
    calc,
    ligand_resnames: set[str],
    exclude: set[str],
    ligand_charge: int = 0,
) -> None:
    """Calculate and save all energy components for one complex."""
    complex_pdb = complex_pdb.resolve()
    ligand_sdf = ligand_sdf.resolve()
    out_csv = out_csv.resolve()

    if not complex_pdb.exists():
        raise FileNotFoundError(f"Complex PDB not found: {complex_pdb}")
    if not ligand_sdf.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {ligand_sdf}")

    workdir = complex_pdb.parent
    receptor_raw = workdir / f"{complex_pdb.stem}_receptor_only.pdb"

    extract_receptor_from_complex(
        complex_pdb=complex_pdb,
        receptor_out=receptor_raw,
        ligand_resnames=ligand_resnames,
        exclude=exclude,
    )

    receptor_with_elements = add_element_symbols(receptor_raw)
    receptor_charge = compute_pdb_charge(receptor_with_elements)

    receptor_fixed = export_fixed_pdb(
        receptor_with_elements,
        exclude=set(),
        charge_a=receptor_charge,
        charge_b=ligand_charge,
    )

    energies = get_mlp_energies_from_files(
        receptor_fixed,
        ligand_sdf,
        calc,
    )

    row = {
        "complex_pdb": str(complex_pdb),
        "receptor_pdb": str(receptor_fixed),
        "ligand_sdf": str(ligand_sdf),
        "receptor_charge": receptor_charge,
        "ligand_charge": ligand_charge,
        "receptor_energy_kcal_mol": energies["receptor_energy"].m,
        "ligand_energy_kcal_mol": energies["ligand_energy"].m,
        "complex_energy_kcal_mol": energies["complex_energy"].m,
        "interaction_energy_kcal_mol": energies["interaction_energy"].m,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"Receptor-only PDB: {receptor_fixed}")
    print(f"Receptor energy:    {energies['receptor_energy'].m:.6f} kcal/mol")
    print(f"Ligand energy:      {energies['ligand_energy'].m:.6f} kcal/mol")
    print(f"Complex energy:     {energies['complex_energy'].m:.6f} kcal/mol")
    print(
        "Interaction energy: "
        f"{energies['interaction_energy'].m:.6f} kcal/mol"
    )
    print(f"Saved results to:   {out_csv}")


# ============================================================
# Command-line interface
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate MACE-OFF receptor, ligand, complex and interaction "
            "energies using a complex PDB and a separate ligand SDF."
        )
    )
    parser.add_argument(
        "--complex",
        type=Path,
        default=Path("complex.pdb"),
        help="Protein-ligand complex PDB. Default: ./complex.pdb",
    )
    parser.add_argument(
        "--ligand",
        type=Path,
        default=Path("ligand.sdf"),
        help="Ligand SDF in the bound pose. Default: ./ligand.sdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mace_off_interaction.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--ligand-charge",
        type=int,
        default=0,
        help="Formal charge of the ligand. Default: 0",
    )
    parser.add_argument(
        "--ligand-resnames",
        default="LIG,UNL",
        help=(
            "Comma-separated ligand residue names to remove from the complex "
            "PDB. Default: LIG,UNL"
        ),
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="MACE-OFF model size/name. Default: medium",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="MACE device. Default: cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ligand_resnames = {
        name.strip()
        for name in args.ligand_resnames.split(",")
        if name.strip()
    }
    excluded_solvent_and_ions = {
        "HOH",
        "WAT",
        "NA",
        "K",
        "CL",
        "MG",
        "CA",
    }

    print(f"Loading MACE-OFF model '{args.model}' on {args.device}...", flush=True)
    calculator = mace_off(args.model, device=args.device)

    process_local_complex(
        complex_pdb=args.complex,
        ligand_sdf=args.ligand,
        out_csv=args.output,
        calc=calculator,
        ligand_resnames=ligand_resnames,
        exclude=excluded_solvent_and_ions,
        ligand_charge=args.ligand_charge,
    )


if __name__ == "__main__":
    main()