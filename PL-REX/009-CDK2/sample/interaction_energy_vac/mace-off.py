
from __future__ import annotations

from pathlib import Path
import argparse
from typing import Literal
import subprocess
import csv
import traceback

import MDAnalysis as mda
from ase import Atoms
from ase.io import read as ase_read
from openff.units import unit
from tqdm import tqdm

import mace.calculators
from mace.calculators import mace_off
# from mace.calculators import mace_omol

# =========================
# Constants
# =========================

EV_TO_KCALMOL = 23.0605


# =========================
# PDB utilities
# =========================

def export_fixed_pdb(
    pdb_file: Path,
    exclude: set[str] | None = None,
    charge_a: int = 0,
    charge_b: int = 0,
) -> Path:
    """
    Remove solvent/ions and add REMARK charge lines.
    """
    if exclude is None:
        exclude = {"HOH", "WAT", "NA", "K", "CL", "MG", "CA"}

    u = mda.Universe(pdb_file)

    if exclude:
        exclude_str = " ".join(sorted(exclude))
        sel = f"not resname {exclude_str}"
        sel_atoms = u.select_atoms(sel)
    else:
        sel_atoms = u.atoms

    fixed_pdb = pdb_file.with_name(pdb_file.stem + "_fixed.pdb")
    sel_atoms.write(fixed_pdb)

    with open(fixed_pdb, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(f"REMARK charge_a {charge_a}\n")
        f.write(f"REMARK charge_b {charge_b}\n")
        f.write(content)

    return fixed_pdb


def compute_pdb_charge(pdb_path, include_n_term=False, include_c_term=True):
    """
    Simple protein charge model (~pH 7).
    """
    SIDECHAIN_CHARGES = {
        "LYS": +1,
        "ARG": +1,
        "ASP": -1,
        "GLU": -1,
        "HIS":  0,
        "HID":  0,
        "HIE":  0,
        "HIP": +1,
    }

    NEUTRAL_CAPS = {"ACE", "NME", "BNC", "BCC", "BCB"}

    residues = {}

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = line[22:26].strip()
                icode = line[26].strip()
                atom = line[12:16].strip()

                key = (chain, resseq, icode)
                residues.setdefault(key, {"name": resname, "atoms": set()})
                residues[key]["atoms"].add(atom)

    def sort_key(k):
        chain, resseq, icode = k
        try:
            resseq = int(resseq)
        except ValueError:
            resseq = 0
        return (chain, resseq, icode)

    keys = sorted(residues, key=sort_key)

    total = sum(SIDECHAIN_CHARGES.get(residues[k]["name"], 0) for k in keys)

    if include_n_term:
        for k in keys:
            r = residues[k]
            if r["name"] not in NEUTRAL_CAPS and "N" in r["atoms"]:
                total += 1
                break

    if include_c_term:
        for k in reversed(keys):
            r = residues[k]
            if r["name"] not in NEUTRAL_CAPS and "OXT" in r["atoms"]:
                total -= 1
                break

    return total


def read_remarks(pdbfile: Path) -> dict[str, int | str]:
    remarks = {}
    with open(pdbfile) as f:
        for line in f:
            if line.startswith("REMARK"):
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    key, val = parts[1], parts[2].strip()
                    if "charge" in key:
                        val = int(val)
                    remarks[key] = val
    return remarks


def add_element_symbols(pdbfile: Path) -> Path:
    new_pdb = pdbfile.with_name(pdbfile.stem + "_el.pdb")
    if not new_pdb.exists():
        subprocess.run(
            ["obabel", str(pdbfile), "-O", str(new_pdb), "--addelement"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return new_pdb


# =========================
# ASE / MACE helpers
# =========================

def mda_to_ase(mda_atoms: mda.core.groups.AtomGroup) -> Atoms:
    symbols = [a.element for a in mda_atoms]
    positions = mda_atoms.positions
    return Atoms(symbols=symbols, positions=positions)


def get_ase_atoms_from_files(
    receptor_pdb: Path,
    ligand_sdf: Path,
) -> dict[Literal["a", "b"], Atoms]:

    u = mda.Universe(receptor_pdb)
    ase_a = mda_to_ase(u.atoms)

    ase_b = ase_read(str(ligand_sdf), index=0)
    if ase_b is None or len(ase_b) == 0:
        raise ValueError(f"Empty ligand: {ligand_sdf}")

    return {"a": ase_a, "b": ase_b}


def get_mlp_energy(
    atoms: Atoms,
    calc,
    total_charge: int,
) -> unit.Quantity:

    atoms.info["charge"] = total_charge
    atoms.calc = calc

    energy_ev = atoms.get_potential_energy()
    return (
        energy_ev * EV_TO_KCALMOL * unit.kilocalorie / unit.mole
    ).to(unit.kilocalorie / unit.mole)


def get_mlp_interaction_energy_from_files(
    receptor_pdb: Path,
    ligand_sdf: Path,
    calc,
) -> unit.Quantity:

    mols = get_ase_atoms_from_files(receptor_pdb, ligand_sdf)
    remarks = read_remarks(receptor_pdb)

    charge_a = remarks["charge_a"]
    charge_b = remarks["charge_b"]

    e_a = get_mlp_energy(mols["a"], calc, charge_a)
    e_b = get_mlp_energy(mols["b"], calc, charge_b)

    complex_atoms = mols["a"] + mols["b"]
    e_ab = get_mlp_energy(complex_atoms, calc, charge_a + charge_b)

    return (e_ab - (e_a + e_b)).to(unit.kilocalorie / unit.mole)


# =========================
# PL-REX driver
# =========================

def iter_plrex_complex_dirs(target_dir: Path) -> list[Path]:
    base = target_dir / "structures_pl-rex"
    return [
        d for d in sorted(base.iterdir())
        if d.is_dir()
        and (d / "receptor.pdb").exists()
        and (d / "ligand.sdf").exists()
    ]


def process_plrex_target(
    target_dir: Path,
    out_csv: Path,
    calc,
    exclude: set[str],
    ligand_charge: int = 0,
):

    complex_dirs = iter_plrex_complex_dirs(target_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for cdir in tqdm(complex_dirs, desc=target_dir.name):
        cid = cdir.name
        rec = cdir / "receptor.pdb"
        lig = cdir / "ligand.sdf"

        try:
            q_rec = compute_pdb_charge(rec)
            rec_el = add_element_symbols(rec)
            rec_fix = export_fixed_pdb(
                rec_el,
                exclude=exclude,
                charge_a=q_rec,
                charge_b=ligand_charge,
            )

            e_int = get_mlp_interaction_energy_from_files(
                rec_fix, lig, calc
            )

            print(f"{cid}: {e_int.m:.3f} kcal/mol")

            rows.append({
                "target": target_dir.name,
                "complex_id": cid,
                "charge_a": q_rec,
                "charge_b": ligand_charge,
                "interaction_energy_kcal_mol": e_int.m,
            })

        except Exception as e:
            rows.append({
                "target": target_dir.name,
                "complex_id": cid,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            })

    fieldnames = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved results to {out_csv}")


# =========================
# Local single-complex driver
# =========================

def extract_receptor_from_complex(
    complex_pdb: Path,
    receptor_out: Path,
    ligand_resnames: set[str],
    exclude: set[str],
) -> Path:
    """Extract a receptor-only PDB from a protein-ligand complex."""
    u = mda.Universe(str(complex_pdb))

    excluded_resnames = set(exclude) | set(ligand_resnames)
    if excluded_resnames:
        selection = "not resname " + " ".join(sorted(excluded_resnames))
        receptor_atoms = u.select_atoms(selection)
    else:
        receptor_atoms = u.atoms

    if len(receptor_atoms) == 0:
        raise ValueError(
            "The receptor selection is empty. Check --ligand-resnames "
            "and the residue names in the complex PDB."
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

    receptor_el = add_element_symbols(receptor_raw)
    receptor_charge = compute_pdb_charge(receptor_el)

    receptor_fixed = export_fixed_pdb(
        receptor_el,
        exclude=set(),
        charge_a=receptor_charge,
        charge_b=ligand_charge,
    )

    interaction_energy = get_mlp_interaction_energy_from_files(
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
        "interaction_energy_kcal_mol": interaction_energy.m,
    }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"Receptor-only PDB: {receptor_fixed}")
    print(f"Interaction energy: {interaction_energy.m:.3f} kcal/mol")
    print(f"Saved results to: {out_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a MACE-OMOL interaction energy using a complex PDB "
            "and a separate ligand SDF stored locally."
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
        default=Path("results/mace_omol_interaction.csv"),
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
        default="extra_large",
        help="MACE-OMOL model size/name. Default: extra_large",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="MACE device, for example cpu or cuda. Default: cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ligand_resnames = {
        name.strip()
        for name in args.ligand_resnames.split(",")
        if name.strip()
    }
    excluded_solvent_and_ions = {"HOH", "WAT", "NA", "K", "CL", "MG", "CA"}

    calc = mace_off(args.model, device=args.device)

    process_local_complex(
        complex_pdb=args.complex,
        ligand_sdf=args.ligand,
        out_csv=args.output,
        calc=calc,
        ligand_resnames=ligand_resnames,
        exclude=excluded_solvent_and_ions,
        ligand_charge=args.ligand_charge,
    )


if __name__ == "__main__":
    main()
