from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re
import subprocess
import shutil
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------
# Find ligand SDF files
# ----------------------------
def find_ligand_sdf_files(root: Path, name_filter: str | None = None) -> List[Path]:
    """
    Find ligand SDFs under root (case-insensitive), excluding AMBER folders.

    Matches by default:
      ligand.sdf
      Ligand.sdf
      ligand.SDF
      anything containing 'ligand' and ending in .sdf
    """
    sdf_files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if "AMBER" in p.parts:
            continue

        name = p.name.lower()
        if name.endswith(".sdf") and "ligand" in name:
            sdf_files.append(p)

    sdf_files = sorted(sdf_files)
    if name_filter:
        pat = re.compile(name_filter, re.IGNORECASE)
        sdf_files = [p for p in sdf_files if pat.search(p.name)]
    return sdf_files


def safe_stem(p: Path) -> str:
    return p.name[:-4] if p.name.lower().endswith(".sdf") else p.stem


def sanitize_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def get_int_prop(mol, key: str, default: int = 0) -> int:
    """
    Read integer-like property from SDF field, e.g.
      >  <charge>
      0
    Handles "0", "+1", "-1", "1.0".
    """
    if not mol.HasProp(key):
        return default
    raw = mol.GetProp(key).strip()
    try:
        return int(raw)
    except ValueError:
        try:
            return int(round(float(raw)))
        except ValueError:
            return default


# ----------------------------
# SDF -> XYZ (RDKit) + capture charge
# ----------------------------
def sdf_to_xyz_with_charge(
    sdf_path: Path,
    xyz_out_dir: Path,
    remove_hs: bool = False,
    charge_prop: str = "charge",
    prefix_from_path: bool = True,
) -> Tuple[int, List[Tuple[Path, int]]]:
    """
    Convert all molecules in an SDF file to XYZ.
    Returns (n_total, written) where written is list of (xyz_path, charge).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as e:
        raise RuntimeError(
            "RDKit is required for ligand.sdf reading and XYZ writing. "
            "Install via conda (recommended)."
        ) from e

    xyz_out_dir.mkdir(parents=True, exist_ok=True)

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=remove_hs)
    n_total = 0
    written: List[Tuple[Path, int]] = []

    if prefix_from_path:
        rel = sdf_path.parent.as_posix().replace("/", "_")
        base_prefix = f"{rel}__{safe_stem(sdf_path)}"
    else:
        base_prefix = safe_stem(sdf_path)

    for i, mol in enumerate(suppl):
        n_total += 1
        if mol is None:
            continue

        charge = get_int_prop(mol, charge_prop, default=0)

        mol_name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
        mol_name = sanitize_token(mol_name) if mol_name else ""

        if mol_name:
            out_name = f"{base_prefix}__{i:04d}__{mol_name}.xyz"
        else:
            out_name = f"{base_prefix}__{i:04d}.xyz"

        xyz_path = xyz_out_dir / out_name

        # Ensure conformer exists
        if mol.GetNumConformers() == 0:
            mol2 = Chem.AddHs(mol)
            ok = AllChem.EmbedMolecule(mol2, AllChem.ETKDG())
            if ok != 0:
                continue
            AllChem.UFFOptimizeMolecule(mol2, maxIters=200)
            mol = mol2

        Chem.MolToXYZFile(mol, str(xyz_path))
        written.append((xyz_path, charge))

    return n_total, written


# ----------------------------
# XYZ -> MOP
# ----------------------------
def xyz_to_mop(
    xyz_path: Path,
    mop_path: Path,
    method: str = "PM7",
    keywords: str = "1SCF",
    charge: int = 0,
    multiplicity: str = "SINGLET",
    eps: float | None = None,
    two_blank_lines: bool = True,
) -> None:
    """
    Convert an XYZ file to a MOPAC .mop input file.
    Keyword line format:
        PM7 1SCF CHARGE=0 SINGLET EPS=78.4
    """
    lines = xyz_path.read_text().splitlines(True)
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {xyz_path}")

    coord_lines = lines[2:]  # skip atom count + comment

    eps_kw = ""
    if eps is not None:
        eps_str = str(int(eps)) if float(eps).is_integer() else str(eps)
        eps_kw = f" EPS={eps_str}"

    header = f"{method} {keywords} CHARGE={charge} {multiplicity}{eps_kw}\n"

    if two_blank_lines:
        title = "\n"
        comment = "\n"
    else:
        title = xyz_path.stem + "\n"
        comment = "Generated automatically from XYZ\n"

    mop_path.parent.mkdir(parents=True, exist_ok=True)
    with mop_path.open("w") as f:
        f.write(header)
        f.write(title)
        f.write(comment)
        f.writelines(coord_lines)


# ----------------------------
# Run MOPAC
# ----------------------------
def run_one_mopac(mop_file: Path, mopac_exe: str = "mopac", skip_if_out_exists: bool = True) -> tuple[Path, int, str]:
    out_file = mop_file.with_suffix(".out")
    if skip_if_out_exists and out_file.exists() and out_file.stat().st_size > 0:
        return mop_file, 0, "SKIPPED (out exists)"

    proc = subprocess.run(
        [mopac_exe, mop_file.name],
        cwd=str(mop_file.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    err_tail = (proc.stderr or "").strip()[-800:]
    return mop_file, proc.returncode, err_tail


def run_all_mopac_jobs(mop_root: Path, mopac_exe: str = "mopac", max_workers: int = 1, skip_if_out_exists: bool = True) -> None:
    if shutil.which(mopac_exe) is None:
        raise RuntimeError(f"Cannot find '{mopac_exe}' on PATH. Set --mopac-exe to full path if needed.")

    mop_files = sorted(mop_root.rglob("*.mop"))
    if not mop_files:
        print(f"No .mop files found under: {mop_root}")
        return

    print(f"\nFound {len(mop_files)} MOPAC input files under {mop_root}")
    print(f"Running MOPAC with max_workers={max_workers}, skip_if_out_exists={skip_if_out_exists}\n")

    failed: list[tuple[Path, int, str]] = []

    if max_workers <= 1:
        for i, f in enumerate(mop_files, 1):
            mop_file, rc, err = run_one_mopac(f, mopac_exe=mopac_exe, skip_if_out_exists=skip_if_out_exists)
            print(f"[{i}/{len(mop_files)}] rc={rc}  {mop_file}")
            if rc != 0:
                failed.append((mop_file, rc, err))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one_mopac, f, mopac_exe, skip_if_out_exists) for f in mop_files]
            done = 0
            for fut in as_completed(futs):
                done += 1
                mop_file, rc, err = fut.result()
                print(f"[{done}/{len(mop_files)}] rc={rc}  {mop_file}")
                if rc != 0:
                    failed.append((mop_file, rc, err))

    print("\n=== MOPAC run summary ===")
    print(f"Total jobs: {len(mop_files)}")
    print(f"Failed   : {len(failed)}")

    if failed:
        print("\nFirst few failures:")
        for mop_file, rc, err in failed[:10]:
            print(f"\n- {mop_file}\n  rc={rc}\n  stderr_tail:\n{err}")


# ----------------------------
# Extract HoF kcal/mol and delta
# ----------------------------
HOF_RE = re.compile(
    r"HEAT OF FORMATION\s*=\s*([-+]?\d+(?:\.\d+)?)\s*KCAL/MOL",
    re.IGNORECASE,
)


def extract_heat_of_formation_kcalmol(out_file: Path) -> float:
    text = out_file.read_text(errors="ignore")
    matches = HOF_RE.findall(text)
    if not matches:
        raise ValueError(f"HEAT OF FORMATION not found in: {out_file}")
    return float(matches[-1])


def collect_hof_map(root: Path) -> dict[Path, float]:
    hof_map: dict[Path, float] = {}
    for out_file in sorted(root.rglob("*.out")):
        key = out_file.relative_to(root).with_suffix("")  # path relative to EPS_* folder, without suffix
        hof_map[key] = extract_heat_of_formation_kcalmol(out_file)
    return hof_map


def write_delta_hof_csv(
    mop_out: Path,
    eps1_folder: str = "EPS_1",
    eps784_folder: str = "EPS_78_4",
    csv_name: str = "delta_hof_eps78_4_minus_eps1.csv",
) -> Path:
    eps1_dir = mop_out / eps1_folder
    eps784_dir = mop_out / eps784_folder

    if not eps1_dir.exists():
        raise FileNotFoundError(f"Missing folder: {eps1_dir}")
    if not eps784_dir.exists():
        raise FileNotFoundError(f"Missing folder: {eps784_dir}")

    hof_1 = collect_hof_map(eps1_dir)
    hof_784 = collect_hof_map(eps784_dir)

    all_keys = sorted(set(hof_1) | set(hof_784))
    out_csv = mop_out / csv_name

    rows = []
    paired_ok = 0
    missing = 0

    for key in all_keys:
        v1 = hof_1.get(key)
        v784 = hof_784.get(key)

        if v1 is None or v784 is None:
            missing += 1
            rows.append(
                {
                    "key": key.as_posix(),
                    "hof_eps_1_kcalmol": "" if v1 is None else v1,
                    "hof_eps_78_4_kcalmol": "" if v784 is None else v784,
                    "delta_78_4_minus_1_kcalmol": "",
                    "status": "MISSING_PAIR",
                }
            )
            continue

        paired_ok += 1
        rows.append(
            {
                "key": key.as_posix(),
                "hof_eps_1_kcalmol": v1,
                "hof_eps_78_4_kcalmol": v784,
                "delta_78_4_minus_1_kcalmol": v784 - v1,
                "status": "OK",
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "hof_eps_1_kcalmol",
                "hof_eps_78_4_kcalmol",
                "delta_78_4_minus_1_kcalmol",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Delta extraction summary ===")
    print(f"EPS_1 .out files   : {len(hof_1)}")
    print(f"EPS_78_4 .out files: {len(hof_784)}")
    print(f"Paired OK          : {paired_ok}")
    print(f"Missing pairs      : {missing}")
    print(f"Wrote CSV          : {out_csv}")

    return out_csv


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ligand pipeline: ligand.sdf -> XYZ -> MOP (EPS=1 and 78.4), optional MOPAC, optional delta."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder to search for ligand.sdf under",
    )
    parser.add_argument("--xyz-out", type=str, default="xyz_out_ligand", help="Output folder for ligand XYZ files")
    parser.add_argument("--mop-out", type=str, default="mopac_inputs_ligand", help="Output folder for ligand MOP files")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help=r"Optional regex filter applied to SDF filenames (e.g. 'ligand\.sdf$')",
    )

    parser.add_argument("--remove-hs", action="store_true", help="Remove hydrogens when reading SDF")
    parser.add_argument("--charge-prop", type=str, default="charge", help="SDF property name storing charge (default: charge)")

    # MOPAC run options (kept for your job.sh compatibility)
    parser.add_argument("--run-mopac", action="store_true", help="Run MOPAC on generated .mop files")
    parser.add_argument("--mopac-exe", type=str, default="mopac", help="MOPAC executable name or full path")
    parser.add_argument("--workers", type=int, default=1, help="Parallel MOPAC jobs (1 = sequential)")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip jobs that already have .out")

    # Delta extraction (HoF delta between EPS folders)
    parser.add_argument(
        "--extract-delta",
        action="store_true",
        help="Extract HEAT OF FORMATION from .out and compute delta (EPS_78.4 - EPS_1) into CSV",
    )
    parser.add_argument("--delta-csv", type=str, default="delta_hof_eps78_4_minus_eps1.csv", help="CSV filename to write under --mop-out")

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    xyz_out = Path(args.xyz_out).expanduser().resolve()
    mop_out = Path(args.mop_out).expanduser().resolve()

    ligand_sdfs = find_ligand_sdf_files(root, args.filter)
    if not ligand_sdfs:
        print(f"No ligand SDFs found under: {root}")
        print("Tip: check --root points to the PL-REX target folder that contains ligand.sdf files.")
        return

    print(f"\nFound {len(ligand_sdfs)} ligand SDF files under: {root}")
    print(f"Writing ligand XYZ to: {xyz_out}")
    print(f"Writing ligand MOP to: {mop_out} (EPS_1 and EPS_78_4)")
    print(f"Reading charge from SDF property: {args.charge_prop}")

    eps_values = [1.0, 78.4]

    total_sdf_mols = 0
    total_xyz = 0
    total_mop = 0

    for sdf in ligand_sdfs:
        n_total, written = sdf_to_xyz_with_charge(
            sdf_path=sdf,
            xyz_out_dir=xyz_out,
            remove_hs=args.remove_hs,
            charge_prop=args.charge_prop,
            prefix_from_path=True,
        )
        total_sdf_mols += n_total
        total_xyz += len(written)

        # XYZ -> MOP (two EPS folders)
        for xyz_path, charge in written:
            for eps in eps_values:
                eps_tag = str(int(eps)) if float(eps).is_integer() else str(eps).replace(".", "_")
                mop_path = (mop_out / f"EPS_{eps_tag}" / xyz_path.relative_to(xyz_out)).with_suffix(".mop")

                # You can customize keywords per EPS if desired; leaving symmetric for ligands.
                kw = "1SCF"

                xyz_to_mop(
                    xyz_path=xyz_path,
                    mop_path=mop_path,
                    method="PM7",
                    keywords=kw,
                    charge=charge,
                    multiplicity="SINGLET",
                    eps=eps,
                    two_blank_lines=True,
                )
                total_mop += 1

        print(f"- {sdf}: molecules={n_total}, xyz={len(written)}, mop={len(written) * len(eps_values)}")

    print("\nDone generating ligand inputs.")
    print(f"Total molecules seen : {total_sdf_mols}")
    print(f"Total XYZ written    : {total_xyz}")
    print(f"Total MOP written    : {total_mop}")

    if args.run_mopac:
        run_all_mopac_jobs(
            mop_root=mop_out,
            mopac_exe=args.mopac_exe,
            max_workers=args.workers,
            skip_if_out_exists=(not args.no_skip),
        )

    if args.extract_delta:
        write_delta_hof_csv(
            mop_out=mop_out,
            eps1_folder="EPS_1",
            eps784_folder="EPS_78_4",
            csv_name=args.delta_csv,
        )


if __name__ == "__main__":
    main()
