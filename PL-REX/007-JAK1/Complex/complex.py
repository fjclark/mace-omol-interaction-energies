from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv


# ----------------------------
# Git: clone + sparse checkout
# ----------------------------
def ensure_pl_rex_sparse_checkout() -> Path:
    """
    Clone PL-REX if missing; otherwise reuse existing repo.
    Ensure sparse checkout for 009-CDK2/structures_pl-rex.
    Returns the local path to structures_pl-rex.
    """
    repo_dir = Path("PL-REX").resolve()

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/Honza-R/PL-REX.git", str(repo_dir)],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)

    subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=str(repo_dir), check=False)
    subprocess.run(
        ["git", "sparse-checkout", "set", "007-JAK1/structures_pl-rex"],
        cwd=str(repo_dir),
        check=True,
    )

    return repo_dir / "007-JAK1" / "structures_pl-rex"


# ----------------------------
# Find receptor.pdb (exclude AMBER)
# ----------------------------
def find_receptor_pdb_files(root: Path) -> List[Path]:
    """
    Find receptor PDBs under root (case-insensitive), excluding AMBER folders.
    """
    receptor_files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if "AMBER" in p.parts:
            continue

        name = p.name.lower()
        if name.endswith(".pdb") and "receptor" in name:
            receptor_files.append(p)

    return sorted(receptor_files)


# ----------------------------
# Find receptor.pdb + ligand.sdf pairs (exclude AMBER)
# ----------------------------
def find_complex_pairs(root: Path, ligand_name: str = "ligand.sdf") -> List[Tuple[Path, Path]]:
    """
    Find (receptor.pdb, ligand.sdf) pairs under root, excluding AMBER folders.
    Assumes receptor.pdb and ligand.sdf live in the same directory.
    """
    pairs: List[Tuple[Path, Path]] = []
    receptor_pdbs = find_receptor_pdb_files(root)

    for rec in receptor_pdbs:
        if "AMBER" in rec.parts:
            continue
        lig = rec.parent / ligand_name
        if lig.exists() and lig.is_file() and "AMBER" not in lig.parts:
            pairs.append((rec, lig))

    return pairs


# ----------------------------
# Ligand charge from SDF property block
# ----------------------------
def ligand_charge_from_sdf_property(sdf_path: Path, prop_name: str = "charge") -> int:
    """
    Reads ligand charge from an SDF property block like:

      >  <charge>
      0

    - Supports multi-molecule SDFs: returns the first molecule's charge found.
    - Charge is interpreted as integer; e.g. "0", "-1", "2".
    """
    prop_re = re.compile(rf"^\s*>\s*<\s*{re.escape(prop_name)}\s*>\s*$", re.IGNORECASE)

    with sdf_path.open("r", errors="ignore") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if prop_re.match(line):
            # next non-empty line is the value (SDF convention)
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j >= len(lines):
                break

            val_str = lines[j].strip()
            if val_str == "$$$$":
                break

            try:
                # allow "0" or "0.0"
                val_float = float(val_str)
                if abs(val_float - round(val_float)) > 1e-8:
                    raise ValueError(f"Non-integer charge value: {val_str}")
                return int(round(val_float))
            except Exception as e:
                raise ValueError(f"Failed to parse <{prop_name}> from {sdf_path}: '{val_str}' ({e})")

    raise ValueError(f"Could not find SDF property <{prop_name}> in: {sdf_path}")


# ----------------------------
# Receptor charge from PDB by residue counting
# ----------------------------
def _pdb_iter_residues(pdb_path: Path) -> tuple[set[tuple[str, int, str, str]], dict[str, list[tuple[int, str]]]]:
    """
    Returns:
      residues: set of (chain, resseq, icode, resname) from ATOM/HETATM records
      chain_positions: dict[chain] -> list of (resseq, icode) for termini inference
    """
    residues: set[tuple[str, int, str, str]] = set()
    chain_positions: dict[str, list[tuple[int, str]]] = {}

    with pdb_path.open("r", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            if len(line) < 27:
                continue

            resname = line[17:20].strip().upper()
            chain = (line[21].strip() or "_")
            resseq_str = line[22:26].strip()
            icode = line[26].strip() or ""

            try:
                resseq = int(resseq_str)
            except ValueError:
                continue

            key = (chain, resseq, icode, resname)
            residues.add(key)
            chain_positions.setdefault(chain, []).append((resseq, icode))

    return residues, chain_positions


def receptor_charge_from_pdb(
    pdb_path: Path,
    his_mode: str = "auto",        # "auto" | "0" | "1"
    include_termini: bool = True,  # add +1 N-terminus and -1 C-terminus per chain
) -> int:
    """
    Estimate receptor net charge from PDB by summing residue charges using residue names.

    Rules:
      - ASP/GLU: -1
      - LYS/ARG: +1
      - Histidine:
          * his_mode="auto": HIS=0, HID/HIE=0, HIP/HSP=+1
          * his_mode="0": all HIS variants treated as 0
          * his_mode="1": all HIS variants treated as +1
      - Common neutral protonated forms supported:
          ASH/GLH treated as 0; LYN treated as 0 (neutral Lys)
    Termini:
      - If include_termini: +1 and -1 per chain (simple approximation)
    """
    residues, chain_positions = _pdb_iter_residues(pdb_path)

    base_charge: dict[str, int] = {
        "ASP": -1,
        "GLU": -1,
        "LYS": +1,
        "ARG": +1,
        # neutral variants used by some pipelines
        "ASH": 0,   # protonated ASP
        "GLH": 0,   # protonated GLU
        "LYN": 0,   # neutral LYS
        "ARN": 0,   # rare neutral ARG naming
    }

    his_mode_norm = his_mode.strip().lower()
    if his_mode_norm not in {"auto", "0", "1"}:
        raise ValueError("--his-mode must be one of: auto, 0, 1")

    total = 0
    for (_chain, _resseq, _icode, resname) in residues:
        if resname in base_charge:
            total += base_charge[resname]
            continue

        if resname in {"HIS", "HID", "HIE", "HIP", "HSP"}:
            if his_mode_norm == "0":
                total += 0
            elif his_mode_norm == "1":
                total += 1
            else:
                total += 1 if resname in {"HIP", "HSP"} else 0
            continue

    if include_termini:
        # +1 N-terminus and -1 C-terminus per chain (net 0 per chain)
        for _chain, pos_list in chain_positions.items():
            if not pos_list:
                continue
            total += 1
            total -= 1

    return total


# ----------------------------
# Any -> XYZ (OpenBabel)
# ----------------------------
def any_to_xyz_openbabel(
    in_path: Path,
    xyz_path: Path,
    obabel_exe: str = "obabel",
    extra_args: List[str] | None = None,
) -> bool:
    """
    Convert any input supported by OpenBabel (e.g., PDB, SDF) -> XYZ.
    Returns True if success, False otherwise.
    """
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [obabel_exe, str(in_path), "-O", str(xyz_path)]
    if extra_args:
        cmd[1:1] = extra_args  # insert after obabel_exe

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[OpenBabel ERROR] {in_path}\n{proc.stderr.strip()}")
        return False
    return True


# ----------------------------
# Merge receptor XYZ + ligand XYZ -> complex XYZ
# ----------------------------
def merge_two_xyz(rec_xyz: Path, lig_xyz: Path, complex_xyz: Path) -> None:
    """
    Merge two XYZ files into one XYZ file: (receptor + ligand).
    """
    r_lines = rec_xyz.read_text().splitlines()
    l_lines = lig_xyz.read_text().splitlines()

    if len(r_lines) < 3:
        raise ValueError(f"Invalid XYZ file: {rec_xyz}")
    if len(l_lines) < 3:
        raise ValueError(f"Invalid XYZ file: {lig_xyz}")

    r_coords = r_lines[2:]
    l_coords = l_lines[2:]

    n_total = len(r_coords) + len(l_coords)

    complex_xyz.parent.mkdir(parents=True, exist_ok=True)
    with complex_xyz.open("w") as f:
        f.write(f"{n_total}\n")
        f.write(f"COMPLEX: {rec_xyz.stem} + {lig_xyz.stem}\n")
        f.write("\n".join(r_coords) + "\n")
        f.write("\n".join(l_coords) + "\n")


# ----------------------------
# XYZ -> MOP
# ----------------------------
def xyz_to_mop(
    xyz_path: Path,
    mop_path: Path,
    method: str = "PM7",
    keywords: str = "1SCF MOZYME",
    charge: int = 0,
    eps: float | None = None,
    two_blank_lines: bool = True,
) -> None:
    """
    Convert an XYZ file to a MOPAC .mop input file.
    If eps is None, EPS=... is omitted from the keyword line.
    """
    lines = xyz_path.read_text().splitlines(True)
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file: {xyz_path}")

    coord_lines = lines[2:]  # skip atom count + comment

    eps_kw = ""
    if eps is not None:
        eps_str = str(int(eps)) if float(eps).is_integer() else str(eps)
        eps_kw = f" EPS={eps_str}"

    header = f"{method} {keywords} CHARGE={charge}{eps_kw}\n"

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
        key = out_file.relative_to(root).with_suffix("")
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
            rows.append({
                "key": key.as_posix(),
                "hof_eps_1_kcalmol": "" if v1 is None else v1,
                "hof_eps_78_4_kcalmol": "" if v784 is None else v784,
                "delta_78_4_minus_1_kcalmol": "",
                "status": "MISSING_PAIR",
            })
            continue

        paired_ok += 1
        rows.append({
            "key": key.as_posix(),
            "hof_eps_1_kcalmol": v1,
            "hof_eps_78_4_kcalmol": v784,
            "delta_78_4_minus_1_kcalmol": v784 - v1,
            "status": "OK",
        })

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
        description=(
            "Build receptor+ligand complex (receptor.pdb + ligand.sdf) -> XYZ -> MOP (EPS=1 and 78.4), "
            "optional MOPAC run, optional delta HoF extraction.\n"
            "Complex charge is computed as: receptor_charge(from PDB residues) + ligand_charge(from SDF property <charge>), "
            "unless overridden by --complex-charge.\n"
            "EPS=1 is omitted from the MOPAC keyword line (i.e., no EPS=1 is written)."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root folder to search (if omitted, auto-clone sparse checkout PL-REX CDK2 structures)",
    )
    parser.add_argument("--xyz-out", type=str, default="xyz_out_complex", help="Output folder for complex XYZ files")
    parser.add_argument("--mop-out", type=str, default="mopac_inputs_complex", help="Output folder for complex MOP files")

    parser.add_argument("--ligand-name", type=str, default="ligand.sdf", help="Ligand filename in each folder (default: ligand.sdf)")

    parser.add_argument(
        "--complex-charge",
        type=int,
        default=None,
        help="Override net charge of receptor+ligand complex. If omitted, receptor charge is estimated from PDB residues and ligand charge is read from SDF property <charge>.",
    )

    # receptor charge model options
    parser.add_argument(
        "--his-mode",
        type=str,
        default="auto",
        help='How to treat histidines for receptor charge: "auto" (HIS/HID/HIE=0, HIP/HSP=+1), "0" (all 0), "1" (all +1). Default: auto.',
    )
    parser.add_argument(
        "--no-termini",
        action="store_true",
        help="Do NOT add N/C termini charges (+1/-1 per chain) in receptor charge estimate.",
    )

    # ligand charge property name
    parser.add_argument(
        "--ligand-charge-prop",
        type=str,
        default="charge",
        help="SDF property name to read ligand charge from (default: charge).",
    )

    # MOPAC run options
    parser.add_argument("--run-mopac", action="store_true", help="Run MOPAC on generated .mop files")
    parser.add_argument("--mopac-exe", type=str, default="mopac", help="MOPAC executable name or full path")
    parser.add_argument("--workers", type=int, default=1, help="Parallel MOPAC jobs (1 = sequential)")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip jobs that already have .out")

    # Delta extraction
    parser.add_argument(
        "--extract-delta",
        action="store_true",
        help="Extract HoF from .out and compute delta (EPS_78.4 - EPS_1) into CSV",
    )
    parser.add_argument(
        "--delta-csv",
        type=str,
        default="delta_hof_eps78_4_minus_eps1.csv",
        help="CSV filename to write under --mop-out",
    )

    # OpenBabel executable name
    parser.add_argument("--obabel-exe", type=str, default="obabel", help="OpenBabel executable (default: obabel)")

    # Optional ligand processing hints (off by default)
    parser.add_argument(
        "--ligand-gen3d",
        action="store_true",
        help="Ask OpenBabel to generate 3D coordinates for the ligand (adds --gen3d when converting ligand.sdf -> xyz)",
    )
    parser.add_argument(
        "--ligand-add-h",
        action="store_true",
        help="Ask OpenBabel to add hydrogens to the ligand (adds -h when converting ligand.sdf -> xyz)",
    )

    # Optional: add PRECISE to improve stability for large systems
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Add PRECISE keyword to MOPAC input (can help convergence for large systems)",
    )

    args = parser.parse_args()

    # Root
    if args.root is None:
        root = ensure_pl_rex_sparse_checkout().resolve()
        print(f"Using auto sparse-checkout root: {root}")
    else:
        root = Path(args.root).expanduser().resolve()
        print(f"Using user root: {root}")

    xyz_out = Path(args.xyz_out).expanduser().resolve()
    mop_out = Path(args.mop_out).expanduser().resolve()

    # Check OpenBabel
    if shutil.which(args.obabel_exe) is None:
        raise RuntimeError(f"Cannot find '{args.obabel_exe}' on PATH. Install OpenBabel or set --obabel-exe.")

    pairs = find_complex_pairs(root, ligand_name=args.ligand_name)
    if not pairs:
        print(f"No (receptor.pdb, {args.ligand_name}) pairs found under: {root}")
        return

    print(f"\nFound {len(pairs)} complex pairs under: {root}")
    print(f"Writing complex XYZ to: {xyz_out}")
    print(f"Writing complex MOP to: {mop_out} (EPS_1 and EPS_78_4)")

    total_xyz = 0
    total_mop = 0

    eps_values = [1.0, 78.4]

    ligand_extra_args: List[str] = []
    if args.ligand_add_h:
        ligand_extra_args.append("-h")
    if args.ligand_gen3d:
        ligand_extra_args.append("--gen3d")

    for rec_pdb, lig_sdf in pairs:
        rel_prefix = rec_pdb.parent.as_posix().replace("/", "_")

        rec_xyz = xyz_out / f"{rel_prefix}__receptor.xyz"
        lig_xyz = xyz_out / f"{rel_prefix}__ligand.xyz"
        complex_xyz = xyz_out / f"{rel_prefix}__complex.xyz"

        ok1 = any_to_xyz_openbabel(rec_pdb, rec_xyz, obabel_exe=args.obabel_exe, extra_args=None)
        ok2 = any_to_xyz_openbabel(lig_sdf, lig_xyz, obabel_exe=args.obabel_exe, extra_args=ligand_extra_args or None)
        if not (ok1 and ok2):
            continue

        merge_two_xyz(rec_xyz, lig_xyz, complex_xyz)
        total_xyz += 1

        # Compute complex charge unless overridden
        if args.complex_charge is None:
            rec_charge = receptor_charge_from_pdb(
                rec_pdb,
                his_mode=args.his_mode,
                include_termini=(not args.no_termini),
            )
            lig_charge = ligand_charge_from_sdf_property(lig_sdf, prop_name=args.ligand_charge_prop)
            complex_charge = rec_charge + lig_charge
        else:
            rec_charge = None
            lig_charge = None
            complex_charge = args.complex_charge

        # Complex XYZ -> MOP (two EPS folders)
        for eps in eps_values:
            eps_tag = str(int(eps)) if float(eps).is_integer() else str(eps).replace(".", "_")
            mop_path = (mop_out / f"EPS_{eps_tag}" / complex_xyz.relative_to(xyz_out)).with_suffix(".mop")

            # EPS=1: NO MOZYME and omit EPS keyword entirely
            # EPS=78.4: WITH MOZYME and include EPS=78.4
            if abs(eps - 1.0) < 1e-6:
                kw = "1SCF"
                eps_for_mop = None   # <-- IMPORTANT: omits "EPS=1"
            elif abs(eps - 78.4) < 1e-6:
                kw = "1SCF MOZYME"
                eps_for_mop = eps
            else:
                kw = "1SCF"
                eps_for_mop = eps

            if args.precise:
                kw += " PRECISE"

            xyz_to_mop(
                xyz_path=complex_xyz,
                mop_path=mop_path,
                method="PM7",
                keywords=kw,
                charge=complex_charge,
                eps=eps_for_mop,   # <-- EPS=1 omitted, EPS=78.4 included
                two_blank_lines=True,
            )
            total_mop += 1

        if args.complex_charge is None:
            print(
                f"- {rec_pdb.parent}: complex_xyz={complex_xyz.name}, mop_written={len(eps_values)} "
                f"(rec_charge={rec_charge}, lig_charge={lig_charge}, complex_charge={complex_charge})"
            )
        else:
            print(
                f"- {rec_pdb.parent}: complex_xyz={complex_xyz.name}, mop_written={len(eps_values)} "
                f"(complex_charge={complex_charge} OVERRIDE)"
            )

    print("\nDone generating inputs.")
    print(f"Total complex XYZ written : {total_xyz}")
    print(f"Total complex MOP written : {total_mop}")

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
