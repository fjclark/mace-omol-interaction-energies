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
        ["git", "sparse-checkout", "set", "009-CDK2/structures_pl-rex"],
        cwd=str(repo_dir),
        check=True,
    )

    return repo_dir / "009-CDK2" / "structures_pl-rex"


# ----------------------------
# Find receptor.pdb (exclude AMBER)
# ----------------------------
def find_receptor_pdb_files(root: Path) -> List[Path]:
    """
    Find receptor PDBs under root (case-insensitive), excluding AMBER folders.

    Matches:
      receptor.pdb
      Receptor.pdb
      receptor.PDB
      anything containing 'receptor' and ending in .pdb
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
# PDB -> XYZ (OpenBabel)
# ----------------------------
def pdb_to_xyz_openbabel(pdb_path: Path, xyz_path: Path) -> bool:
    """
    Convert PDB -> XYZ using Open Babel (obabel).
    Returns True if success, False otherwise.
    """
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["obabel", str(pdb_path), "-O", str(xyz_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[OpenBabel ERROR] {pdb_path}\n{proc.stderr.strip()}")
        return False
    return True


# ----------------------------
# XYZ -> MOP
# ----------------------------
def xyz_to_mop(
    xyz_path: Path,
    mop_path: Path,
    method: str = "PM7",
    keywords: str = "1SCF MOZYME",
    charge: int = 2,
    eps: float | None = None,
    two_blank_lines: bool = True,
) -> None:
    """
    Convert an XYZ file to a MOPAC .mop input file.
    Keyword line format:
        PM7 1SCF CHARGE=0 EPS=78.4

    If two_blank_lines=True, writes blank title + blank comment (two blank lines).
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
        description="Use receptor.pdb under PL-REX structures: receptor.pdb -> XYZ -> MOP (EPS=1 and 78.4), optional MOPAC, optional delta."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root folder to search (if omitted, auto-clone sparse checkout PL-REX CDK2 structures)",
    )
    parser.add_argument("--xyz-out", type=str, default="xyz_out_receptor", help="Output folder for receptor XYZ files")
    parser.add_argument("--mop-out", type=str, default="mopac_inputs_receptor", help="Output folder for receptor MOP files")

    parser.add_argument("--remove-hs", action="store_true", help="Remove hydrogens while reading PDB (OpenBabel may ignore)")

    # MOPAC run options
    parser.add_argument("--run-mopac", action="store_true", help="Run MOPAC on generated .mop files")
    parser.add_argument("--mopac-exe", type=str, default="mopac", help="MOPAC executable name or full path")
    parser.add_argument("--workers", type=int, default=1, help="Parallel MOPAC jobs (1 = sequential)")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip jobs that already have .out")

    # Delta extraction
    parser.add_argument("--extract-delta", action="store_true",
                        help="Extract HEAT OF FORMATION from .out and compute delta (EPS_78.4 - EPS_1) into CSV")
    parser.add_argument("--delta-csv", type=str, default="delta_hof_eps78_4_minus_eps1.csv",
                        help="CSV filename to write under --mop-out")

    # OpenBabel executable name (in case your system uses obabel or obabel3 etc.)
    parser.add_argument("--obabel-exe", type=str, default="obabel", help="OpenBabel executable (default: obabel)")

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

    receptor_pdbs = find_receptor_pdb_files(root)
    if not receptor_pdbs:
        print(f"No receptor.pdb found under: {root}")
        return

    print(f"\nFound {len(receptor_pdbs)} receptor.pdb files under: {root}")
    print(f"Writing receptor XYZ to: {xyz_out}")
    print(f"Writing receptor MOP to: {mop_out} (EPS_1 and EPS_78_4)")

    total_xyz = 0
    total_mop = 0

    eps_values = [1.0, 78.4]

    for pdb in receptor_pdbs:
        rel_prefix = pdb.parent.as_posix().replace("/", "_")
        xyz_file = xyz_out / f"{rel_prefix}__receptor.xyz"

        # PDB -> XYZ
        ok = pdb_to_xyz_openbabel(pdb, xyz_file)
        if not ok:
            continue
        total_xyz += 1

        # XYZ -> MOP (two EPS folders)
        for eps in eps_values:
            eps_tag = str(int(eps)) if float(eps).is_integer() else str(eps).replace(".", "_")
            mop_path = (mop_out / f"EPS_{eps_tag}" / xyz_file.relative_to(xyz_out)).with_suffix(".mop")

            # EPS-specific keywords:
            # - EPS=1: run WITHOUT MOZYME
            # - EPS=78.4: run WITH MOZYME
            if abs(eps - 1.0) < 1e-6:
                kw = "1SCF"          # no MOZYME
            elif abs(eps - 78.4) < 1e-6:
                kw = "1SCF MOZYME"  # with MOZYME
            else:
                kw = "1SCF"         # default

            xyz_to_mop(
                xyz_path=xyz_file,
                mop_path=mop_path,
                method="PM7",
                keywords=kw,
                charge=2,
                eps=eps,
                two_blank_lines=True,
            )
            total_mop += 1


    print("\nDone generating inputs.")
    print(f"Total XYZ written : {total_xyz}")
    print(f"Total MOP written : {total_mop}")

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

