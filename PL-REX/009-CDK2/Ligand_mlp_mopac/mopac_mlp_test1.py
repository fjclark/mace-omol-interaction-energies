#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast
from urllib.request import Request, urlopen, urlretrieve

import mdtraj as md
import numpy as np
import openmm
import typer
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmm.app import Simulation
from openmm.unit import Quantity, kilocalorie_per_mole, radian
from rdkit import Chem
from rdkit.Chem import AllChem

from presto import mlp as presto_mlp
from presto.find_torsions import get_rot_torsions_by_rot_bond


TARGET_FOLDER = "009-CDK2"
GITHUB_API = "https://api.github.com/repos/Honza-R/PL-REX/contents"
RAW_BASE = "https://raw.githubusercontent.com/Honza-R/PL-REX/main"

NUM_CONFS = 10
RANDOM_SEED = 2026
PRUNE_RMS = -1

DEFAULT_FORCE_CONSTANT = 10000.0
CHARGED_LIGAND_MODEL = "aceff-2.0"

AvailableModel = Literal[
    "aceff-2.0",
    "mace-off23-small",
    "mace-off23-medium",
    "mace-off23-large",
    "egret-1",
    "aimnet2_b973c_d3_ens",
    "aimnet2_wb97m_d3_ens",
]

OpenMMPlatform = Literal["CUDA", "OpenCL", "HIP", "CPU", "Reference"]

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass
class SystemPaths:
    pdb_id: str
    base_dir: Path
    input_dir: Path
    conf_dir: Path
    mopac_water_dir: Path
    mopac_vac_dir: Path
    results_dir: Path

    input_sdf: Path
    conf_sdf: Path

    water_results_csv: Path
    vacuum_results_csv: Path
    delta_results_csv: Path

    best_sdf: Path
    mlp_minimised_sdf: Path
    mlp_json: Path
    mlp_csv: Path

    rdkit_same_conf_sdf: Path
    rdkit_same_conf_mlp_sdf: Path
    rdkit_same_conf_mlp_json: Path
    rdkit_same_conf_mlp_csv: Path


@dataclass(frozen=True)
class MLPResult:
    pdb_id: str
    source_type: str
    sdf_path: str
    molecule_index: int
    conformer_index: int
    formal_charge_e: float
    requested_ml_model: str
    ml_model: str
    auto_model_by_charge: bool
    n_rotatable_torsions: int
    restraint_force_constant_kcal_per_mol_rad2: float
    minimisation_max_iterations: int
    openmm_platform: str
    minimised_sdf_path: str | None
    energy_kcal_per_mol_excluding_restraints: float
    status: str
    error_message: str | None


def make_paths(pdb_id: str, output_root: Path) -> SystemPaths:
    base_dir = output_root / pdb_id
    input_dir = base_dir / "input"
    conf_dir = base_dir / "conformers"
    mopac_water_dir = base_dir / "mopac_water"
    mopac_vac_dir = base_dir / "mopac_vacuum"
    results_dir = base_dir / "results"

    return SystemPaths(
        pdb_id=pdb_id,
        base_dir=base_dir,
        input_dir=input_dir,
        conf_dir=conf_dir,
        mopac_water_dir=mopac_water_dir,
        mopac_vac_dir=mopac_vac_dir,
        results_dir=results_dir,
        input_sdf=input_dir / "ligand.sdf",
        conf_sdf=conf_dir / "ligand_10confs.sdf",
        water_results_csv=results_dir / "water_results.csv",
        vacuum_results_csv=results_dir / "vacuum_results.csv",
        delta_results_csv=results_dir / "water_minus_vacuum.csv",
        best_sdf=results_dir / "best_water_stabilised.sdf",
        mlp_minimised_sdf=results_dir / "best_water_stabilised_mlp_minimised.sdf",
        mlp_json=results_dir / "mlp_result_mopac_water_selected.json",
        mlp_csv=results_dir / "mlp_result_mopac_water_selected.csv",
        rdkit_same_conf_sdf=results_dir / "same_selected_rdkit_conformer.sdf",
        rdkit_same_conf_mlp_sdf=results_dir / "same_selected_rdkit_conformer_mlp_minimised.sdf",
        rdkit_same_conf_mlp_json=results_dir / "same_selected_rdkit_conformer_mlp_result.json",
        rdkit_same_conf_mlp_csv=results_dir / "same_selected_rdkit_conformer_mlp_result.csv",
    )


def setup_dirs(paths: SystemPaths) -> None:
    for d in [
        paths.input_dir,
        paths.conf_dir,
        paths.mopac_water_dir,
        paths.mopac_vac_dir,
        paths.results_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def github_api_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "python-009-cdk2-pipeline"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_009_cdk2_pdb_ids() -> list[str]:
    url = f"{GITHUB_API}/{TARGET_FOLDER}/structures_pl-rex"
    data = github_api_json(url)
    return sorted(item["name"] for item in data if item.get("type") == "dir")


def download_ligand(paths: SystemPaths, overwrite: bool = False) -> None:
    if paths.input_sdf.exists() and not overwrite:
        print(f"[{paths.pdb_id}] ligand.sdf exists; skipping download.", flush=True)
        return

    url = f"{RAW_BASE}/{TARGET_FOLDER}/structures_pl-rex/{paths.pdb_id}/ligand.sdf"
    print(f"[{paths.pdb_id}] Downloading {url}", flush=True)
    urlretrieve(url, paths.input_sdf)


def get_charge_rdkit(mol: Chem.Mol) -> int:
    if mol.HasProp("charge"):
        return int(float(mol.GetProp("charge").strip()))
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


def generate_conformers(paths: SystemPaths) -> int:
    suppl = Chem.SDMolSupplier(str(paths.input_sdf), removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"Could not read molecule from {paths.input_sdf}")

    charge = get_charge_rdkit(mol)
    print(f"[{paths.pdb_id}] Charge: {charge}", flush=True)

    mol = Chem.AddHs(mol, addCoords=True)

    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    params.pruneRmsThresh = PRUNE_RMS
    params.useSmallRingTorsions = True
    params.enforceChirality = True

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, params=params))
    if not conf_ids:
        raise RuntimeError("No conformers generated.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        print(f"[{paths.pdb_id}] Optimising conformers with MMFF.", flush=True)
        for cid in conf_ids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    else:
        print(f"[{paths.pdb_id}] MMFF unavailable; using UFF.", flush=True)
        for cid in conf_ids:
            AllChem.UFFOptimizeMolecule(mol, confId=cid)

    writer = Chem.SDWriter(str(paths.conf_sdf))
    for i, cid in enumerate(conf_ids, start=1):
        mol.SetProp("_Name", f"{paths.pdb_id}_conf_{i:02d}")
        writer.write(mol, confId=cid)
    writer.close()

    print(f"[{paths.pdb_id}] Wrote {len(conf_ids)} conformers -> {paths.conf_sdf}", flush=True)
    return charge


def build_water_opt_keywords(charge: int) -> str:
    return f"PM7 CHARGE={charge} EPS=78.4"


def build_vacuum_sp_keywords(charge: int) -> str:
    return f"PM7 CHARGE={charge} 1SCF"


def mol_to_mopac_xyz_block(mol: Chem.Mol, conf_id: int = 0) -> str:
    conf = mol.GetConformer(conf_id)
    lines = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(
            f"{atom.GetSymbol():2s} "
            f"{pos.x:12.6f} 1 "
            f"{pos.y:12.6f} 1 "
            f"{pos.z:12.6f} 1"
        )
    return "\n".join(lines)


def write_water_optimization_inputs(paths: SystemPaths, charge: int) -> list[Path]:
    mols = [m for m in Chem.SDMolSupplier(str(paths.conf_sdf), removeHs=False) if m is not None]
    keywords = build_water_opt_keywords(charge)
    mop_files = []

    for i, mol in enumerate(mols, start=1):
        mop_path = paths.mopac_water_dir / f"conf_{i:02d}_water_opt.mop"
        mop_path.write_text(f"{keywords}\n\n\n{mol_to_mopac_xyz_block(mol)}\n\n")
        mop_files.append(mop_path)

    print(f"[{paths.pdb_id}] Wrote {len(mop_files)} water MOPAC inputs.", flush=True)
    return mop_files


def find_mopac_executable() -> str:
    for exe in ["mopac", "MOPAC2016.exe", "MOPAC.exe"]:
        path = shutil.which(exe)
        if path:
            return path
    raise FileNotFoundError("Could not find MOPAC executable in PATH.")


def run_mopac_jobs(mop_files: list[Path], workdir: Path, pdb_id: str) -> None:
    mopac_exe = find_mopac_executable()
    print(f"[{pdb_id}] Using MOPAC: {mopac_exe}", flush=True)

    for mop_file in mop_files:
        out_file = workdir / f"{mop_file.stem}.out"
        if out_file.exists():
            print(f"[{pdb_id}] Existing {out_file.name}; skipping.", flush=True)
            continue

        print(f"[{pdb_id}] Running {mop_file.name}", flush=True)
        subprocess.run([mopac_exe, mop_file.name], cwd=str(workdir), check=True)


def is_float(s: str) -> bool:
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
    lines = []
    for line in cart_block.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            atom = parts[1]
            x, y, z = parts[2:5]
            lines.append(
                f"{atom:<2} {x} {optimize_flags} "
                f"{y} {optimize_flags} {z} {optimize_flags}"
            )
    return "\n".join(lines)


def write_vacuum_singlepoint_inputs_from_water_outputs(
    paths: SystemPaths,
    charge: int,
) -> list[Path]:
    keywords = build_vacuum_sp_keywords(charge)
    vacuum_mop_files = []

    for out_file in sorted(paths.mopac_water_dir.glob("conf_*_water_opt.out")):
        final_cart_block = extract_final_block(out_file)

        if not final_cart_block.strip():
            print(f"[{paths.pdb_id}] Warning: no final coordinates in {out_file.name}", flush=True)
            continue

        coord_txt = paths.mopac_vac_dir / f"{out_file.stem}_final_cartesian_coordinates.txt"
        coord_txt.write_text(final_cart_block + "\n")

        vacuum_mop = paths.mopac_vac_dir / f"{out_file.stem}_vacuum_sp.mop"
        vacuum_mop.write_text(
            f"{keywords}\n\n\n{cartesian_block_to_mopac_geometry(final_cart_block, 0)}\n"
        )
        vacuum_mop_files.append(vacuum_mop)

    print(f"[{paths.pdb_id}] Wrote {len(vacuum_mop_files)} vacuum SP inputs.", flush=True)
    return vacuum_mop_files


def extract_heat_of_formation(text: str) -> float | None:
    patterns = [
        re.compile(r"FINAL\s+HEAT\s+OF\s+FORMATION\s*=\s*([-+0-9.Ee]+)", re.I),
        re.compile(r"HEAT\s+OF\s+FORMATION\s*=\s*([-+0-9.Ee]+)", re.I),
    ]
    for pat in patterns:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    return None


def extract_total_energy(text: str) -> float | None:
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

    if (
        "job ended normally" in low
        or "== mopac done ==" in low
        or "mopac done" in low
        or "normal termination" in low
        or "final heat of formation" in low
        or "total job time" in low
    ):
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


def parse_outputs(output_dir: Path, csv_path: Path, pdb_id: str) -> None:
    rows = []

    for out_file in sorted(output_dir.glob("*.out")):
        text = out_file.read_text(errors="ignore")
        rows.append(
            [
                out_file.name,
                extract_status(text),
                extract_heat_of_formation(text),
                extract_total_energy(text),
            ]
        )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "status", "heat_of_formation", "total_energy"])
        writer.writerows(rows)

    print(f"[{pdb_id}] Parsed {len(rows)} outputs -> {csv_path}", flush=True)


def _water_key(filename: str) -> str:
    return filename.replace("_water_opt.out", "")


def _vacuum_key(filename: str) -> str:
    return filename.replace("_water_opt_vacuum_sp.out", "")


def compare_water_minus_vacuum(paths: SystemPaths) -> dict[str, Any]:
    water: dict[str, dict[str, Any]] = {}
    vacuum: dict[str, dict[str, Any]] = {}

    with paths.water_results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ok" and row["heat_of_formation"]:
                water[_water_key(row["file"])] = row

    with paths.vacuum_results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] == "ok" and row["heat_of_formation"]:
                vacuum[_vacuum_key(row["file"])] = row

    rows = []
    for key in sorted(water):
        if key not in vacuum:
            continue

        water_hof = float(water[key]["heat_of_formation"])
        vacuum_hof = float(vacuum[key]["heat_of_formation"])
        delta = water_hof - vacuum_hof

        rows.append(
            {
                "pdb_id": paths.pdb_id,
                "conformer": key,
                "water_file": water[key]["file"],
                "vacuum_file": vacuum[key]["file"],
                "water_heat_of_formation": water_hof,
                "vacuum_heat_of_formation": vacuum_hof,
                "water_minus_vacuum": delta,
            }
        )

    if not rows:
        raise RuntimeError("No valid water/vacuum pairs found.")

    rows.sort(key=lambda r: r["water_minus_vacuum"])

    with paths.delta_results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print(
        f"[{paths.pdb_id}] Best={best['conformer']} "
        f"water-minus-vacuum={best['water_minus_vacuum']}",
        flush=True,
    )
    return best


def selected_conformer_index(best: dict[str, Any]) -> int:
    match = re.search(r"conf_(\d+)", best["conformer"])
    if not match:
        raise ValueError(f"Cannot extract conformer number from {best['conformer']}")
    return int(match.group(1)) - 1


def write_best_water_optimised_sdf(paths: SystemPaths, best: dict[str, Any]) -> Path:
    sdf_index = selected_conformer_index(best)

    mols = [m for m in Chem.SDMolSupplier(str(paths.conf_sdf), removeHs=False) if m is not None]
    if sdf_index < 0 or sdf_index >= len(mols):
        raise IndexError(f"Conformer index {sdf_index} not found in {paths.conf_sdf}")

    mol = Chem.Mol(mols[sdf_index])
    cart_block = extract_final_block(paths.mopac_water_dir / best["water_file"])

    coord_rows = []
    for line in cart_block.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].isdigit():
            coord_rows.append((parts[1], float(parts[2]), float(parts[3]), float(parts[4])))

    if len(coord_rows) != mol.GetNumAtoms():
        raise ValueError(
            f"Atom count mismatch: MOPAC={len(coord_rows)}, SDF={mol.GetNumAtoms()}"
        )

    conf = mol.GetConformer()
    for i, (_, x, y, z) in enumerate(coord_rows):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))

    mol.SetProp("_Name", f"{paths.pdb_id}_{best['conformer']}_mopac_water_best")
    mol.SetProp("pdb_id", paths.pdb_id)
    mol.SetProp("source_type", "mopac_water_optimised_selected")
    mol.SetProp("selected_conformer", best["conformer"])
    mol.SetProp("water_minus_vacuum", str(best["water_minus_vacuum"]))
    mol.SetProp("water_heat_of_formation", str(best["water_heat_of_formation"]))
    mol.SetProp("vacuum_heat_of_formation", str(best["vacuum_heat_of_formation"]))

    writer = Chem.SDWriter(str(paths.best_sdf))
    writer.write(mol)
    writer.close()

    print(f"[{paths.pdb_id}] MOPAC-selected SDF -> {paths.best_sdf}", flush=True)
    return paths.best_sdf


def write_same_selected_rdkit_conformer_sdf(paths: SystemPaths, best: dict[str, Any]) -> Path:
    sdf_index = selected_conformer_index(best)

    mols = [m for m in Chem.SDMolSupplier(str(paths.conf_sdf), removeHs=False) if m is not None]
    if sdf_index < 0 or sdf_index >= len(mols):
        raise IndexError(f"Conformer index {sdf_index} not found in {paths.conf_sdf}")

    mol = Chem.Mol(mols[sdf_index])
    mol.SetProp("_Name", f"{paths.pdb_id}_{best['conformer']}_original_rdkit")
    mol.SetProp("pdb_id", paths.pdb_id)
    mol.SetProp("source_type", "original_rdkit_same_selected_conformer")
    mol.SetProp("selected_conformer", best["conformer"])
    mol.SetProp("selection_source", "lowest_water_minus_vacuum")
    mol.SetProp("water_minus_vacuum", str(best["water_minus_vacuum"]))

    writer = Chem.SDWriter(str(paths.rdkit_same_conf_sdf))
    writer.write(mol)
    writer.close()

    print(f"[{paths.pdb_id}] Same selected RDKit SDF -> {paths.rdkit_same_conf_sdf}", flush=True)
    return paths.rdkit_same_conf_sdf


def load_sdf_molecules(sdf_path: Path) -> list[Molecule]:
    loaded = Molecule.from_file(str(sdf_path), file_format="SDF", allow_undefined_stereo=True)
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def molecule_total_charge_e(molecule: Molecule) -> float:
    return float(molecule.total_charge.m_as(off_unit.e))


def choose_model_for_charge(
    molecule: Molecule,
    requested_model: AvailableModel,
    auto_model_by_charge: bool,
) -> AvailableModel:
    if auto_model_by_charge and abs(molecule_total_charge_e(molecule)) > 1e-6:
        return cast(AvailableModel, CHARGED_LIGAND_MODEL)
    return requested_model


def copy_with_single_conformer(molecule: Molecule, conformer_index: int) -> Molecule:
    if molecule.n_conformers == 0:
        raise ValueError("Molecule has no conformers.")
    if conformer_index < 0 or conformer_index >= molecule.n_conformers:
        raise IndexError(f"Conformer index {conformer_index} out of range.")
    mol = Molecule(molecule)
    mol._conformers = [mol.conformers[conformer_index]]
    return mol


def normalise_torsion_values(torsion_map: dict[Any, Any]) -> list[tuple[int, int, int, int]]:
    torsions = []
    for value in torsion_map.values():
        if value is None:
            continue

        if (
            isinstance(value, (tuple, list))
            and len(value) == 4
            and all(isinstance(x, (int, np.integer)) for x in value)
        ):
            torsions.append(tuple(int(x) for x in value))
        elif isinstance(value, (tuple, list)):
            for item in value:
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 4
                    and all(isinstance(x, (int, np.integer)) for x in item)
                ):
                    torsions.append(tuple(int(x) for x in item))
        else:
            raise TypeError(f"Unsupported torsion format: {value!r}")

    return torsions


def dihedral_radians(positions_angstrom: np.ndarray, torsion: tuple[int, int, int, int]) -> float:
    xyz_nm = np.asarray(positions_angstrom, dtype=np.float64)[np.newaxis, :, :] / 10.0

    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain)
    for _ in range(xyz_nm.shape[1]):
        topology.add_atom("C", md.element.carbon, residue)

    traj = md.Trajectory(xyz=xyz_nm, topology=topology)
    angles = md.compute_dihedrals(
        traj,
        np.asarray([torsion], dtype=np.int32),
        periodic=False,
        opt=True,
    )
    return float(angles[0, 0])


def find_available_force_group(system: openmm.System) -> int:
    used = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    for group in range(32):
        if group not in used:
            return group
    raise RuntimeError("All OpenMM force groups are used.")


def add_torsion_restraints(
    system: openmm.System,
    positions_angstrom: np.ndarray,
    torsions: Iterable[tuple[int, int, int, int]],
    force_constant_kcal_per_mol_rad2: float,
    restraint_force_group: int,
) -> int:
    k = force_constant_kcal_per_mol_rad2 * kilocalorie_per_mole / radian**2
    added = 0

    for torsion in torsions:
        theta0 = dihedral_radians(positions_angstrom, torsion)
        restraint = openmm.CustomTorsionForce(
            "0.5*k*min(dtheta, 2*pi-dtheta)^2; "
            "dtheta=abs(theta-theta0); pi=3.141592653589793"
        )
        restraint.addPerTorsionParameter("k")
        restraint.addPerTorsionParameter("theta0")
        restraint.addTorsion(*torsion, [k, theta0])
        restraint.setForceGroup(restraint_force_group)
        system.addForce(restraint)
        added += 1

    return added


def build_mlp_system(molecule: Molecule, model: AvailableModel) -> openmm.System:
    model_typed = cast(presto_mlp.AvailableModels, model)
    presto_mlp.validate_model_charge_compatibility(model_typed, molecule)
    potential = presto_mlp.get_mlp(model_typed)
    charge_e = molecule.total_charge.m_as(off_unit.e)
    return potential.createSystem(molecule.to_topology().to_openmm(), charge=charge_e)


def select_openmm_platform(
    requested_platform: OpenMMPlatform,
) -> tuple[openmm.Platform, dict[str, str]]:
    available = {
        openmm.Platform.getPlatform(i).getName(): openmm.Platform.getPlatform(i)
        for i in range(openmm.Platform.getNumPlatforms())
    }

    if requested_platform not in available:
        raise RuntimeError(
            f"Requested platform {requested_platform!r} unavailable. "
            f"Detected: {sorted(available.keys())}"
        )

    properties = {}
    if requested_platform in {"CUDA", "OpenCL", "HIP"}:
        properties["DeviceIndex"] = "0"
        properties["Precision"] = "mixed"

    return available[requested_platform], properties


def run_mlp_on_sdf(
    pdb_id: str,
    source_type: str,
    sdf_path: Path,
    model: AvailableModel,
    auto_model_by_charge: bool,
    restraint_k: float,
    max_iterations: int,
    platform_name: OpenMMPlatform,
    minimised_sdf_out: Path,
) -> MLPResult:
    print(f"[{pdb_id}] Starting MLP for {source_type}: {sdf_path}", flush=True)

    molecule = load_sdf_molecules(sdf_path)[0]
    mol = copy_with_single_conformer(molecule, 0)

    formal_charge_e = molecule_total_charge_e(mol)
    selected_model = choose_model_for_charge(mol, model, auto_model_by_charge)

    torsion_map = get_rot_torsions_by_rot_bond(mol)
    torsions = normalise_torsion_values(torsion_map)

    system = build_mlp_system(mol, selected_model)
    restraint_group = find_available_force_group(system)

    conformer = mol.conformers[0]
    positions_angstrom = conformer.m_as(off_unit.angstrom)

    add_torsion_restraints(
        system=system,
        positions_angstrom=np.asarray(positions_angstrom, dtype=float),
        torsions=torsions,
        force_constant_kcal_per_mol_rad2=restraint_k,
        restraint_force_group=restraint_group,
    )

    topology = mol.to_topology().to_openmm()
    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtosecond)
    platform, platform_properties = select_openmm_platform(platform_name)
    simulation = Simulation(topology, system, integrator, platform, platform_properties)

    try:
        print(
            f"[{pdb_id}] minimiseEnergy source={source_type} "
            f"model={selected_model} charge={formal_charge_e:.2f} "
            f"torsions={len(torsions)} max_iter={max_iterations} "
            f"restraint_k={restraint_k}",
            flush=True,
        )

        simulation.context.setPositions(cast(Quantity, conformer.to_openmm()))
        simulation.minimizeEnergy(maxIterations=max_iterations)

        groups_mask = sum(1 << group for group in range(32) if group != restraint_group)
        state = simulation.context.getState(
            getEnergy=True,
            getPositions=True,
            groups=groups_mask,
        )

        energy = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)

        minimised_positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        minimised_mol = Molecule(mol)
        minimised_mol._conformers = [
            np.asarray(minimised_positions, dtype=float) * off_unit.angstrom
        ]
        minimised_sdf_out.parent.mkdir(parents=True, exist_ok=True)
        minimised_mol.to_file(str(minimised_sdf_out), file_format="SDF")

        print(
            f"[{pdb_id}] Finished MLP source={source_type} "
            f"E={energy:.6f} kcal/mol",
            flush=True,
        )

        return MLPResult(
            pdb_id=pdb_id,
            source_type=source_type,
            sdf_path=str(sdf_path),
            molecule_index=0,
            conformer_index=0,
            formal_charge_e=formal_charge_e,
            requested_ml_model=model,
            ml_model=selected_model,
            auto_model_by_charge=auto_model_by_charge,
            n_rotatable_torsions=len(torsions),
            restraint_force_constant_kcal_per_mol_rad2=restraint_k,
            minimisation_max_iterations=max_iterations,
            openmm_platform=platform.getName(),
            minimised_sdf_path=str(minimised_sdf_out),
            energy_kcal_per_mol_excluding_restraints=float(energy),
            status="ok",
            error_message=None,
        )

    finally:
        del simulation
        del integrator


def write_single_mlp_result(json_path: Path, csv_path: Path, result: MLPResult) -> None:
    payload = asdict(result)

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def run_one_pdb(
    pdb_id: str,
    output_root: Path,
    model: AvailableModel,
    auto_model_by_charge: bool,
    platform: OpenMMPlatform,
    restraint_k: float,
    max_iterations: int,
    overwrite_download: bool,
    skip_mopac: bool,
    skip_mlp: bool,
) -> dict[str, Any]:
    paths = make_paths(pdb_id, output_root)

    print("\n" + "=" * 80, flush=True)
    print(f"Running {pdb_id}", flush=True)
    print("=" * 80, flush=True)

    setup_dirs(paths)
    download_ligand(paths, overwrite=overwrite_download)

    charge = generate_conformers(paths)

    water_mop_files = write_water_optimization_inputs(paths, charge)
    if not skip_mopac:
        run_mopac_jobs(water_mop_files, paths.mopac_water_dir, pdb_id)

    parse_outputs(paths.mopac_water_dir, paths.water_results_csv, pdb_id)

    vacuum_mop_files = write_vacuum_singlepoint_inputs_from_water_outputs(paths, charge)
    if not vacuum_mop_files:
        raise RuntimeError("No vacuum single-point MOPAC files were generated.")

    if not skip_mopac:
        run_mopac_jobs(vacuum_mop_files, paths.mopac_vac_dir, pdb_id)

    parse_outputs(paths.mopac_vac_dir, paths.vacuum_results_csv, pdb_id)

    best = compare_water_minus_vacuum(paths)

    mopac_best_sdf = write_best_water_optimised_sdf(paths, best)
    same_rdkit_sdf = write_same_selected_rdkit_conformer_sdf(paths, best)

    row: dict[str, Any] = {
        "pdb_id": pdb_id,
        "status": "ok",
        "selected_conformer": best["conformer"],
        "water_minus_vacuum": best["water_minus_vacuum"],
        "mopac_water_selected_sdf": str(mopac_best_sdf),
        "same_rdkit_selected_sdf": str(same_rdkit_sdf),
    }

    if not skip_mlp:
        mopac_mlp_result = run_mlp_on_sdf(
            pdb_id=pdb_id,
            source_type="mopac_water_optimised_selected",
            sdf_path=mopac_best_sdf,
            model=model,
            auto_model_by_charge=auto_model_by_charge,
            restraint_k=restraint_k,
            max_iterations=max_iterations,
            platform_name=platform,
            minimised_sdf_out=paths.mlp_minimised_sdf,
        )
        write_single_mlp_result(paths.mlp_json, paths.mlp_csv, mopac_mlp_result)

        rdkit_mlp_result = run_mlp_on_sdf(
            pdb_id=pdb_id,
            source_type="original_rdkit_same_selected_conformer",
            sdf_path=same_rdkit_sdf,
            model=model,
            auto_model_by_charge=auto_model_by_charge,
            restraint_k=restraint_k,
            max_iterations=max_iterations,
            platform_name=platform,
            minimised_sdf_out=paths.rdkit_same_conf_mlp_sdf,
        )
        write_single_mlp_result(
            paths.rdkit_same_conf_mlp_json,
            paths.rdkit_same_conf_mlp_csv,
            rdkit_mlp_result,
        )

        row.update(
            {
                "mopac_mlp_status": mopac_mlp_result.status,
                "mopac_mlp_model": mopac_mlp_result.ml_model,
                "mopac_mlp_energy": mopac_mlp_result.energy_kcal_per_mol_excluding_restraints,
                "mopac_mlp_minimised_sdf": mopac_mlp_result.minimised_sdf_path,
                "rdkit_same_conf_mlp_status": rdkit_mlp_result.status,
                "rdkit_same_conf_mlp_model": rdkit_mlp_result.ml_model,
                "rdkit_same_conf_mlp_energy": rdkit_mlp_result.energy_kcal_per_mol_excluding_restraints,
                "rdkit_same_conf_mlp_minimised_sdf": rdkit_mlp_result.minimised_sdf_path,
            }
        )

    return row


@app.command()
def main(
    output_root: Path = typer.Option(
        Path("009-CDK2_batch_results"),
        "--output-root",
        help="Output directory.",
    ),
    pdb_id: list[str] | None = typer.Option(
        None,
        "--pdb-id",
        help="Specific PDB ID(s). If omitted, all 009-CDK2 systems are discovered.",
    ),
    model: AvailableModel = typer.Option(
        "mace-off23-medium",
        "--model",
        help="MLP model for neutral ligands.",
    ),
    auto_model_by_charge: bool = typer.Option(
        True,
        "--auto-model-by-charge/--no-auto-model-by-charge",
        help="Use aceff-2.0 for charged ligands.",
    ),
    platform: OpenMMPlatform = typer.Option(
        "CPU",
        "--platform",
        help="OpenMM platform.",
    ),
    restraint_k: float = typer.Option(
        DEFAULT_FORCE_CONSTANT,
        "--restraint-k",
        min=0.0,
    ),
    max_iterations: int = typer.Option(
        100,
        "--max-iterations",
        min=0,
    ),
    overwrite_download: bool = typer.Option(
        False,
        "--overwrite-download",
    ),
    skip_mopac: bool = typer.Option(
        False,
        "--skip-mopac",
    ),
    skip_mlp: bool = typer.Option(
        False,
        "--skip-mlp",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error/--stop-on-error",
    ),
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    pdb_ids = sorted(pdb_id) if pdb_id else discover_009_cdk2_pdb_ids()

    print(f"Found {len(pdb_ids)} 009-CDK2 systems:", flush=True)
    print(", ".join(pdb_ids), flush=True)

    summary_rows: list[dict[str, Any]] = []

    for current_pdb_id in pdb_ids:
        try:
            summary_rows.append(
                run_one_pdb(
                    pdb_id=current_pdb_id,
                    output_root=output_root,
                    model=model,
                    auto_model_by_charge=auto_model_by_charge,
                    platform=platform,
                    restraint_k=restraint_k,
                    max_iterations=max_iterations,
                    overwrite_download=overwrite_download,
                    skip_mopac=skip_mopac,
                    skip_mlp=skip_mlp,
                )
            )
        except Exception as exc:
            print(f"[{current_pdb_id}] FAILED: {exc}", flush=True)
            summary_rows.append(
                {
                    "pdb_id": current_pdb_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                break

    summary_csv = output_root / "batch_summary.csv"
    all_keys: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nBatch complete. Summary -> {summary_csv}", flush=True)


if __name__ == "__main__":
    app()
