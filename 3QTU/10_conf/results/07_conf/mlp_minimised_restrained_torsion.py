"""Minimise an SDF conformer with strong restraints on all rotatable torsions.

This script uses Presto functionality to:
1. Load a molecule and conformer from an SDF file.
2. Identify rotatable torsions.
3. Add strong harmonic torsion restraints to keep those torsions near
   their starting values.
4. Minimise with an MLPotential.
5. Report energy excluding the restraint force group.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, cast

import mdtraj as md
import numpy as np
import openmm
import typer
from openff.toolkit import Molecule
from openff.units import unit as off_unit
from openmm.app import Simulation
from openmm.unit import Quantity, kilocalorie_per_mole, radian
from presto.find_torsions import get_rot_torsions_by_rot_bond

from presto import mlp as presto_mlp
import csv


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

DEFAULT_FORCE_CONSTANT: float = 100000.0

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@dataclass(frozen=True)
class MinimisationResult:
    """Result of restrained minimisation for a single conformer."""

    sdf_path: str
    molecule_index: int
    conformer_index: int
    ml_model: str
    n_rotatable_torsions: int
    restraint_force_constant_kcal_per_mol_rad2: float
    minimisation_max_iterations: int
    openmm_platform: str
    minimised_sdf_path: str | None
    energy_kcal_per_mol_excluding_restraints: float


def _load_sdf_molecules(sdf_path: Path) -> list[Molecule]:
    """Load all molecules from an SDF into OpenFF Molecule objects."""
    loaded: Molecule | list[Molecule] = Molecule.from_file(
        str(sdf_path),
        file_format="SDF",
        allow_undefined_stereo=True,
    )
    if isinstance(loaded, list):
        return loaded
    return [loaded]


def _copy_with_single_conformer(
    molecule: Molecule,
    conformer_index: int,
) -> Molecule:
    """Return a deep copy of a molecule containing only one selected conformer."""
    if molecule.n_conformers == 0:
        raise ValueError("Selected molecule has no conformers in the SDF.")
    if conformer_index < 0 or conformer_index >= molecule.n_conformers:
        raise IndexError(
            f"Conformer index {conformer_index} out of range [0, {molecule.n_conformers - 1}]"
        )

    mol = Molecule(molecule)
    selected = mol.conformers[conformer_index]
    mol._conformers = [selected]
    return mol


def _dihedral_radians(
    positions_angstrom: np.ndarray,
    torsion: tuple[int, int, int, int],
) -> float:
    """Calculate a dihedral angle in radians using MDTraj."""
    xyz_nm = np.asarray(positions_angstrom, dtype=np.float64)[np.newaxis, :, :] / 10.0

    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("MOL", chain)
    for _ in range(xyz_nm.shape[1]):
        topology.add_atom("C", md.element.carbon, residue)

    trajectory = md.Trajectory(xyz=xyz_nm, topology=topology)
    torsion_indices = np.asarray([torsion], dtype=np.int32)
    angles = md.compute_dihedrals(
        trajectory,
        torsion_indices,
        periodic=False,
        opt=True,
    )
    return float(angles[0, 0])


def _find_available_force_group(system: openmm.System) -> int:
    """Find the first unused OpenMM force group in [0, 31]."""
    used_groups = {system.getForce(i).getForceGroup() for i in range(system.getNumForces())}
    for group in range(32):
        if group not in used_groups:
            return group
    raise RuntimeError("All OpenMM force groups (0-31) are already in use.")


def _add_torsion_restraints(
    system: openmm.System,
    positions_angstrom: np.ndarray,
    torsions: Iterable[tuple[int, int, int, int]],
    force_constant_kcal_per_mol_rad2: float,
    restraint_force_group: int,
) -> int:
    """Add one harmonic restraint force per torsion and return count added."""
    k = force_constant_kcal_per_mol_rad2 * kilocalorie_per_mole / radian**2
    added = 0
    for torsion in torsions:
        theta0 = _dihedral_radians(positions_angstrom, torsion)
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


def _build_mlp_system(molecule: Molecule, model: AvailableModel) -> openmm.System:
    """Create an OpenMM system from a Presto MLPotential model."""
    model_typed = cast(presto_mlp.AvailableModels, model)
    presto_mlp.validate_model_charge_compatibility(model_typed, molecule)
    potential = presto_mlp.get_mlp(model_typed)
    charge_e = molecule.total_charge.m_as(off_unit.e)
    return potential.createSystem(
        molecule.to_topology().to_openmm(),
        charge=charge_e,
    )


def _select_openmm_platform(
    requested_platform: OpenMMPlatform,
) -> tuple[openmm.Platform, dict[str, str]]:
    """Select an OpenMM platform by name with sensible GPU defaults."""
    available = {
        openmm.Platform.getPlatform(i).getName(): openmm.Platform.getPlatform(i)
        for i in range(openmm.Platform.getNumPlatforms())
    }
    if requested_platform not in available:
        raise RuntimeError(
            f"Requested platform '{requested_platform}' is not available. "
            f"Detected platforms: {sorted(available.keys())}."
        )

    platform = available[requested_platform]

    properties: dict[str, str] = {}
    if requested_platform in {"CUDA", "OpenCL", "HIP"}:
        properties["DeviceIndex"] = "0"
        properties["Precision"] = "mixed"

    return platform, properties


def minimise_with_restrained_torsions(
    sdf_path: Path,
    molecule_index: int,
    conformer_index: int,
    model: AvailableModel,
    force_constant_kcal_per_mol_rad2: float,
    max_iterations: int,
    openmm_platform: OpenMMPlatform,
    minimised_sdf_out: Path | None,
) -> MinimisationResult:
    """Minimise one conformer with torsion restraints and return restraint-excluded energy."""
    molecules = _load_sdf_molecules(sdf_path)
    if not molecules:
        raise ValueError(f"No molecules were loaded from SDF: {sdf_path}")
    if molecule_index < 0 or molecule_index >= len(molecules):
        raise IndexError(
            f"Molecule index {molecule_index} out of range [0, {len(molecules) - 1}]"
        )

    mol = _copy_with_single_conformer(molecules[molecule_index], conformer_index)
    torsion_map = get_rot_torsions_by_rot_bond(mol)
    torsions = list(torsion_map.values())

    system = _build_mlp_system(mol, model)
    restraint_group = _find_available_force_group(system)

    conformer = mol.conformers[0]
    positions_angstrom = conformer.m_as(off_unit.angstrom)
    _add_torsion_restraints(
        system=system,
        positions_angstrom=np.asarray(positions_angstrom, dtype=float),
        torsions=torsions,
        force_constant_kcal_per_mol_rad2=force_constant_kcal_per_mol_rad2,
        restraint_force_group=restraint_group,
    )

    topology = mol.to_topology().to_openmm()
    integrator = openmm.VerletIntegrator(1.0 * openmm.unit.femtosecond)
    platform, platform_properties = _select_openmm_platform(openmm_platform)
    simulation = Simulation(
        topology,
        system,
        integrator,
        platform,
        platform_properties,
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

    minimised_sdf_path_str: str | None = None
    if minimised_sdf_out is not None:
        minimised_positions = state.getPositions(asNumpy=True).value_in_unit(
            openmm.unit.angstrom
        )
        minimised_mol = Molecule(mol)
        minimised_mol._conformers = [
            np.asarray(minimised_positions, dtype=float) * off_unit.angstrom
        ]
        minimised_sdf_out.parent.mkdir(parents=True, exist_ok=True)
        minimised_mol.to_file(str(minimised_sdf_out), file_format="SDF")
        minimised_sdf_path_str = str(minimised_sdf_out)

    return MinimisationResult(
        sdf_path=str(sdf_path),
        molecule_index=molecule_index,
        conformer_index=conformer_index,
        ml_model=model,
        n_rotatable_torsions=len(torsions),
        restraint_force_constant_kcal_per_mol_rad2=force_constant_kcal_per_mol_rad2,
        minimisation_max_iterations=max_iterations,
        openmm_platform=platform.getName(),
        minimised_sdf_path=minimised_sdf_path_str,
        energy_kcal_per_mol_excluding_restraints=float(energy),
    )


@app.command()
def main(
    sdf: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to input SDF containing one or more molecules/conformers.",
    ),
    model: AvailableModel = typer.Option(
        "mace-off23-medium",
        "--model",
        help="Presto MLP model name.",
    ),
    molecule_index: int = typer.Option(
        0,
        min=0,
        help="Index of molecule entry in SDF.",
    ),
    conformer_index: int = typer.Option(
        0,
        min=0,
        help="Index of conformer for the selected molecule.",
    ),
    restraint_k_kcal_mol_rad2: float = typer.Option(
        DEFAULT_FORCE_CONSTANT,
        "--restraint-k",
        min=0.0,
        help="Torsion restraint force constant in kcal mol^-1 rad^-2.",
    ),
    max_iterations: int = typer.Option(
        0,
        min=0,
        help="OpenMM minimisation max iterations; 0 means until convergence.",
    ),
    platform: OpenMMPlatform = typer.Option(
        "CUDA",
        "--platform",
        help="OpenMM platform to use (defaults to CUDA).",
    ),
    minimised_sdf_out: Path | None = typer.Option(
        None,
        "--minimised-sdf-out",
        help="Optional path to write the minimised structure as SDF.",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json-out",
        help="Optional path to write JSON output.",
    ),
    text_out: Path | None = typer.Option(
        None,
        "--text-out",
        help="Optional path to write plain-text output.",
    ),
     csv_out: Path | None = typer.Option(
        None,
        "--csv-out",
        help="Optional path to write CSV output.",
    ),
) -> None:
    """Run restrained torsion minimisation and print MLP energy excluding restraints."""
    result = minimise_with_restrained_torsions(
        sdf_path=sdf,
        molecule_index=molecule_index,
        conformer_index=conformer_index,
        model=model,
        force_constant_kcal_per_mol_rad2=restraint_k_kcal_mol_rad2,
        max_iterations=max_iterations,
        openmm_platform=platform,
        minimised_sdf_out=minimised_sdf_out,
    )

    payload: dict[str, Any] = asdict(result)
    payload_text = json.dumps(payload, indent=2, sort_keys=True)
    typer.echo(payload_text)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(payload_text + "\n", encoding="utf-8")

    if text_out is not None:
        text_out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"sdf_path: {result.sdf_path}",
            f"molecule_index: {result.molecule_index}",
            f"conformer_index: {result.conformer_index}",
            f"ml_model: {result.ml_model}",
            f"n_rotatable_torsions: {result.n_rotatable_torsions}",
            (
                "restraint_force_constant_kcal_per_mol_rad2: "
                f"{result.restraint_force_constant_kcal_per_mol_rad2}"
            ),
            f"minimisation_max_iterations: {result.minimisation_max_iterations}",
            f"openmm_platform: {result.openmm_platform}",
            f"minimised_sdf_path: {result.minimised_sdf_path}",
            (
                "energy_kcal_per_mol_excluding_restraints: "
                f"{result.energy_kcal_per_mol_excluding_restraints}"
            ),
        ]
        text_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    
    if csv_out is not None:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_out.exists()

        with csv_out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(payload)

if __name__ == "__main__":
    app()
