from GNNImplicitSolvent import minimize_mol, calculate_entropy
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from openff.toolkit import Molecule, ForceField
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

KJ_TO_KCAL = 4.184
ASH_GC_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"


def process_molecule(
    row_data: tuple, charge_method: Literal["ASH_GC", "AM1-BCC"] = "ASH_GC"
) -> dict | None:
    """
    Process a single molecule to calculate solvation and vacuum energies.

    Parameters:
    -----------
    row_data : tuple
        (idx, row) where idx is the index and row contains molecule data

    Returns:
    --------
    dict
        Dictionary with calculated energies or None if processing failed
    """
    idx, row = row_data
    smiles = row["SMILES"]
    compound_id = row["compound_id"]
    name = row["iupac_name"]

    print(f"Processing index {idx} of molecule: {compound_id}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES at index {idx}: {smiles}")
        return None

    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=256, useExpTorsionAnglePrefs=False)

    # Get conformation-independent GNN partial charges
    off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    if charge_method == "AM1-BCC":
        off_mol.assign_partial_charges(partial_charge_method="am1-bcc")
    elif charge_method == "ASH_GC":
        off_mol.assign_partial_charges(
            partial_charge_method=ASH_GC_MODEL,
            toolkit_registry=NAGLToolkitWrapper(),
        )
    partial_charges = off_mol.partial_charges.m_as("e")

    _, energies_sol = minimize_mol(
        mol, "tip3p", constraints=None, partial_charges=partial_charges
    )
    min_sol = np.min(energies_sol)
    min_index = int(np.argmin(energies_sol))
    min_mol = Chem.Mol(mol)  # copy the molecule (no conformers)
    conf = mol.GetConformer(min_index)
    min_mol.RemoveAllConformers()
    min_mol.AddConformer(conf, assignId=True)

    try:
        _, energies_vac = minimize_mol(
            min_mol, "vac", constraints=None, partial_charges=partial_charges
        )

    except Exception as e:
        print(f"Failed for {name} ({smiles}): {e}")
        return None

    min_vac = np.min(energies_vac)

    return {
        "compound_id": compound_id,
        "iupac_name": name,
        "SMILES": smiles,
        "min_vac_kjmol": min_vac,
        "min_vac_kcalmol": min_vac / KJ_TO_KCAL,
        "min_sol_kjmol": min_sol,
        "min_sol_kcalmol": min_sol / KJ_TO_KCAL,
    }


def main():
    # Determine number of processors to use (leave 1 core free for system)
    n_processes = max(1, mp.cpu_count() - 1)
    print(f"Using {n_processes} processes for parallel processing")

    url = "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
    df = pd.read_csv(url, sep=";", comment="#", header=None)
    df.columns = [
        "compound_id",
        "SMILES",
        "iupac_name",
        "experimental_value",
        "experimental_uncertainty",
        "mobley_group_calculated_value",
        "calculated_uncertainty",
        "experimental_referece",
        "calculated_referece",
        "text_notes",
    ]

    # min_dg_hyd_idx = df["experimental_value"].idxmin()
    # df = df.loc[[min_dg_hyd_idx]].reset_index(drop=True)
    # min_dg_hyd = df["experimental_value"].min()
    # max_dg_hyd = df["experimental_value"].max()
    # target_dg_hyd_values = np.linspace(min_dg_hyd, max_dg_hyd, 10)
    # results = []
    # for target_dg_hyd in target_dg_hyd_values:
    #     closest_idx = (df["experimental_value"] - target_dg_hyd).abs().idxmin()
    #     results.append(df.loc[closest_idx])

    # Filter to unique entries only
    # df = pd.DataFrame(results).reset_index(drop=True)

    # Prepare data for parallel processing
    row_data = list(df.iterrows())

    # Make sure we don't fork and break CUDA contexts
    mp.set_start_method("spawn", force=True)

    # Process molecules in parallel
    with mp.Pool(processes=n_processes) as pool:
        # Use tqdm with pool.imap for progress tracking
        results_raw = list(
            tqdm(
                pool.imap(process_molecule, row_data),
                total=len(row_data),
                desc="Processing molecules",
            )
        )

    # Filter out None results (failed processing)
    results = [result for result in results_raw if result is not None]
    print(f"Successfully processed {len(results)} out of {len(row_data)} molecules")

    # Continue with analysis
    df_results = pd.DataFrame(results)
    df_results.to_csv("minimised_energies.csv", index=False)
    print("Saved results to minimised_energies.csv")

    # Map experimental values back to the results based on compound_id
    # Create a mapping from compound_id to experimental_value
    exp_mapping = df.set_index("compound_id")["experimental_value"].to_dict()
    df_results["DGhyd_expt"] = df_results["compound_id"].map(exp_mapping)

    df_results["DGhyd_pred"] = (
        df_results["min_sol_kcalmol"] - df_results["min_vac_kcalmol"]
    )

    # Calculate error metrics
    df_results["abs_error"] = abs(df_results["DGhyd_pred"] - df_results["DGhyd_expt"])
    mae = df_results["abs_error"].mean()

    mae_df = pd.DataFrame([{"MAE_kcal_per_mol": mae}])
    mae_df.to_csv("mae.csv", index=False)

    df_results.to_csv("freesolv_predictions.csv", index=False)

    # Plot predicted vs experimental Delta G
    plt.figure(figsize=(6, 6))
    plt.scatter(df_results["DGhyd_expt"], df_results["DGhyd_pred"], alpha=0.6)
    plt.plot([-30, 10], [-30, 10], "r--", label="Ideal(y = x)")
    plt.xlabel("Experimental Delta G (kcal/mol)")
    plt.ylabel("Predicted Delta G (kcal/mol)")
    plt.title("Hydration Free Energies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("freesolv_plot.png", dpi=300)

    print(f"Mean Absolute Error: {mae:.3f} kcal/mol")


if __name__ == "__main__":
    main()
