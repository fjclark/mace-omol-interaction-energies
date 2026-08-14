# Calculate PLA15 Interaction Energies with MACE-OMOL

To use:

- Install the environment with e.g.: `mamba env create -f env.yaml` and activate it
- Install `mace-torch` with pip: `pip install mace-torch`
- Download the PLA15 or PLF547 input structures from https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b01171 and unzip them in a given directory
- run `mace_omol_scoring.ipynb`, ensuring that you change `STRUCTURES_DIR` to point to where you have extracted the output, and `CALC` to your desired MACE calculator. You will likely need to run on CPU due to the large size of some complexes.

- The script for generating different conformers and running MOPAC with PM7 optimisation can be found here:
https://github.com/chikitng/mace-omol-interaction-energies/blob/features-diff-conf/3QTU/10_conf/test.py

- Add the NLP optimisation script

Please note that the MACE models are released under the ASL.
