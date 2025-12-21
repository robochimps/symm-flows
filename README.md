# symm-flows

This repository implements symmetry-aware invertible residual networks (iResNets) for calculating vibrational spectra of molecules, as described in the associated publication.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/robochimps/symm-flows.git
   cd symm-flows
   ```

2. Install dependencies (recommended: use a virtual environment):
   ```sh
   pip install external_packages/pyhami-0.0.1-py3-none-any.whl
   pip install jax flax optax joblib tasmanian
   ```

## Usage

The main workflow is in `nh3.py`, which demonstrates the full pipeline for the NH3 molecule.

To run an experiment:

```sh
python nh3.py <pmax> <nblocks>
# Example:
python nh3.py 16 5
```

Model parameters and logs are saved in folders named like `nh3_se100_iresnet_nblocks_{nblocks}_pmax_{pmax}_sym`.

### Checkpointing and Restarting
- Set the `restart` variable in `nh3.py` to control checkpoint loading:
  - `0`: start fresh
  - `1`: load from default checkpoint
  - `2`: load from latest checkpoint

## Project Structure

- `nh3.py`: Main script for data preparation, model definition, training, and evaluation.
- `external_packages/pyhami-0.0.1-py3-none-any.whl`: Custom dependency for molecular Hamiltonian and symmetry utilities.
- `models/`, `hamiltonian/`, `symmetry_functions/`, `basis/`: Imported as submodules in `nh3.py` (expected to be in the same directory or as part of `pyhami`).

## Key Patterns and Conventions

- Symmetry operations (G12 group) are handled via utility functions from `pyhami`.
- Basis functions and indices are generated with `generate_prod_ind` and Hermite polynomials.
- Tasmanian sparse grids are used for quadrature and data generation.
- Model is a custom iResNet2 with symmetry wrappers and SVD-based kernel clipping.
- Logging and checkpointing use `joblib` and plain text files.

## References

- See `.github/copilot-instructions.md` for detailed developer and AI agent instructions.
- For scientific details, refer to the associated publication.
