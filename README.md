# symm-flows

This repository implements symmetry-aware invertible residual networks (iResNets) for calculating vibrational spectra of molecules, as described in the associated publication.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/robochimps/symm-flows.git
   cd symm-flows
   ```

2. Build the environment (recommended: use conda):
   ```sh
    conda env create -f environment.yml
   ```

3. Activate the environment:
   ```sh
    conda activate flows-symm
   ```

3. Install the `pyhami` dependency:
   ```sh
    pip install external_packages/pyhami-0.0.1-py3-none-any.whl
   ```

## Usage

To compute the vibrational spectra of NH$_3$ run
   ```python
   python nh3.py {n}
   ```

from the repository root directory, where 
    - `n=0` initializes the parameters of the normalizing flow randomly and
    - `n=1` loads previously installed parameters.

Results for H_2CO can be obtained analogously by running
   ```python
   python h2co.py {n}
   ```

Both scripts produce results using a standard iResNet and a symmetry-aware iResNet, allowing for direct comparison between the two approaches.

## Citation

If you use this code in your research, please cite:

>  E. Vogt and Á. F. Corral, Symmtery-aware normalizing flows for computing
>  vibrational spectra of molecules (submitted). arXiv: eprint:
>  xxx.yyy (2026)
>  [DOI:10.48550/XXX](https://doi.org/10.48550/XXX)

```bibtex
@Misc{Vogt:arXivXXX:YYY,
  title        = {Symmtery-aware normalizing flows for computing
>  vibrational spectra of molecules},
  author       = {Vogt, Emil and Corral Fernández, Álvaro and Saleh, Yahya},
  year         = 2026,
  archivePrefix= {arXiv},
  eprint       = {XXX.YYYYY},
  primaryClass = {phys-chem},
  howpublished = {preprint},
}
