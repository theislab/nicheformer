# nicheformer

This is the official repository for **Nicheformer: a foundation model for single-cell and spatial omics**

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.04.15.589472v1) &nbsp;

A rendered Jupyter book version of this repository will be available soon.

## Citation

If you use our tool or build upon our concepts in your own work, please cite it as

```
Schaar, A.C., Tejada-Lapuerta, A., et al. Nicheformer: a foundation model for single-cell and spatial omics. bioRxiv (2024). doi: https://doi.org/10.1101/2024.04.15.589472
```

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).


<!--
1) Install the latest release of `nicheformer` from `PyPI <https://pypi.org/project/nicheformer/>`_:

```bash
pip install nicheformer
```
-->

Install the latest development version:

```bash
git clone https://github.com/theislab/nicheformer.git
cd nicheformer
pip install -e .
```
## Nicheformer data
We provide examplary data loading scripts in the data subdirectory that can be used as templates for loading the spatial omics datasets and datasets retreived from GEO. 

## Pretraining weights
We provide the Nicheformer pretraining weights on Mendeley data, they can be downloaded from [here](https://data.mendeley.com/preview/87gm9hrgm8?a=d95a6dde-e054-4245-a7eb-0522d6ea7dff). 

## Contact

For questions and help requests, you can reach out (preferably) on GitHub or email to the corresponding author. 



[issue-tracker]: https://github.com/theislab/nicheformer/issues
[changelog]: https://nicheformer.readthedocs.io/latest/changelog.html
[link-docs]: https://nicheformer.readthedocs.io
[link-api]: https://nicheformer.readthedocs.io/latest/api.html
