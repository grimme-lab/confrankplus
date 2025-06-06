<h1 align="center"><img src="./assets/logo.png" alt="ConfRank+" width="600"></h3>

<h3 align="center">ConfRank+: Extending Conformer Ranking to Charged Molecules</h3>
<p align="center"> A High-Throughput Machine Learning Model for Charged Molecular Conformers </p>

<p align="center">
  <strong><a href="https://doi.org/10.26434/chemrxiv-2025-xkwk6">Preprint</a></strong> | 
  <strong><a href="https://zenodo.org/records/15465665">Dataset</a></strong> | 
  <strong><a href="https://doi.org/10.1021/acs.jcim.4c01524">ConfRank 1.0</a></strong>
</p>

# Installation

You can install the python code for running the ConfRankPlus model via pip:

```bash
pip3 install git+https://github.com/grimme-lab/confrankplus.git
```

It is **highly recommended** to install `torch_cluster` for significant faster computation of atomistic graphs, e.g., by
running:

```bash
pip install git+https://github.com/rusty1s/pytorch_cluster.git
```

For further installation guidelines, see [here](https://github.com/rusty1s/pytorch_cluster/tree/master).

# Inference

## Label xyz files from command line

You can label (multiple) xyz files (or other formats supported by the `ase.io` module) with the `confrankplus` command
in your command line:

```bash
 confrankplus --files *.xyz --output_path confrank_output.xyz --total_charge 0  --fidelity r2SCAN-3c --batch_size 20
``` 

The results will be written to the path specified via `--output_path`. The units of the energies written to the output
path will be in kcal/mol.

Furthermore, in the directory `evaluate` there are example scripts for running inference on the test datasets of our
paper.

## ASE Calculator

In addition to the `confrankplus` command line interface, we provide an ASE calculator.

Example:

```python
import numpy as np

np.random.seed(0)
from ase.collections import s22
from ConfRankPlus.inference.calculator import ConfRankPlusCalculator

# load the ConfRankPlus calculator with r2SCAN-3c fidelity
calculator = ConfRankPlusCalculator.load_default(fidelity="r2SCAN-3c",
                                                 compute_forces=False)

molecule = s22['Adenine-thymine_Watson-Crick_complex']
molecule.set_calculator(calculator)
# energy before displacements:
energy_1 = molecule.get_potential_energy()
# Apply random displacements:
displacements = np.random.uniform(-0.01, 0.01, molecule.positions.shape)  # in Angstrom
molecule.set_positions(molecule.get_positions() + displacements)
# energy after displacement:
energy_2 = molecule.get_potential_energy()

# ASE uses eV as unit for energy:
print(f"The energy difference is {energy_2 - energy_1:.2E} eV.")
```

Note: our model has not been tested for geometry optimization or MD simulations.

# Loading datasets

You can load the datasets from [Zenodo](https://zenodo.org/records/15465665) as follows:

```python
from ConfRankPlus.data.dataset import HDF5Dataset

filepath = ...  # path to .h5 file 
dataset = HDF5Dataset.from_hdf5(filepath=filepath, precision=64)
```

`dataset` will be an instance of the `InMemoryDataset` class from PyTorch Geometric. 

# Citation

When using or referring to ConfRankPlus, please cite it as follows:

```
@misc{ConfRankPlus25, 
author={Oerder, Rick and H{\"o}lzer , Christian and Hamaekers, Jan}, 
title={ConfRank+: Extending Conformer Ranking to Charged Molecules},
year={2025},
doi={10.26434/chemrxiv-2025-xkwk6}, 
note={ChemRxiv Preprint}}
```

and

```
@article{ConfRank24,
author = {H{\"o}lzer, Christian and Oerder, Rick and Grimme, Stefan and Hamaekers, Jan},
title = {ConfRank: Improving GFN-FF Conformer Ranking with Pairwise Training},
journal = {Journal of Chemical Information and Modeling},
volume = {64},
number = {23},
pages = {8909-8925},
year = {2024},
doi = {10.1021/acs.jcim.4c01524},
note ={PMID: 39565928},
URL = {https://doi.org/10.1021/acs.jcim.4c01524},
eprint = {https://doi.org/10.1021/acs.jcim.4c01524}}
```


# License
 <p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><span property="dct:title">The content of this repository is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>
