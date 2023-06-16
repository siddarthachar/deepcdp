# DeepCDP
DeepCDP: Deep learning Charge Density Prediction

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fnano13121853-blue)](https://doi.org/10.3390/nano13121853)

Code and examples for predicting electron density of atomistic systems using NNs. This work is based on our paper ["Machine Learning Electron Density Prediction Using Weighted Smooth Overlap of Atomic Positions"](https://doi.org/10.3390/nano13121853). 

## Paper Abstract
Having access to accurate electron densities in chemical systems, especially for dynamical systems involving chemical reactions, ion transport, and other charge transfer processes, is crucial for numerous applications in materials chemistry. Traditional methods for computationally predicting electron density data for such systems include quantum mechanical (QM) techniques, such as density functional theory. However, poor scaling of these QM methods restricts their use to relatively small system sizes and short dynamic time scales. To overcome this limitation, we have developed a deep neural network machine learning formalism, which we call deep charge density prediction (DeepCDP), for predicting charge densities by only using atomic positions for molecules and condensed phase (periodic) systems. Our method uses the weighted smooth overlap of atomic positions to fingerprint environments on a grid-point basis and map it to electron density data generated from QM simulations. %please confirm intended meaning has been retained with punctuation
We trained models for bulk systems of copper, LiF, and silicon; for a molecular system, water; and for two-dimensional charged and uncharged systems, hydroxyl-functionalized graphane, with and without an added proton. We showed that DeepCDP achieves prediction $R^2$ values greater than 0.99 and mean squared error values on the order of $10^{-5}$ $e^2$  $&Aring^{-6}$ for most systems. DeepCDP scales linearly with system size, is highly parallelizable, and is capable of accurately predicting the  excess charge in protonated hydroxyl-functionalized graphane. We demonstrate how DeepCDP can be used to accurately track the location of charges (protons) by computing electron densities at a few selected grid points in the materials, thus significantly reducing the computational cost. We also show that our models can be transferable, allowing prediction of electron densities for systems on which it has not been trained but that contain a subset of atomic species on which it has been trained. Our approach can be used to develop models that span different chemical systems and train them for the study of large-scale charge transport and chemical reactions.

## In this repository
We provide examples of building and training a DeepCDP network that is capable of predicting electron density data. These examples are demonstrated in Jupyter notebooks.  We have validated our method by successfully predicting charge densities for a bulk metal, Cu; a covalently bound semiconductor, bulk Si; an ionic insulator, LiF; an inorganic molecular fluid, H<sub>2</sub>O; and an organic-like 2-D material, graphanol. Selected notebooks are: 

 - [Bulk metal: Cu](Bulk-Copper-SOAP-importance.ipynb)
 - [Covalently bound semiconductor: Si](Bulk-Si.ipynb)
 - [Ionic insulator: LiF](LiF.ipynb)
 - [Inorganic molecular fluid: Water molecules](Water-5-model-largeR-MAE.ipynb)
 - [Organic-like 2-D material: Graphanol](GOH_24C-charged-model-electron-contraint.ipynb)

These notebooks demonstrate from start to finish, how one can use `.cube` file data to train machine learning models that can predict electron density data of a system. We provide [training data](data/) for each system as well as CP2K [input scripts](cp2k/) to generate it. Input structures as `.xyz` files are provided in [example/structures](example/structures) and cube file predictions from our examples are provided in [example/predictions](example/predictions). Images that are used in the manuscript are generated from one of these notebooks and are stored in [Images](Images). 

## Using DeepCDP

We have tested DeepCDP with [Pytorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). Calculations were performed on an M1 Mac GPU. The device was specified using the command `device = 'mps' if torch.backends.mps.is_available() else 'cpu'`. We have also used [MLPs in Scikit Learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) to train some test models. We recommend creating a virtual environment and install all dependencies there. For instance:

1. Create a virtual environment called `deepcdp-env` and activate it:

```
python3 -m venv deepcdp-env
source deepcdp/bin/activate
```

2. Install DeepCDP dependencies:

```
pip install matplotlib
pip install dscribe
pip install torch
pip install py3Dmol
pip install rdkit
pip install IPython
```

In your python script, make sure to `sys.path.append('deepcdp/')` the [deepcdp](deepcdp/) folder to use the DeepCDP functions.

## Citing 

```
@Article{achar2023DeepCDP,
AUTHOR = {Achar, Siddarth K. and Bernasconi, Leonardo and Johnson, J. Karl},
TITLE = {Machine Learning Electron Density Prediction Using Weighted Smooth Overlap of Atomic Positions},
JOURNAL = {Nanomaterials},
VOLUME = {13},
YEAR = {2023},
NUMBER = {12},
ARTICLE-NUMBER = {1853},
URL = {https://doi.org/10.3390/nano13121853},
ISSN = {2079-4991},
DOI = {10.3390/nano13121853}
}
```

## Contact

[Contact Siddarth Achar](mailto:ska31@pitt.edu) for questions regarding this project. 
