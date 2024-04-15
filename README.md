# NF-DiagMC

The package offers a general C++ framework to perform:

- Diagrammatic Monte Carlo (DMC) simulations for general Hamiltonians
- Construction of Normalizing Flows (NF) model using Pytorch C++ API

The code was though, and used, in order to use Normalizing Flow for the generation of Feynman's diagrams to reduce autocorrelation in Diagrammatic Monte Carlo simulations.
Such use is displayed inside the folder Holstein, where an application on the single-site Holstein model is present.

## Get Started
### Compilation
The package is divided in the DMC module and the NF one, which can be compiled separately.
The simultaneous compilation of both can be performed using the CMakeLists.txt file present in the repository using the following command:
```
cmake -B build -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)' .
cmake --build build
```
This allows CMake to find the path to the installation of Pytorch present on the system.
Such command would build the DMC module of the package by standard, and only if Pytorch and CUDA toolkit are found to be correctly installed on the system then also the NF module is compiled too.

The code inside the Holstein folder is compiled only alongside the NF module, since it depends on it.

### Usage
Once fully compiled the repository using the command given above three different scripts gets created:

##### Holstein
Holstein, inside the build/Holstein directory, that performs two simulations of the single-site Holstein model for all the values of the coupling strength in the spectrum: a first using only local update and another using both local and global NF updates. Such simulation can be performed using the command:

```
./build/Holstein/Holstein #n-walkers
```

Where #n-walkers tells the number of walkers to be used at the same time in a single simulation and needs to be <= to the number of cores at disposal.

##### Htrain and Heval
This are the scripts used to train the NF model for the Holstein diagrams and to use it the following command can be used
```
./build/Holstein/ML/HLtrain path/to/folder/with/.ini/file/
```
The code will search for a sim.ini file inside the folder path given as input and will start the model using the file's specifics training it ans saving it on that folder. The evaluation script works the same.

## Citation
If you use this code please cite the following paper:
```
@misc{Leoni2024,
author = {Leoni, Luca and Franchini, Cesare},
doi = {10.48550/arXiv.2402.00736},
month = feb,
title = {{Global sampling of Feynman's diagrams through Normalizing Flow}},
url = {https://arxiv.org/abs/2402.00736},
year = {2024}
}
```
