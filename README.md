# Preprocessing Pipeline of Volume Electron Microscope Images

The software incorporates various methods used in a multi-step pre-processing pipeline specifically designed to improve the image quality of specimen #3. These methods serve as essential components within the broader pipeline, which can be executed on a computational cluster. The input image consists of a stack of 11260 re-aligned sections, each with dimensions of 13750 x 9000 pixels. The image files are stored in .tif format, and the intensity values are represented as unsigned 16-bit. It is worth noting that the original intensity values are inverted. Additionally, the actual dynamic range of the images is notably lower than the available range of intensity.

## Directory Structure

The project directory is organized as follows:
```  
├── data/ # Directory for storing the data files  
│ ├── 05600.tif # section 5600 of the aligned dataset used as the original data for pre-processing pipeline  
│ ├── info.h5 # hdf5 file contains the intensity correction offset for each section
│ ├── polymodel.h5 # hdf5 file contains polynomial fitting coefficients used background correction
│ └── ...
│
├── python/ # Directory for python files containing python code for image processing algoritms
│ ├── fltemd.py # python module containing main methods and algorithms
│ ├── fltlib.py # python module containing helpers and utilities functions and classes 
│ └── ...
│
├── README.md # Project README file
├── requirements.txt # List of project dependencies
├── requirements.yml # yml-file for project environment
├── LICENSE # License file for the project
├── pipeline.ipynb # Jupyter notebook
└── ...
```

## Files

Here is a brief description of the main files in the project:

- **data/**: This directory contains the data files used as input for different steps of pipeline

- **python/**: This directory contains python modules, which provide methods and algorithms used in pre-processing pipeline. The `fltemd.py` covers image-processing methods and algorithms, while the `fltlib.py` provides more generic helpers and methods, including some helpers used in implementation of the full stack processing pipepline of Flatiron Cluster (*not covered by this notebook*).

- **README.md**: This file provides an overview of the project, including its purpose, installation instructions, and usage guidelines.

- **requirements.txt**: This file lists all the project dependencies, making it easier to set up the development environment with `pip`.

- **requirements.yml**: yml-file for project environment used by `conda update env` command

- **LICENSE.txt**: This file contains the project's license information, specifying the terms and conditions for using the code.

- **pipeline.ipynb**: jupyter notebook, which demonstrates the key methods used in pre-processing pipeline

### Download image section

    cd data 
    wget https://s3.us-east-2.amazonaws.com/ccn-connectomics-mega-ng/jwu/05600.tif

## Required python packages

Python modules in this project requires that the following packages are installed:
```
h5py>=3.6.0
matplotlib>=3.4.2
numpy>=1.20.3
pillow>=8.3.1
scikit-image>=0.18.1
scikit-learn>=0.24.2
scipy>=1.6.2
opencv>=4.5.5
```

# Publication
```bibtex
@article{kreinin2023high,
  title={High-fidelity Image Restoration of Large 3D Electron Microscopy Volume},
  author={Kreinin, Yuri and Gunn, Pat and Chklovskii, Dmitri and Jingpeng Wu},
  journal={bioRxiv},
  pages={2023--09},
  year={2023},
  publisher={Cold Spring Harbor Laboratory},
  doi = {10.1101/2023.09.14.557785}
}
```
