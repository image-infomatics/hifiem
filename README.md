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

# FAQ
## How many scales are in step 1 of de-stripping? How do you determine how many scales are needed?
In the process of decomposing the input image into multiple scales using a bank of separable filters, the determination of the number of scales is guided by the imperative to effectively address stripe artifacts. The decision on how many scales to employ is based on observing the presence of stripes in the approximation (low-pass component) at each scale. As a standard practice, we systematically review the approximation at different scales until we are certain that the resulting low-pass component sufficiently minimizes stripe artifacts. In our experience, this determination often converges around approximately 6 scales. It is worth noting that exceeding the sufficient number of scales is generally safe, as it tends to have a minor, if any, negative impact on the overall improvement. However, increasing the number of scales beyond the critical point involves unnecessary computations without meaningful improvement in artifact reduction. Therefore, the chosen number of scales strikes a balance between computational efficiency and achieving the desired outcome in terms of stripe artifact suppression.

## How is the rectangular block filter size N determined?  What values were used here? How many N needed to be tried to figure out the “best”
During the experimental phase of our study, determining the "best" value for the rectangular block filter size N, the size of the rectangular block filter, played a crucial role in optimizing performance at each scale. We systematically explored various values for N to evaluate their impact on weight map smoothing and, consequently, the overall algorithm performance. A heuristic rule guided these selections, ensuring that the size N of the block filter is meaningfully less than the compound Field of View of the convolution kernel used to decompose each scale. For each scale, experiments were conducted with different values of N, observing the effects on the algorithm's performance. After thorough experimentation, values in the following Table were selected for optimal performance at each scale for our EM datasets.
| Scale        | FoV           | Block Filter Size N |
| ------------- |:-------------:| -----:|
| 1      | 7 | 5 |
| 2      | 19 | 9 |
| 3      | 43 | 13 |
| 4      | 91 | 19 |
| 5      | 187 | 43|
| 6      | 379 | 75 |
| 7      | 763 | 91 |


## How would the neighborhood size Ω vary with resolution?
The primary goal of our contrast adjustment approach is to enhance the visibility of specific neuropil structures, such as plasma membranes and T-bars. The size of these structures increases with rising resolution. However, this doesn't necessarily imply the need for an increase in the neighborhood size (Ω), as higher resolution tends to improve overall image quality and details. Consequently, the demand for contrast enhancement may decrease with higher spatial resolution. Notably, the selection of the neighborhood size (Ω) is not solely contingent on resolution but is influenced by a diverse range of image characteristics.

In our methodology, we anticipate a relatively small neighborhood size, typically ranging from 5x5 to 13x13 pixels. The precise size is determined empirically through experimentation and visual inspection of the achieved improvements.


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
