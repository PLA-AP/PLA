# **Physical Layer Authentication (PLA) Approach**

## **Description**

Physical Layer Authentication (PLA) is a technique that leverages the unique characteristics of physical signals to authenticate devices in wireless communication systems. This repository contains the code, datasets, and documentation necessary to reproduce the results presented in our paper `Robust Device Authentication in Multi-Node Networks: ML-Assisted Hybrid PLA
Exploiting Hardware Impairments` submitted to the ACSAC conference.

## **Abstract**

This repository contains the implementation of a hybrid physical layer authentication (PLA) method designed to enhance security in multi-node networks. The approach leverages inherent hardware impairments such as carrier frequency offset (CFO), direct current offset (DCO), and phase offset (PO) as features to authenticate devices and detect unauthorized access.

Machine learning (ML) models are utilized to perform device authentication without requiring prior knowledge of potential attacks. The effectiveness of this approach has been validated through experimental evaluations on a commercial software-defined radio (SDR) platform, achieving high authentication rates and reliable detection of malicious devices. 

This repository includes all necessary code, datasets, and documentation to reproduce the results and explore the PLA method under different scenarios.

## **Repository Structure**

The structure of the repository is aligned with the process flow of our PLA approach, as illustrated in the following diagram:

![PLA Approach Process Flow](Setup_Fingerprint.png)

This repository is organized into the following folders:
- **`src/`**: Contains all the functions and modules necessary for training and validating the models.
  - **`dataloader.py`**: Handles the loading and management of datasets.
  - **`evaluation.py`**: Provides functions for evaluating the performance of the models.
  - **`features_generation.py`**: Includes methods for generating features from raw data.
  - **`features_selections.py`**: Contains functions for selecting the most relevant features.
  - **`preprocessing.py`**: Handles the preprocessing of raw data.

- **`dataset/`**: Contains the processed data and any datasets used for training and validation.
  - **`processed_fingerprints_data.h5`**: The HDF5 file containing processed RF fingerprint data for each device.

- **`RF_Fingerprint.py`**: The main script for training the RF fingerprinting models.

- **`rf_fingerprint_env.yml`**: YAML file for setting up the same environment using Conda.

<!--
- **data/**: Contains the dataset folder, which includes the data captured for this project. Due to the large size of the dataset, it is hosted externally. You can download the dataset from the following link:

- [Download Dataset from Google Drive]([https://drive.google.com/file/d/1t1jih0RLrD_XSyBUC3d8pBvOSxNzEbHS/view?usp=drive_link](https://drive.google.com/file/d/1Hj6V6LVJnZMDRaQczt9gFOyiJhImjWnx/view?pli=1)), see the [Dataset README](Dataset%20README.md).

- **nfc_rfml/**: Contains the source code used to preprocess the signals, train the machine learning models, and test their performance. Detailed usage instructions can be found in the [project README](nfc_rfml/README.md).

- **notebooks/**: Houses the Jupyter notebooks used during the analysis and prototyping phases of the project. These notebooks provide insights into the exploratory data analysis and model prototyping processes.

- **report/**: Contains the LaTeX source files and the compiled PDF of the report. This folder also includes the bibliography and the figures used in the report.

- **scripts/**: Contains small utility programs, such as the script used for data acquisition. These scripts are essential for setting up the experimental environment.
-->

## **Dataset**

The data used in this project are in I/Q format, collected using the BladeRF AX4 and GNU Radio software. It consists of raw signal data captured during various stages of the experiment. The dataset is crucial for reproducing the experiments and validating the PLA approach. 

**Note:** Due to the large size of the dataset, it is stored in an external Drive. You can download the dataset from the following link: 
[Download Dataset from Google Drive](https://drive.google.com/file/d/1Hj6V6LVJnZMDRaQczt9gFOyiJhImjWnx/view?pli=1)

## **How to Run**

To reproduce the main experiment presented in the paper, follow these steps:

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python version**: 3.x
- **Dependencies**: All necessary dependencies are listed in the `training_env.yml` file.

### Setting Up the Environment

To set up the environment, you can use the `training_env.yml` file:

```bash
conda env create -f training_env.yml
conda activate <your-environment-name>
```

### Data Preparation

The data required for training is provided in the `dataset/processed_data.h5` file. This file contains the processed RF fingerprint data for each device, which is used to train the models.

### Training the Model

To train the RF fingerprinting model, run the `rf_fingerprint_training.py` script:

```bash
python rf_fingerprint_training.py
```

This script will load the processed data, train the model, and validate its performance.

### Evaluating the Model

Evaluation of the model's performance can be done using the functions provided in the `evaluation.py` module. The script `rf_fingerprint_training.py` includes an evaluation phase, but you can run additional evaluations as needed.

## **Documentation and Support**

For detailed instructions on setting up the environment, running the experiments, and interpreting the results, please refer to the documentation within each folder. If you encounter any issues or have questions, feel free to open an issue in this repository.
## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{YourArticle,
  title={Your Title},
  author={Your Name and Others},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XX-XX},
  doi={XX.XXXX/XXXXXX},
}
```
