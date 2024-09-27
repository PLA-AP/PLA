# **Physical Layer Authentication (PLA) Approach**

## **Description**

Physical Layer Authentication (PLA) is a technique that leverages the unique characteristics of physical signals to authenticate devices in wireless communication systems. This repository contains the code, datasets, and documentation necessary to reproduce the results presented in our paper `Robust Device Authentication in Multi-Node Networks: ML-Assisted Hybrid PLA
Exploiting Hardware Impairments` submitted to the ACSAC conference.

## **Abstract**

This repository contains the implementation of a hybrid physical layer authentication (PLA) method designed to enhance security in multi-node networks. The approach leverages inherent hardware impairments such as carrier frequency offset (CFO), direct current offset (DCO), and phase offset (PO) as features to authenticate devices and detect unauthorized access.

Machine learning (ML) models are utilized to perform device authentication without requiring prior knowledge of potential attacks. The effectiveness of this approach has been validated through experimental evaluations on a commercial software-defined radio (SDR) platform, achieving high authentication rates and reliable detection of malicious devices. 

This repository includes all necessary code, datasets, and documentation to reproduce the results and explore the PLA method under different scenarios.

## **Supported Environments and Hardware Requirements**
The code and models are supported on the following environments and hardware configurations:

- **`Operating Systems`**: Windows 11/Ubuntu 22.04.
- **`Processor`**: 12th Gen Intel® Core™ i7-12800H, 14 cores, 2.4 GHz base clock speed.
- **`Memory`**: 32 GB RAM.
- **`Python Version`**: 3.x 
- **`Specialized Hardware`**: BladeRF AX4 for data collection.


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

- **`results`**: Contains the results for each trial, saved in .json files.

- **`results_noise`**: Contains the results for each trial at different SNR levels, saved in .json files.

- **`RF_Fingerprint.py`**: The main script for training and evaluation of RF fingerprinting models.
- **`Plotting_Results.py`**: The script used to generate figures from the results.
- **`rf_fingerprint_env.yml`**: YAML file for setting up the same environment using Conda.

## **Dataset**

The data used in this project are in I/Q format, collected using the BladeRF AX4 and GNU Radio software. It consists of raw signal data captured during various stages of the experiment. The dataset is crucial for reproducing the experiments and validating the PLA approach. 

**Note:** Due to the size of the dataset, it is stored in an external Drive. You can download the dataset from the following link: 
[Download Dataset from Google Drive](https://drive.google.com/file/d/1Hj6V6LVJnZMDRaQczt9gFOyiJhImjWnx/view?pli=1)

## **How to Run**

To reproduce the main experiment presented in the paper, follow these steps:

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python version**: 3.x
- **Dependencies**: All necessary dependencies are listed in the `rf_fingerprint_env.yml` file.

### Setting Up the Environment

To set up the environment, you can use the `rf_fingerprint_env.yml` file:

```bash
conda env create -f rf_fingerprint_env.yml
conda activate <your-environment-name>
```

### Data Preparation

The data required for training is provided in the `dataset/processed_fingerprints_data.h5` file. This file contains the processed RF fingerprint data for each device, which is used to train and validate the models.
### **Training and Evaluating the Model**

To run the experiments and automatically test different combinations of machine learning models, feature selection methods, and scenarios, simply execute the following script:

```bash
python RF_Fingerprint.py --trial trial_x --model x --feature x
```
You can customize the execution by specifying trials, models, and feature selection methods through the following command-line arguments:
#### **Command-Line Arguments**
- **`--trial`**:  Specify one or more trials to run (e.g., `trial_1`, `trial_2`, and `trial_3`). Leave this argument blank to run all available trials.
- **`--model`**: Choose specific machine learning models to use (e.g., `random_forest`, `svc`, `knn`, `xgb`, and `logistic_regression`). Leave blank to test all models.
- **`--feature`**:  Select one or more feature selection methods (e.g., `pca`, `anova`, `mutual_info`, and `rfe` ). Leave this argument blank to test all available feature selection methods.
- **`--run_all`**:  The script will run all combinations of trials, models, and feature selection methods.
##### **Example Usage**
Here are some examples of how to run the experiments with different configurations:

##### 1. **Run All Combinations of Trials, Models, and Features**

   To run all trials, models, and feature selection methods, use the following command:

   ```bash
   python RF_Fingerprint.py --run_all
   ```
##### 2. **Run a Specific Trial with Specific Models and Features**
To run `trial_1` and `trial_2` using the `random_forest` and `svc` models, with `pca` and `anova` for feature selection:
   ```bash
python RF_Fingerprint.py --trial trial_1 trial_2 --model random_forest svc --feature pca anova
```
##### 3. **Run a Specific Model for All Trials with All Feature Selection Methods**
To run a specific model (e.g., `random_forest`) for all trials with all feature selection methods, use the following command:

```bash
python RF_Fingerprint.py --model random_forest
```
##### 4. **Run All Models for All Trials with a Specific Feature Selection Method**
To run all trials and models but only use pca for feature selection: 
```bash
python RF_Fingerprint.py --feature pca
```

### **Expected Output**

When you run the `RF_Fingerprint.py` script with the default settings, you will obtain the following performance metrics:

- **TDR (True Detection Rate)** for authorized nodes.
- **FDR (False Detection Rate)** for authorized nodes.
- **TDR (True Detection Rate)** for malicious nodes.

### **Estimated Time to Run Experiments**
Below are approximate times for running the experiments based on our hardware configuration (e.g., Intel Xeon Gold 6226 CPU @ 2.70GHz, 16GB RAM):

| Model          | Feature Selection | Trial 1   | Trial 2   | Trial 3   |
|----------------|-------------------|-----------|-----------|-----------|
| **Random Forest** | ANOVA           | 00:17     | 00:24     | 01:11     |
|                | MI                | 03:09     | 01:16     | 17:30     |
|                | PCA               | 01:34     | 03:40     | 05:20     |
|                | RFE               | 1:26:42   | 1:54:03   | 4:50:35   |
| **SVC**        | ANOVA             | 00:31     | 00:54     | 02:05     |
|                | MI                | 18:49     | 26:44     | 58:11     |
|                | PCA               | 00:43     | 01:07     | 02:18     |
|                | RFE               | 3:10:38   | 4:39:52   | 5:44:05   |
| **KNN**        | ANOVA             | 00:56     | 00:15     | 01:24       |
|                | MI                | 24:42     | 24:22     |34:37       |
|                | PCA               | 00:57     | 01:17     | 01:41       |
|                | RFE               | 3:38:47   | 1:55:18   |6:16:24       |
| **XGB**        | ANOVA             | 02:14     | 03:04     |04:32       |
|                | MI                | 34:13     | 50:49     |58:32       |
|                | PCA               | 01:57     | 03:02     |03:22      |
|                | RFE               | 3:56:04   | 4:42:09   |7:20:31       |
| **Logistic Regression** | ANOVA     | 00:37     | 00:34     |00:48       |
|                | MI                | 22:18     | 25:50     |40:05      |
|                | PCA               | 00:17     | 01:00     |01:22      |
|                | RFE               | 3:39:36   | 3:34:49   | N/A       |




> **Note**: The time may vary depending on the hardware and specific system setup.




<!--
### Training the Model

To train the RF fingerprinting model, run the `RF_Fingerprint.py` script:

```bash
python RF_Fingerprint.py
```

This script will load the processed data, train the model, and validate its performance.

### Evaluating the Model

Evaluation of the model's performance can be done using the functions provided in the `evaluation.py` module. The script `RF_Fingerprint.py` includes an evaluation phase, but you can run additional evaluations as needed.
-->

## **Main Results**

This section summarizes the main results you should expect when running the provided code, as discussed in our paper. The script now automates the testing of different machine learning models, feature selection methods, and scenarios.


### **Key Results and Figures**

For each combination of machine learning models and feature selection methods, we have stored the best-performing models in `.joblib` files. These can be downloaded from [Download Saved Model](https://drive.google.com/file/d/1DYO9NHnAmMnK_zhbacqEtEP-sY11uLk4/view?usp=drive_link). All results, including those for every combination of machine learning models, feature selection methods, and scenarios, are available in the `results` folder. Additionally, the results corresponding to different SNR values are saved in the `results_noise` folder. The dataset used for different SNR values can be downloaded from [Download Noise Dataset](https://drive.google.com/file/d/1AO0Pwg1gGaDCQ0_R6aV1NH8qUYoLAE54/view?usp=drive_link).

To generate the key figures from the paper, run the following script:
```bash
python Plotting_Results.py

```
This script will generate the main figures using the saved models and their associated performance metrics.

The following figure illustrates the TDR and FDR for different combinations of FS methods and ML models across three scenarios.
![Authentication Rate for different combinations of FS and ML models](Main%20Results.png)

Each subfigure shows the performance across the three scenarios (Scenario 1, Scenario 2, Scenario 3) and for different devices. The results demonstrate the variations in detection rates depending on the combination of FS methods and ML models used.

For a deeper analysis of the performance and detailed discussion, please refer to **Section 5.2.1. Performance Analysis** in the paper.

## **Documentation and Support**

For detailed instructions on setting up the environment, running experiments, and interpreting results, please refer to the documentation and the paper. If you encounter any issues or have questions, feel free to open an issue in this repository.

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
