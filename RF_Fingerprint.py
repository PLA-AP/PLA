import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from src.dataloader import load_data_and_unique_labels, load_data_from_hdf5
from src.evaluation import evaluate_model_criteria, calculate_closeness_score
from src.features_selections import select_features
import joblib

# Define data paths and processed data file
DATA_DIRECTORY = r'dataset'  # Directory containing the raw data files
PROCESSED_DATA_PATH = r'dataset/processed_fingerprints_data.h5'  # Path to save the processed fingerprints data

# Check if processed data exists
if os.path.exists(PROCESSED_DATA_PATH):
    # Load processed data from HDF5 file
    X, y, device_id_mapping, total_devices = load_data_from_hdf5(PROCESSED_DATA_PATH)
else:
    # Load the data and unique labels
    X, y, device_id_mapping, total_devices = load_data_and_unique_labels(
        DATA_DIRECTORY,
        sample_rate=20e6,
        cutoff_freq=5e6,
        M=150,
        KG=150,
        N=1,
        NP=100,
        NT=15,
        NF=15,
        transient_threshold=0.38,
        specific_duration_threshold=0.005,
        specific_magnitude_threshold=0.3,
        min_transient_duration=0.005,
        filter_type='chebyshev',
        filter_order=4,
        filter_ripple=0.5,
        mode='diagonal',
        save_path=PROCESSED_DATA_PATH
    )

# Example trial configuration (authorized and rogue devices)
trials_info = {
    'trial_1': {
        'authorized': ['device3', 'device2', 'device12', 'device9'],
        'rogue': ['device8', 'device11', 'device5', 'device1', 'device6', 'device10', 'device7', 'device4']
    },
    'trial_2': {
        'authorized': ['device10', 'device6', 'device12', 'device3', 'device11', 'device1'],
        'rogue': ['device2', 'device5', 'device8', 'device7', 'device4', 'device9']
    },
    'trial_3': {
        'authorized': ['device1', 'device10', 'device9', 'device8', 'device12', 'device11', 'device6', 'device7'],
        'rogue': ['device4', 'device2', 'device5', 'device3']
    }
}

def get_model(model_type, **kwargs):
    if model_type == 'svc':
        return SVC(**kwargs)
    elif model_type == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_type == 'knn':
        return KNeighborsClassifier(**kwargs)
    elif model_type == 'xgb':
        return XGBClassifier(**kwargs)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**kwargs)
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_adr_for_trial(results):
    trial_adr = {}
    for device_id, device_results in results.items():
        auth_tvr = device_results['auth_tvr']
        rogue_tvr = device_results['rogue_tvr']
        adr = (auth_tvr + rogue_tvr) / 2
        trial_adr[device_id] = adr
    overall_adr = np.mean(list(trial_adr.values()))
    return trial_adr, overall_adr

def save_results_to_json(trial_name, model_type, method, results, overall_adr, output_dir='results'):
    # Create a folder for each trial and each model type within the main output directory
    trial_dir = os.path.join(output_dir, trial_name, model_type) 
    os.makedirs(trial_dir, exist_ok=True)  # Create trial- and model-specific directory if it doesn't exist
    
    # Define the filename including method
    filename = os.path.join(trial_dir, f'{method}_results.json')  # File name based on the method

    # Prepare the data in the structured format for JSON saving
    results_data = {
        'trial_name': trial_name,
        'model_type': model_type,
        'method': method,
        'overall_adr': overall_adr,
        'devices': {}
    }

    for device_id, result in results.items():
        results_data['devices'][device_id] = {
            'auth_tvr': result['auth_tvr'],
            'auth_fvr': result['auth_fvr'],
            'rogue_tvr': result['rogue_tvr']
        }

    # Save the results in a JSON file
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=4)

    print(f"Results saved to {filename}")

def train_model_per_trial(device_rf_data, trials_info, model_configs, feature_selection_methods):
    all_results = {}
    all_adr_results = {}

    for trial_name, devices in trials_info.items():
        authorized_devices = devices['authorized']
        rogue_devices = devices['rogue']

        for model_type, model_params in model_configs.items():
            for method in feature_selection_methods:
                results = {}
                trial_weighted_accuracy = []

                for device_id in tqdm(authorized_devices, desc=f"Training {model_type} with {method} for {trial_name}"):
                    X_auth, y_auth, X_rogue = [], [], []
                    sample_device_mapping_auth, sample_device_mapping_rogue = [], []

                    for did, rf_data in device_rf_data.items():
                        for feature_vector in rf_data:
                            if did == device_id:
                                X_auth.append(feature_vector)
                                y_auth.append(1)
                                sample_device_mapping_auth.append(did)
                            elif did in authorized_devices:
                                X_auth.append(feature_vector)
                                y_auth.append(0)
                                sample_device_mapping_auth.append(did)
                            if did in rogue_devices:
                                X_rogue.append(feature_vector)
                                sample_device_mapping_rogue.append(did)

                    X_auth = np.array(X_auth)
                    y_auth = np.array(y_auth)
                    X_rogue = np.array(X_rogue)
                    y_rogue = np.zeros(len(X_rogue), dtype=int)

                    best_score = float('inf')
                    best_model_details = None
                    best_weighted_accuracy = None

                    for n_features_to_select in range(1, X_auth.shape[1] + 1):
                        X_auth_selected, selected_features_indices = select_features(X_auth, y_auth, n_features_to_select, method=method)
                        X_rogue_selected = X_rogue[:, selected_features_indices]

                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        for train_index, test_index in skf.split(X_auth_selected, y_auth):
                            X_train, X_test = X_auth_selected[train_index], X_auth_selected[test_index]
                            y_train, y_test = y_auth[train_index], y_auth[test_index]

                            model = get_model(model_type, **model_params)
                            model.fit(X_train, y_train)

                            y_pred_auth = model.predict(X_test)
                            auth_tvr, auth_fvr, misclassified_auth = evaluate_model_criteria(
                                y_test, y_pred_auth, positive_label=1,
                                sample_device_mapping=[sample_device_mapping_auth[i] for i in test_index]
                            )
                            y_pred_rogue = model.predict(X_rogue_selected)
                            rogue_tvr, rogue_fvr, misclassified_rogue = evaluate_model_criteria(
                                y_rogue, y_pred_rogue, positive_label=0, sample_device_mapping=sample_device_mapping_rogue
                            )

                            score = calculate_closeness_score(auth_tvr, auth_fvr, rogue_tvr, rogue_fvr)
                            weighted_accuracy = (auth_tvr * 0.5 + rogue_tvr * 0.5)

                            if auth_tvr >= 0.95 and auth_fvr <= 0.05 and rogue_tvr >= 0.95 and rogue_fvr <= 0.05:
                                results[device_id] = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_tvr': auth_tvr,
                                    'auth_fvr': auth_fvr,
                                    'rogue_tvr': rogue_tvr,
                                    'rogue_fvr': rogue_fvr,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                trial_weighted_accuracy.append(weighted_accuracy)
                                print(f"Device {device_id} in Trial {trial_name}: Model meets criteria with TDR: {auth_tvr}, FDR: {auth_fvr} for Authorized, and TDR: {rogue_tvr} for Malicious.")
                                save_model(device_id, trial_name, model, selected_features_indices, model_type, method)
                                break
                            elif score < best_score:
                                best_score = score
                                best_model_details = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_tvr': auth_tvr,
                                    'auth_fvr': auth_fvr,
                                    'rogue_tvr': rogue_tvr,
                                    'rogue_fvr': rogue_fvr,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                best_weighted_accuracy = weighted_accuracy

                        if device_id in results:
                            break

                    if device_id not in results and best_model_details is not None:
                        results[device_id] = best_model_details
                        trial_weighted_accuracy.append(best_weighted_accuracy)
                        print(f"Device {device_id} in Trial {trial_name}: No model met full criteria. Using closest model with TDR: {best_model_details['auth_tvr']}, FDR: {best_model_details['auth_fvr']} for Authorized, and TDR : {best_model_details['rogue_tvr']} for Malicious.")
                        # Save the closest model as well
                        save_model(device_id, trial_name, best_model_details['model'], best_model_details['features_indices'], model_type, method)

                # Calculate ADR for the trial and combination
                trial_adr, overall_adr = calculate_adr_for_trial(results)
                all_adr_results[(trial_name, model_type, method)] = {
                    'adr_per_device': trial_adr,
                    'overall_adr': overall_adr
                }

                # Output the overall ADR for the trial and combination
                print(f"Average Detection Rate for Trial {trial_name} with {model_type} and {method}: {overall_adr}")

                # Save the results to JSON
                save_results_to_json(trial_name, model_type, method, results, overall_adr)

                # Store the results for this combination
                all_results[(trial_name, model_type, method)] = {
                    'results': results,
                    'overall_adr': overall_adr
                }

    return all_results, all_adr_results

def save_model(device_id, trial_name, model, selected_features_indices, model_type, method):
    """
    Saves the trained model and selected feature indices to a file using joblib.

    Args:
    - device_id (str): Device ID.
    - trial_name (str): Trial name.
    - model (sklearn.base.BaseEstimator): Trained model.
    - selected_features_indices (np.ndarray): Indices of selected features.
    - model_type (str): Type of model used.
    - method (str): Feature selection method used.
    """
    # Create the directory for the trial and combination if it doesn't exist
    trial_dir = os.path.join('model_saved', trial_name, model_type, method)
    os.makedirs(trial_dir, exist_ok=True)

    model_details = {
        'model': model,
        'features_indices': selected_features_indices
    }
    filename = os.path.join(trial_dir, f'model_{device_id}.joblib')
    joblib.dump(model_details, filename)
    print(f"Model and features indices saved: {filename}")


# Define all the model configurations and feature selection methods
model_configs = {
    'random_forest': {
        'n_estimators': 10,
        'max_depth': 8,
        'random_state': 42,
        'n_jobs': 1
    },
    'svc': {
        'kernel': 'poly',
        'C': 1.0,
        'probability': True
    },
    'knn': {
        'n_neighbors': 5
    },
    'xgb': {
        'objective': 'binary:logistic',
        'max_depth': 8,
        'n_estimators': 10,
        'learning_rate': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 10,
        'learning_rate': 0.1,
        'max_depth': 8,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1.0,
        'random_state': 42,
        'solver': 'liblinear',
        'max_iter': 500
    }
}

feature_selection_methods = ['pca', 'mutual_info','anova','rfe']

# Example call to train_model_per_trial with all combinations
device_rf_data = {device: X[y == idx] for device, idx in device_id_mapping.items()}
all_results, all_adr_results = train_model_per_trial(device_rf_data, trials_info, model_configs, feature_selection_methods)

# After this execution, all results across all trials, model configurations, and feature selection methods will be generated, saved, and printed.
