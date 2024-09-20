import os
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

# Trial configurations
all_trials_info = {
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

def train_model_per_trial(device_rf_data, all_trials_info, model_types, model_params, methods):
    results = {}
    overall_weighted_accuracy = {}

    for trial_name, devices in all_trials_info.items():  # This iterates over each trial
        authorized_devices = devices['authorized']
        rogue_devices = devices['rogue']
        trial_weighted_accuracy = []

        for device_id in tqdm(authorized_devices, desc=f"Training for {trial_name}"):
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

            for method in methods:
                for model_type, params in model_params.items():
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

                            model = get_model(model_type, **params)
                            model.fit(X_train, y_train)

                            y_pred_auth = model.predict(X_test)
                            auth_TDR, auth_FDR, misclassified_auth = evaluate_model_criteria(
                                y_test, y_pred_auth, positive_label=1,
                                sample_device_mapping=[sample_device_mapping_auth[i] for i in test_index]
                            )
                            y_pred_rogue = model.predict(X_rogue_selected)
                            rogue_TDR, rogue_FDR, misclassified_rogue = evaluate_model_criteria(
                                y_rogue, y_pred_rogue, positive_label=0, sample_device_mapping=sample_device_mapping_rogue
                            )

                            score = calculate_closeness_score(auth_TDR, auth_FDR, rogue_TDR, rogue_FDR)
                            weighted_accuracy = (auth_TDR * 0.5 + rogue_TDR * 0.5)

                            if auth_TDR >= 0.95 and auth_FDR <= 0.05 and rogue_TDR >= 0.95 and rogue_FDR <= 0.05:
                                results[(device_id, trial_name, model_type, method)] = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_TDR': auth_TDR,
                                    'auth_FDR': auth_FDR,
                                    'rogue_TDR': rogue_TDR,
                                    'rogue_FDR': rogue_FDR,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                trial_weighted_accuracy.append(weighted_accuracy)
                                print(f"Device {device_id} in {trial_name} with model {model_type} and FS method {method}: Model meets criteria with TDR: {auth_TDR}, FDR: {auth_FDR} for Authorized, and TDR: {rogue_TDR} for Malicious.")
                                save_model(device_id, trial_name, model, selected_features_indices, model_type, method)
                                break
                            elif score < best_score:
                                best_score = score
                                best_model_details = {
                                    'trial_name': trial_name,
                                    'optimal_n_features': n_features_to_select,
                                    'auth_TDR': auth_TDR,
                                    'auth_FDR': auth_FDR,
                                    'rogue_TDR': rogue_TDR,
                                    'rogue_FDR': rogue_FDR,
                                    'model': model,
                                    'features_indices': selected_features_indices,
                                    'accuracy': weighted_accuracy,
                                    'misclassified_auth': misclassified_auth,
                                    'misclassified_rogue': misclassified_rogue
                                }
                                best_weighted_accuracy = weighted_accuracy

                        if (device_id, trial_name, model_type, method) in results:
                            break

                    if (device_id, trial_name, model_type, method) not in results and best_model_details is not None:
                        results[(device_id, trial_name, model_type, method)] = best_model_details
                        trial_weighted_accuracy.append(best_weighted_accuracy)
                        print(f"Device {device_id} in {trial_name} with model {model_type} and FS method {method}: No model met full criteria. Using closest model with TDR: {best_model_details['auth_TDR']}, FDR: {best_model_details['auth_FDR']} for Authorized, and TDR: {best_model_details['rogue_TDR']} for Malicious.")
                        save_model(device_id, trial_name, best_model_details['model'], best_model_details['features_indices'], model_type, method)

        # overall_weighted_accuracy[trial_name] = trial_weighted_accuracy

    return results
def save_model(device_id, trial_name, model, selected_features_indices, model_type, method):
    trial_dir = os.path.join('model_results', trial_name, model_type, method)
    os.makedirs(trial_dir, exist_ok=True)

    model_details = {
        'model': model,
        'features_indices': selected_features_indices
    }
    filename = os.path.join(trial_dir, f'model_{device_id}.joblib')
    joblib.dump(model_details, filename)
    print(f"Model and features indices saved: {filename}")

# Example call to train_model_per_trial
device_rf_data = {device: X[y == idx] for device, idx in device_id_mapping.items()}


# Parameters for Random Forest model
model_params_rnf = {
    'n_estimators': 10,       # Number of trees in the forest
    'max_depth': 8,           # Maximum depth of each tree
    'random_state': 42,       # Random seed for reproducibility
    'n_jobs': 1               # Number of parallel jobs to run (1 means no parallelism)
}

# Parameters for Support Vector Classifier (SVC) model
model_params_svc = {
    'kernel': 'poly',         # Specifies the kernel type to be used in the algorithm ('poly' for polynomial)
    'C': 1.0,                 # Regularization parameter. The strength of the regularization is inversely proportional to C
    'probability': True       # Whether to enable probability estimates
}

# Parameters for K-Nearest Neighbors (KNN) model
model_params_knn = {
    'n_neighbors': 5          # Number of neighbors to use for k-neighbors queries
}

# Parameters for XGBoost model
model_params_xgb = {
    'objective': 'binary:logistic',  # Specify the learning task and the corresponding learning objective
    'max_depth': 8,                  # Maximum depth of a tree (controls the complexity of the model)
    'n_estimators': 10,              # Number of boosting rounds (trees) to be added
    'learning_rate': 0.1,            # Step size shrinkage used in updates to prevent overfitting
    'subsample': 0.7,                # Subsample ratio of the training instance (controls overfitting)
    'colsample_bytree': 0.7,         # Subsample ratio of columns when constructing each tree
    'random_state': 42               # Random seed for reproducibility
}

# Parameters for Gradient Boosting model
model_params_gb = {
    'n_estimators': 10,       # Number of boosting stages to be run
    'learning_rate': 0.1,     # Learning rate shrinks the contribution of each tree
    'max_depth': 8,           # Maximum depth of the individual trees
    'random_state': 42        # Random seed for reproducibility
}

# Parameters for Logistic Regression model
model_params_lr = {
    'C': 1.0,                 # Inverse of regularization strength; must be a positive float
    'random_state': 42,       # Random seed for reproducibility
    'solver': 'liblinear',    # Algorithm to use in the optimization problem (liblinear is good for small datasets)
    'max_iter': 500           # Maximum number of iterations taken for the solvers to converge
}

# Dictionary of model types and their corresponding parameters
model_params = {
    'random_forest': model_params_rnf,
    'svc': model_params_svc,
    'logistic_regression': model_params_lr,
    'knn': model_params_knn,
    'xgb': model_params_xgb,
    'gradient_boosting': model_params_gb,
}


# Dictionary of model types and their corresponding parameters
model_params = {
    'random_forest': model_params_rnf,
    'svc': model_params_svc,
    'logistic_regression': model_params_lr,
    'knn': model_params_knn,
    'xgb': model_params_xgb,
    'gradient_boosting': model_params_gb,
}

# List of feature selection methods
methods = ['anova']  # Add your feature selection methods here

# Train the model for each trial using all model types and feature selection methods
results, overall_weighted_accuracy = train_model_per_trial(device_rf_data, all_trials_info, model_params.keys(), model_params, methods)



