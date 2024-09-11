tioimport numpy as np
from sklearn.metrics import confusion_matrix

## Phase 4: Machine Learning Model Training and Evaluation

# Section 4. Model Evaluation (Refer to the paper for more details)

def evaluate_model_criteria(y_true, y_pred, positive_label=1, sample_device_mapping=None):
    """
    Evaluates model performance based on true positive and false positive rates.
    Extends the evaluation to include device IDs for misclassified samples.

    Args:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - positive_label (int): The label considered as positive class (default is 1).
    - sample_device_mapping (list): Optional. List mapping each sample index to its device ID.

    Returns:
    - Tuple[float, float, list]: True Detection Rate (TDR), False Detection Rate (FDR), 
      and a list of misclassified sample indices or tuples of (index, device ID) if sample_device_mapping is provided.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, 1-positive_label])
    
    # Calculate True Detection Rate (TDR) and False Detection Rate (FDR)
    TDR = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
    FDR = cm[1, 0] / np.sum(cm[1, :]) if np.sum(cm[1, :]) > 0 else 0
    
    # Identify misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Include device IDs for misclassified samples if sample_device_mapping is provided
    if sample_device_mapping is not None:
        misclassified_info = [(index, sample_device_mapping[index]) for index in misclassified_indices]
    else:
        misclassified_info = misclassified_indices.tolist()
    
    return TDR, FDR, misclassified_info

def calculate_closeness_score(auth_tvr, auth_fvr, rogue_tvr, rogue_fvr, tvr_threshold=0.95, fvr_threshold=0.05):
    """
    Calculates a score representing how close a model's performance is to the desired thresholds.

    Args:
    - auth_tvr (float): True positive rate for authenticated devices.
    - auth_fvr (float): False positive rate for authenticated devices.
    - rogue_tvr (float): True positive rate for rogue devices.
    - rogue_fvr (float): False positive rate for rogue devices.
    - tvr_threshold (float): Desired true positive rate threshold (default is 0.95).
    - fvr_threshold (float): Desired false positive rate threshold (default is 0.05).

    Returns:
    - float: Closeness score representing the model's performance relative to the desired thresholds.
    """
    # Calculate the closeness score
    score = (
        abs(auth_tvr - tvr_threshold) + 
        abs(auth_fvr - fvr_threshold) + 
        abs(rogue_tvr - tvr_threshold) + 
        abs(rogue_fvr - fvr_threshold)
    )
    return score
