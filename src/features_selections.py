from sklearn.decomposition import PCA
from skrebate import ReliefF
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import numpy as np

## Phase 3: Features Selection for devices

## Section 3.3. Features Selection (Refer to the paper for more details)

def select_features(X, y, n_features_to_select, method="anova"):
    """
    Dynamically selects the top N features based on the specified method.

    Args:
    - X (np.ndarray): Feature matrix.
    - y (np.array): Labels.
    - n_features_to_select (int): Number of top features to select.
    - method (str): The feature selection method to use. Options: "relieff", "mutual_info", "anova", "pca".

    Returns:
    - Tuple[np.ndarray, np.ndarray]: The reduced feature matrix and the indices of the selected features.
    """
    if method == "relieff":
        # Feature Selection using ReliefF algorithm
        relief = ReliefF(n_neighbors=100)
        relief.fit(X, y)
        top_features_indices = np.argsort(relief.feature_importances_)[::-1][:n_features_to_select]
        return X[:, top_features_indices], top_features_indices
    
    elif method == "mutual_info":
        # Feature Selection using Mutual Information
        selector = SelectKBest(mutual_info_classif, k=n_features_to_select)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        return X_selected, selected_indices

    elif method == "anova":
        # Feature Selection using ANOVA F-value
        selector = SelectKBest(f_classif, k=n_features_to_select)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        return X_selected, selected_indices

    elif method == "pca":
        # Feature Selection using Principal Component Analysis (PCA)
        max_components = min(X.shape[0], X.shape[1], n_features_to_select)
        pca = PCA(n_components=max_components, random_state=42)
        X_selected = pca.fit_transform(X)
        selected_indices = np.arange(max_components)
        return X_selected, selected_indices

    else:
        raise ValueError(f"Unknown method: {method}. Available methods: 'relieff', 'mutual_info', 'anova', 'pca'.")