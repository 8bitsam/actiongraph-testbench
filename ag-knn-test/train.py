# train_ag.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
# RF parts removed as per current strategy focusing on k-NN with AG features
from sklearn.preprocessing import QuantileTransformer 
from sklearn.metrics import pairwise_distances
import joblib
import os
import time
import sys
import json
import argparse
from metrics import WeightedCosineDistance

# --- Configuration ---
DATA_DIR = "../Data/"
# Input: Featurized ActionGraph data
FEATURIZED_DATA_DIR_AG = os.path.join(DATA_DIR, "featurized-data-actiongraph/")
# Output: Models trained on AG features
MODEL_AG_BASE_DIR = os.path.join(DATA_DIR, "models_ag") 

TEST_SIZE = 0.25 # 75% train, 25% test
RANDOM_STATE = 42
# k-NN specific
N_NEIGHBORS = 1 # As specified for "database" like behavior

def run_train_ag(model_type='knn'): # model_type currently fixed to knn for this strategy
    """Loads AG-featurized data, trains k-NN model, saves artifacts."""
    if model_type != 'knn':
        print(f"Error: This script is configured for k-NN with ActionGraph features. Model type '{model_type}' not supported here.", file=sys.stderr)
        return False
        
    print(f"--- Running Training Step (Model: {model_type.upper()} with ActionGraph Features) ---")

    MODEL_DIR = os.path.join(MODEL_AG_BASE_DIR, f"{model_type}-model")
    print(f"Model artifacts will be saved to: {MODEL_DIR}")

    # --- Load Featurized ActionGraph Data ---
    print(f"\nLoading AG-featurized data from: {FEATURIZED_DATA_DIR_AG}")
    try:
        X_ag = np.load(os.path.join(FEATURIZED_DATA_DIR_AG, 'features_ag.npy'))
        # recipes_mp_style_for_ag.json contains MP-style dicts needed for evaluation compatibility
        with open(os.path.join(FEATURIZED_DATA_DIR_AG, 'recipes_mp_style_for_ag.json'), 'r') as f:
            # These are the "ground truth" recipes (precursors/operations) associated with each AG feature vector
            recipes_for_knn_retrieval = json.load(f) 
        
        if len(X_ag) != len(recipes_for_knn_retrieval):
             raise ValueError("Mismatch between AG features and associated MP-style recipes length.")
        print(f"Loaded {len(X_ag)} AG-featurized samples.")
        print(f"AG Feature array shape: {X_ag.shape}")
    except FileNotFoundError:
        print(f"Error: Featurized AG data not found in {FEATURIZED_DATA_DIR_AG}.", file=sys.stderr)
        print("Please run the 'featurize_ag' step first.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error loading featurized AG data: {e}", file=sys.stderr)
        return False

    # --- Train/Test Split ---
    # We need to split X_ag and maintain correspondence with recipes_for_knn_retrieval
    # The k-NN model is trained on X_train_ag_scaled.
    # During prediction, it finds neighbors in X_train_ag_scaled. The indices of these neighbors
    # map to the original indices within X_ag / recipes_for_knn_retrieval *before the split*.
    # So, we save the *original indices* of the training samples.
    print(f"\nPerforming train/test split (Test size: {TEST_SIZE})...")
    if len(X_ag) < 2: # Need at least 2 samples for a split
        print("Error: Not enough data points (< 2) for split.", file=sys.stderr)
        return False

    # Create an array of indices [0, 1, ..., N-1]
    original_indices_all = np.arange(len(X_ag)) 

    X_train_ag, _X_test_ag, train_original_indices, _test_original_indices = train_test_split(
        X_ag, original_indices_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Note: _X_test_ag and _test_original_indices are not directly used in *this* training script,
    # but the split consistency is important for evaluation.

    # recipes_for_knn_retrieval is not split here; it's indexed by train_original_indices later.
    # The k-NN model will be fitted on X_train_ag (after scaling).
    # y_train_recipes_for_knn (the actual MP-style recipes) are not directly part of k-NN fitting,
    # but are needed at evaluation time to know *what* recipe a neighbor corresponds to.
    # We save `train_original_indices` which allows evaluation to retrieve these from the full `recipes_for_knn_retrieval` list.

    print(f"Training set size for AG features: {len(X_train_ag)}")
    if len(X_train_ag) == 0:
        print("Error: Training set (AG features) has size 0 after split.", file=sys.stderr)
        return False

    # --- Scale Features ---
    print("\nFitting scaler on AG training data...")
    # QuantileTransformer can map to normal distribution, might help distance metrics for k-NN.
    # Using n_quantiles = min(1000, len(X_train_ag)) to avoid issues with small datasets.
    scaler_ag = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE, n_quantiles=max(1, min(1000, len(X_train_ag))))
    start_scale_time = time.time()
    try:
        scaler_ag.fit(X_train_ag) # Fit only on training data
    except ValueError as e: # e.g. if n_quantiles > n_samples
        print(f"Error fitting scaler: {e}. Ensure n_quantiles <= n_samples for QuantileTransformer.", file=sys.stderr)
        # Fallback or adjust n_quantiles if this happens often.
        # For now, assume n_quantiles adjustment above handles it.
        return False

    X_train_ag_scaled = scaler_ag.transform(X_train_ag)
    # Ensure no NaNs/Infs after scaling, can happen with some distributions or data issues
    X_train_ag_scaled = np.nan_to_num(X_train_ag_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    end_scale_time = time.time()
    print(f"Scaling took {end_scale_time - start_scale_time:.2f} seconds.")

    # --- Model Training (k-NN) ---
    start_train_time = time.time()
    model_artifacts = {'scaler_ag': scaler_ag}

    if model_type == 'knn':
        knn_model_ag = NearestNeighbors(
            n_neighbors=N_NEIGHBORS,
            metric='cosine',
            algorithm='brute',  # Explicitly use brute-force but enable parallelization
            n_jobs=-1           # Use all CPU cores
        )
        knn_model_ag.fit(X_train_ag_scaled) # Fit k-NN on scaled AG training feature vectors
        model_artifacts['knn_model_ag'] = knn_model_ag
        
        # Save the *original indices* of the samples used for training.
        # These indices refer to the full `X_ag` and `recipes_for_knn_retrieval` lists.
        # This allows mapping a neighbor (index within the training subset) back to its original data.
        model_artifacts['train_original_indices_ag'] = train_original_indices.tolist()
    
    # RF parts are omitted as this script focuses on k-NN with AG features.

    end_train_time = time.time()
    print(f"Model training took {end_train_time - start_train_time:.2f} seconds.")

    # --- Save Model Artifacts ---
    print(f"\nSaving trained AG model artifacts to: {MODEL_DIR}")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, artifact in model_artifacts.items():
            joblib.dump(artifact, os.path.join(MODEL_DIR, f'{name}.joblib'))
        
        # Save model type info for reference during evaluation
        with open(os.path.join(MODEL_DIR, 'model_info_ag.json'), 'w') as f:
            json.dump({'model_type': model_type, 'data_type': 'actiongraph'}, f)

    except Exception as e:
        print(f"Error saving AG model artifacts: {e}", file=sys.stderr)
        return False

    print("AG Training step completed successfully.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train k-NN model on ActionGraph features.')
    # For this version, model_type is fixed to 'knn' in practice.
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn'],
                        help='Type of model to train (fixed to knn for AG features)')
    args = parser.parse_args()

    if not run_train_ag(model_type=args.model_type):
        sys.exit(1)
