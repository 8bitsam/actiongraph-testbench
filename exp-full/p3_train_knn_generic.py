# 3_train_knn_generic.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler # Changed from Quantile for Euclidean
import joblib
import os
import time
import sys
import json
import argparse

# --- Configuration ---
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_NEIGHBORS = 1

def run_train_knn(featurized_input_dir, model_output_dir, metric='euclidean'):
    print(f"--- Running Generic k-NN Training ---")
    print(f"Reading features from: {featurized_input_dir}")
    print(f"Saving model artifacts to: {model_output_dir}")
    print(f"Using k={N_NEIGHBORS}, distance_metric='{metric}'")

    try:
        X = np.load(os.path.join(featurized_input_dir, 'features.npy'))
        with open(os.path.join(featurized_input_dir, 'recipes.json'), 'r') as f:
            recipes = json.load(f) # Needed for length check and evaluation mapping
        if len(X) != len(recipes): raise ValueError("Length mismatch.")
        print(f"Loaded {len(X)} samples. Feature shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data from {featurized_input_dir}: {e}", file=sys.stderr)
        return False

    if len(X) < 2: print("Error: Not enough data for split.", file=sys.stderr); return False
    indices = np.arange(len(X))
    X_train, _, train_indices, _ = train_test_split(
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if len(X_train) == 0: print("Error: Training set size is 0.", file=sys.stderr); return False
    print(f"Training set size: {len(X_train)}")

    print("Fitting scaler (StandardScaler)...")
    scaler = StandardScaler()
    try:
        scaler.fit(X_train)
    except Exception as e:
        print(f"Error fitting scaler: {e}", file=sys.stderr); return False
    X_train_scaled = scaler.transform(X_train)
    # No need for nan_to_num if input features are clean and StandardScaler is used

    print("Training k-NN model...")
    start_train_time = time.time()
    model_artifacts = {'scaler': scaler}
    knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric=metric, algorithm='auto')
    knn_model.fit(X_train_scaled)
    model_artifacts['knn_model'] = knn_model
    model_artifacts['train_indices'] = train_indices.tolist() # Important for eval
    end_train_time = time.time()
    print(f"k-NN training took {end_train_time - start_train_time:.2f} seconds.")

    print(f"Saving artifacts...")
    try:
        os.makedirs(model_output_dir, exist_ok=True)
        for name, artifact in model_artifacts.items():
            joblib.dump(artifact, os.path.join(model_output_dir, f'{name}.joblib'))
        # Save info about the model trained
        with open(os.path.join(model_output_dir, 'model_info.json'), 'w') as f:
            json.dump({'model_type': 'knn', 'k': N_NEIGHBORS, 'metric': metric,
                       'featurized_input_dir': os.path.basename(featurized_input_dir)}, f, indent=2)
    except Exception as e:
        print(f"Error saving artifacts to {model_output_dir}: {e}", file=sys.stderr)
        return False

    print("Generic k-NN Training step completed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic k-NN Trainer.')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing featurized data (features.npy, recipes.json, etc.)')
    parser.add_argument('--model_out_dir', type=str, required=True, help='Directory to save trained model artifacts')
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine', 'manhattan'], help='Distance metric for k-NN')
    args = parser.parse_args()

    if not run_train_knn(featurized_input_dir=args.feature_dir, model_output_dir=args.model_out_dir, metric=args.metric):
        sys.exit(1)
