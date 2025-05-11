# train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
import sys
import json
import argparse

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
# Default input/output dirs if run standalone
DEFAULT_FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-weighted/")
DEFAULT_MODEL_DIR = os.path.join(DATA_DIR, "models/knn-weighted-model/")

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_NEIGHBORS = 1 # k-NN Specific

def run_train(featurized_dir=DEFAULT_FEATURIZED_DATA_DIR, model_out_dir=DEFAULT_MODEL_DIR):
    """Loads featurized data, trains k-NN, saves artifacts."""
    print(f"--- Running Training Step (k-NN) ---")
    print(f"Reading features from: {featurized_dir}")
    print(f"Saving model artifacts to: {model_out_dir}")

    try:
        X = np.load(os.path.join(featurized_dir, 'features.npy'))
        with open(os.path.join(featurized_dir, 'recipes.json'), 'r') as f:
            recipes = json.load(f) # Needed only for length check
        if len(X) != len(recipes): raise ValueError("Length mismatch.")
        print(f"Loaded {len(X)} samples. Feature shape: {X.shape}")
    except Exception as e:
        print(f"Error loading data from {featurized_dir}: {e}", file=sys.stderr)
        return False

    if len(X) < 2:
        print("Error: Not enough data for split.", file=sys.stderr)
        return False
    indices = np.arange(len(X))
    X_train, _, train_indices, _ = train_test_split(
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if len(X_train) == 0:
        print("Error: Training set size is 0.", file=sys.stderr)
        return False
    print(f"Training set size: {len(X_train)}")

    print("Fitting scaler...")
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE, n_quantiles=min(1000, len(X_train)))
    # scaler = StandardScaler()
    try:
        scaler.fit(X_train)
    except Exception as e:
        print(f"Error fitting scaler: {e}", file=sys.stderr)
        return False
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    print("Training k-NN model...")
    start_train_time = time.time()
    model_artifacts = {'scaler': scaler}
    knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='auto')
    knn_model.fit(X_train_scaled)
    model_artifacts['knn_model'] = knn_model
    model_artifacts['train_indices'] = train_indices.tolist()
    end_train_time = time.time()
    print(f"k-NN training took {end_train_time - start_train_time:.2f} seconds.")

    print(f"Saving artifacts...")
    try:
        os.makedirs(model_out_dir, exist_ok=True)
        for name, artifact in model_artifacts.items():
            joblib.dump(artifact, os.path.join(model_out_dir, f'{name}.joblib'))
        with open(os.path.join(model_out_dir, 'model_info.json'), 'w') as f:
            json.dump({'model_type': 'knn'}, f)
    except Exception as e:
        print(f"Error saving artifacts to {model_out_dir}: {e}", file=sys.stderr)
        return False

    print("Training step completed.")
    return True

if __name__ == "__main__":
    print("Running train.py standalone with default paths...")
    if not run_train():
        sys.exit(1)
