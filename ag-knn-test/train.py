import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

DATA_DIR = "../Data/"
FEATURIZED_DATA_DIR_AG = os.path.join(DATA_DIR, "featurized-data-actiongraph/")
MODEL_AG_BASE_DIR = os.path.join(DATA_DIR, "models_ag")

TEST_SIZE = 0.25
RANDOM_STATE = 42
N_NEIGHBORS = 1


def run_train_ag(model_type="knn"):
    if model_type != "knn":
        print(
            f"Error: This script is configured for k-NN with ActionGraph \
                features. Model type '{model_type}' not supported here.",
            file=sys.stderr,
        )
        return False

    print(
        f"--- Running Training Step (Model: {model_type.upper()} with \
            ActionGraph Features) ---"
    )

    MODEL_DIR = os.path.join(MODEL_AG_BASE_DIR, f"{model_type}-model")
    print(f"Model artifacts will be saved to: {MODEL_DIR}")

    print(f"\nLoading AG-featurized data from: {FEATURIZED_DATA_DIR_AG}")
    try:
        X_ag = np.load(os.path.join(FEATURIZED_DATA_DIR_AG, "features_ag.npy"))
        with open(
            os.path.join(FEATURIZED_DATA_DIR_AG, "recipes_mp_style_for_ag.json"), "r"
        ) as f:
            recipes_for_knn_retrieval = json.load(f)

        if len(X_ag) != len(recipes_for_knn_retrieval):
            raise ValueError(
                "Mismatch between AG features and associated MP-style \
                    recipes length."
            )
        print(f"Loaded {len(X_ag)} AG-featurized samples.")
        print(f"AG Feature array shape: {X_ag.shape}")
    except FileNotFoundError:
        print(
            f"Error: Featurized AG data not found in \
                {FEATURIZED_DATA_DIR_AG}.",
            file=sys.stderr,
        )
        print("Please run the 'featurize_ag' step first.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error loading featurized AG data: {e}", file=sys.stderr)
        return False

    # Train test split
    print(f"\nPerforming train/test split (Test size: {TEST_SIZE})...")
    if len(X_ag) < 2:
        print("Error: Not enough data points (< 2) for split.", file=sys.stderr)
        return False

    # Create an array of indices [0, 1, ..., N-1]
    original_indices_all = np.arange(len(X_ag))

    X_train_ag, _, train_original_indices, _ = train_test_split(
        X_ag, original_indices_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"Training set size for AG features: {len(X_train_ag)}")
    if len(X_train_ag) == 0:
        print(
            "Error: Training set (AG features) has size 0 after split.", file=sys.stderr
        )
        return False

    print("\nFitting scaler on AG training data...")
    scaler_ag = QuantileTransformer(
        output_distribution="normal",
        random_state=RANDOM_STATE,
        n_quantiles=max(1, min(1000, len(X_train_ag))),
    )
    start_scale_time = time.time()
    try:
        scaler_ag.fit(X_train_ag)
    except ValueError as e:
        print(
            f"Error fitting scaler: {e}. Ensure n_quantiles <= n_samples for \
                QuantileTransformer.",
            file=sys.stderr,
        )
        return False

    X_train_ag_scaled = scaler_ag.transform(X_train_ag)
    X_train_ag_scaled = np.nan_to_num(
        X_train_ag_scaled, nan=0.0, posinf=0.0, neginf=0.0
    )
    end_scale_time = time.time()
    print(f"Scaling took {end_scale_time - start_scale_time:.2f} seconds.")

    # Training
    start_train_time = time.time()
    model_artifacts = {"scaler_ag": scaler_ag}

    if model_type == "knn":
        knn_model_ag = NearestNeighbors(
            n_neighbors=N_NEIGHBORS,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        knn_model_ag.fit(X_train_ag_scaled)
        model_artifacts["knn_model_ag"] = knn_model_ag
        model_artifacts["train_original_indices_ag"] = train_original_indices.tolist()

    end_train_time = time.time()
    print(
        f"Model training took {end_train_time - start_train_time:.2f} \
          seconds."
    )
    print(f"\nSaving trained AG model artifacts to: {MODEL_DIR}")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, artifact in model_artifacts.items():
            joblib.dump(artifact, os.path.join(MODEL_DIR, f"{name}.joblib"))
        with open(os.path.join(MODEL_DIR, "model_info_ag.json"), "w") as f:
            json.dump({"model_type": model_type, "data_type": "actiongraph"}, f)
    except Exception as e:
        print(f"Error saving AG model artifacts: {e}", file=sys.stderr)
        return False
    print("AG Training step completed successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train k-NN model on ActionGraph features."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="knn",
        choices=["knn"],
        help="Type of model to train (fixed to knn for AG features)",
    )
    args = parser.parse_args()
    if not run_train_ag(model_type=args.model_type):
        sys.exit(1)
