# train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer
import joblib
import os
import time
import sys
import json
import argparse

# --- Configuration ---
DATA_DIR = "../Data/"
FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-baseline/") # Common features
MODEL_BASE_DIR = os.path.join(DATA_DIR, "models") # Base directory for models

TEST_SIZE = 0.25 # 75% train, 25% test
RANDOM_STATE = 42
# k-NN specific
N_NEIGHBORS = 1
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = None # Default
RF_MIN_SAMPLES_SPLIT = 5 # Adjusted from default 2
RF_N_JOBS = -1 # Use all available CPU cores

def run_train(model_type='knn'):
    """Loads featurized data, trains specified model, saves artifacts."""
    print(f"--- Running Training Step (Model: {model_type.upper()}) ---")

    # --- Define model-specific directory ---
    MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}-model")
    print(f"Model artifacts will be saved to: {MODEL_DIR}")

    # --- Load Featurized Data ---
    print(f"\nLoading featurized data from: {FEATURIZED_DATA_DIR}")
    try:
        X = np.load(os.path.join(FEATURIZED_DATA_DIR, 'features.npy'))
        with open(os.path.join(FEATURIZED_DATA_DIR, 'recipes.json'), 'r') as f:
            recipes = json.load(f)
        if len(X) != len(recipes):
             raise ValueError("Mismatch between features and recipes length.")
        print(f"Loaded {len(X)} samples.")
        print(f"Feature array shape: {X.shape}")
    except FileNotFoundError:
        print(f"Error: Featurized data not found in {FEATURIZED_DATA_DIR}.", file=sys.stderr)
        print("Please run the 'featurize' step first.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error loading featurized data: {e}", file=sys.stderr)
        return False

    # --- Train/Test Split ---
    print(f"\nPerforming train/test split (Test size: {TEST_SIZE})...")
    if len(X) < 2:
        print("Error: Not enough data points (< 2) for split.", file=sys.stderr)
        return False

    indices = np.arange(len(X))
    X_train, X_test, train_indices, test_indices = train_test_split(
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # Get recipes corresponding to the training set
    y_train_recipes = [recipes[i] for i in train_indices]

    print(f"Training set size: {len(X_train)}")
    if len(X_train) == 0:
        print("Error: Training set has size 0 after split.", file=sys.stderr)
        return False

    # --- Scale Features (Common Step) ---
    print("\nFitting scaler on training data...")
    # Note: RF is less sensitive, but scaling doesn't hurt and QuantileTransformer
    # mapping to normal might help some tree algorithms slightly. Keeps consistency.
    scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE, n_quantiles=min(1000, len(X_train)))
    start_scale_time = time.time()
    try:
        scaler.fit(X_train)
    except ValueError as e:
        print(f"Error fitting scaler: {e}.", file=sys.stderr)
        return False
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    end_scale_time = time.time()
    print(f"Scaling took {end_scale_time - start_scale_time:.2f} seconds.")

    # --- Model Specific Training ---
    start_train_time = time.time()
    model_artifacts = {'scaler': scaler} # Scaler is always saved

    if model_type == 'knn':
        print(f"\nTraining k-NN model (k={N_NEIGHBORS}, metric='cosine')...")
        knn_model = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric='cosine', algorithm='auto')
        knn_model.fit(X_train_scaled) # Fit k-NN on scaled training data
        model_artifacts['knn_model'] = knn_model
        # Note: k-NN needs the training recipes during evaluation to retrieve the neighbor's recipe.
        # We save the training recipes *indices* here, evaluation will load all recipes and use indices.
        model_artifacts['train_indices'] = train_indices.tolist() # Save indices of training samples


    elif model_type == 'rf':
        print("\nPreparing targets for Random Forest (Multi-Label Classification)...")
        # 1. Create Multi-Label Targets for Precursors
        prec_binarizer = MultiLabelBinarizer()
        train_precursors = [
            tuple(sorted([p['material_formula'] for p in r['precursors'] if p.get('material_formula')]))
            for r in y_train_recipes
        ]
        try:
             y_train_prec_multi = prec_binarizer.fit_transform(train_precursors)
             print(f"  Precursor Vocabulary Size: {len(prec_binarizer.classes_)}")
             if len(prec_binarizer.classes_) == 0:
                  print("Warning: No precursors found in training data.", file=sys.stderr)
        except Exception as e:
             print(f"Error creating precursor multi-labels: {e}", file=sys.stderr)
             return False

        # 2. Create Multi-Label Targets for Operation Types
        op_type_binarizer = MultiLabelBinarizer()
        train_op_types = [
            tuple(sorted([op['type'] for op in r['operations'] if op.get('type')]))
             for r in y_train_recipes
        ]
        try:
             y_train_op_type_multi = op_type_binarizer.fit_transform(train_op_types)
             print(f"  Operation Type Vocabulary Size: {len(op_type_binarizer.classes_)}")
             if len(op_type_binarizer.classes_) == 0:
                  print("Warning: No operation types found in training data.", file=sys.stderr)
        except Exception as e:
             print(f"Error creating operation type multi-labels: {e}", file=sys.stderr)
             return False

        # 3. Train RF models (using MultiOutputClassifier for multi-label tasks)
        print(f"\nTraining Random Forest models (n_estimators={RF_N_ESTIMATORS}, n_jobs={RF_N_JOBS})...")
        # Base RF model
        base_rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            random_state=RANDOM_STATE,
            n_jobs=RF_N_JOBS,
            class_weight='balanced' # Often helpful for multi-label
        )

        # RF for Precursors
        print("  Training precursor predictor...")
        rf_prec_model = MultiOutputClassifier(base_rf, n_jobs=RF_N_JOBS)
        if y_train_prec_multi.shape[1] > 0: # Only train if there are precursors
            rf_prec_model.fit(X_train_scaled, y_train_prec_multi)
            model_artifacts['rf_prec_model'] = rf_prec_model
            model_artifacts['prec_binarizer'] = prec_binarizer
        else:
            print("  Skipping precursor model training (no precursor classes).")


        # RF for Operation Types
        print("  Training operation type predictor...")
        rf_op_type_model = MultiOutputClassifier(base_rf, n_jobs=RF_N_JOBS)
        if y_train_op_type_multi.shape[1] > 0: # Only train if there are op types
            rf_op_type_model.fit(X_train_scaled, y_train_op_type_multi)
            model_artifacts['rf_op_type_model'] = rf_op_type_model
            model_artifacts['op_type_binarizer'] = op_type_binarizer
        else:
             print("  Skipping operation type model training (no op type classes).")

    else:
        print(f"Error: Unknown model_type '{model_type}'", file=sys.stderr)
        return False

    end_train_time = time.time()
    print(f"Model training took {end_train_time - start_train_time:.2f} seconds.")

    # --- Save Model Artifacts ---
    print(f"\nSaving trained artifacts to: {MODEL_DIR}")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        for name, artifact in model_artifacts.items():
            joblib.dump(artifact, os.path.join(MODEL_DIR, f'{name}.joblib'))
        # Save model type info for evaluation reference
        with open(os.path.join(MODEL_DIR, 'model_info.json'), 'w') as f:
            json.dump({'model_type': model_type}, f)

    except Exception as e:
        print(f"Error saving model artifacts: {e}", file=sys.stderr)
        return False

    print("Training step completed successfully.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train k-NN or RF model.')
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'rf'],
                        help='Type of model to train (knn or rf)')
    args = parser.parse_args()

    if not run_train(model_type=args.model_type):
        sys.exit(1)
