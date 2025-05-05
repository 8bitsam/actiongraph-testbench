# evaluate.py

import numpy as np
from sklearn.model_selection import train_test_split
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
# Default metric for standalone run optimization goal (though not optimized here)
DEFAULT_METRIC_TO_OPTIMIZE = 'avg_precursors_f1'

TEST_SIZE = 0.25
RANDOM_STATE = 42

# --- Helper Functions ---
def calculate_prf1(pred_set, true_set):
    """Calculates Precision, Recall, and F1 score for two sets."""
    if not isinstance(pred_set, set): pred_set = set(pred_set)
    if not isinstance(true_set, set): true_set = set(true_set)
    true_positives = len(pred_set.intersection(true_set))
    false_positives = len(pred_set.difference(true_set))
    false_negatives = len(true_set.difference(pred_set))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else (1.0 if not true_set else 0.0)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compare_recipes_detailed(pred_recipe, true_recipe):
    """Compares predicted recipe to the true one using detailed metrics (k-NN only)."""
    results = {
        "precursors_exact_match": False, "precursors_formula_jaccard": 0.0,
        "precursors_precision": 0.0, "precursors_recall": 0.0, "precursors_f1": 0.0,
        "operations_exact_match": False, "operations_type_jaccard": 0.0,
        "op_type_precision": 0.0, "op_type_recall": 0.0, "op_type_f1": 0.0,
        "operations_length_match": False
    }
    pred_recipe = pred_recipe if isinstance(pred_recipe, dict) else {}
    true_recipe = true_recipe if isinstance(true_recipe, dict) else {}

    # Precursors
    pred_precursors_list = pred_recipe.get('precursors', [])
    true_precursors_list = true_recipe.get('precursors', [])
    if not isinstance(pred_precursors_list, list): pred_precursors_list = []
    if not isinstance(true_precursors_list, list): true_precursors_list = []
    pred_prec_formulas = set(p['material_formula'] for p in pred_precursors_list if isinstance(p, dict) and p.get('material_formula'))
    true_prec_formulas = set(p['material_formula'] for p in true_precursors_list if isinstance(p, dict) and p.get('material_formula'))
    if pred_prec_formulas and true_prec_formulas and pred_prec_formulas == true_prec_formulas: results["precursors_exact_match"] = True
    intersection_prec = len(pred_prec_formulas.intersection(true_prec_formulas))
    union_prec = len(pred_prec_formulas.union(true_prec_formulas))
    if union_prec > 0: results["precursors_formula_jaccard"] = intersection_prec / union_prec
    prec_p, prec_r, prec_f1 = calculate_prf1(pred_prec_formulas, true_prec_formulas)
    results["precursors_precision"], results["precursors_recall"], results["precursors_f1"] = prec_p, prec_r, prec_f1

    # Operations
    pred_ops = pred_recipe.get('operations', [])
    true_ops = true_recipe.get('operations', [])
    if not isinstance(pred_ops, list): pred_ops = []
    if not isinstance(true_ops, list): true_ops = []
    try: # Exact match
        sorted_pred_ops = sorted(pred_ops, key=lambda x: json.dumps(x, sort_keys=True))
        sorted_true_ops = sorted(true_ops, key=lambda x: json.dumps(x, sort_keys=True))
        if json.dumps(sorted_pred_ops, sort_keys=True) == json.dumps(sorted_true_ops, sort_keys=True):
             results["operations_exact_match"] = True
    except Exception: pass
    pred_op_types = set(op['type'] for op in pred_ops if isinstance(op, dict) and op.get('type'))
    true_op_types = set(op['type'] for op in true_ops if isinstance(op, dict) and op.get('type'))
    intersection_ops = len(pred_op_types.intersection(true_op_types))
    union_ops = len(pred_op_types.union(true_op_types))
    if union_ops > 0: results["operations_type_jaccard"] = intersection_ops / union_ops
    op_p, op_r, op_f1 = calculate_prf1(pred_op_types, true_op_types)
    results["op_type_precision"], results["op_type_recall"], results["op_type_f1"] = op_p, op_r, op_f1
    results["operations_length_match"] = (len(pred_ops) == len(true_ops))

    return results

def run_evaluate(featurized_dir=DEFAULT_FEATURIZED_DATA_DIR, model_dir=DEFAULT_MODEL_DIR, metric_to_optimize=DEFAULT_METRIC_TO_OPTIMIZE, verbose=True):
    """Loads k-NN model, scaler, test data, evaluates predictions, returns key metric."""
    if verbose: print(f"--- Running Evaluation Step (k-NN) ---")
    if verbose: print(f"Reading features from: {featurized_dir}")
    if verbose: print(f"Loading model artifacts from: {model_dir}")

    try:
        X = np.load(os.path.join(featurized_dir, 'features.npy'))
        with open(os.path.join(featurized_dir, 'recipes.json'), 'r') as f: all_recipes = json.load(f)
        with open(os.path.join(featurized_dir, 'targets.json'), 'r') as f: all_targets = json.load(f)
        with open(os.path.join(featurized_dir, 'original_indices.json'), 'r') as f: all_original_indices = json.load(f)
        if not (len(X) == len(all_recipes) == len(all_targets) == len(all_original_indices)):
             raise ValueError("Length mismatch.")
        if verbose: print(f"Loaded {len(X)} total samples.")
    except FileNotFoundError:
        print(f"Error: Featurized data not found in {featurized_dir}.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading featurized data: {e}", file=sys.stderr)
        return None

    model_artifacts = {}
    try:
        model_artifacts['scaler'] = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        model_artifacts['knn_model'] = joblib.load(os.path.join(model_dir, 'knn_model.joblib'))
        model_artifacts['train_indices'] = joblib.load(os.path.join(model_dir, 'train_indices.joblib'))
    except FileNotFoundError:
        print(f"Error: Model artifacts not found in {model_dir}.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error loading model artifacts: {e}", file=sys.stderr)
        return None

    if len(X) < 2: return 0.0 # Cannot split or evaluate
    indices = np.arange(len(X))
    _, X_test, _, test_indices = train_test_split(
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_test_recipes = [all_recipes[i] for i in test_indices]
    y_test_targets = [all_targets[i] for i in test_indices]
    test_original_ids = [all_original_indices[i] for i in test_indices]
    train_indices = model_artifacts['train_indices']
    y_train_recipes = [all_recipes[i] for i in train_indices]

    if len(X_test) == 0: return 0.0 # Return 0 if no test data

    scaler = model_artifacts['scaler']
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    predictions = []
    knn_model = model_artifacts['knn_model']
    distances_all, neighbor_idx_in_train_all = knn_model.kneighbors(X_test_scaled)
    for i in range(len(X_test_scaled)):
        nearest_train_list_index = neighbor_idx_in_train_all[i][0]
        predicted_recipe = y_train_recipes[nearest_train_list_index]
        predictions.append(predicted_recipe)

    evaluation_results = []
    for i in range(len(predictions)):
        eval_metrics = compare_recipes_detailed(predictions[i], y_test_recipes[i])
        evaluation_results.append(eval_metrics)

    avg_results = {}
    if evaluation_results:
        metric_keys = list(evaluation_results[0].keys())
        for key in metric_keys:
            values = [res.get(key) for res in evaluation_results if res.get(key) is not None]
            if values:
                 if isinstance(values[0], bool): avg_results[f"avg_{key}"] = np.mean([float(v) for v in values])
                 elif isinstance(values[0], (int, float)): avg_results[f"avg_{key}"] = np.mean(values)
                 else: avg_results[f"avg_{key}"] = "N/A"
            else: avg_results[f"avg_{key}"] = "N/A"

    if verbose:
        print("\n--- Evaluation Summary ---")
        print(f"Model Type: k-NN")
        print(f"Test Set Size: {len(predictions)}")
        print("\n-- Precursor Metrics --")
        print(f"{'avg_precursors_exact_match':<30}: {avg_results.get('avg_precursors_exact_match', 'N/A'):.4f}")
        print(f"{'avg_precursors_formula_jaccard':<30}: {avg_results.get('avg_precursors_formula_jaccard', 'N/A'):.4f}")
        print(f"{'avg_precursors_precision':<30}: {avg_results.get('avg_precursors_precision', 'N/A'):.4f}")
        print(f"{'avg_precursors_recall':<30}: {avg_results.get('avg_precursors_recall', 'N/A'):.4f}")
        print(f"{'avg_precursors_f1':<30}: {avg_results.get('avg_precursors_f1', 'N/A'):.4f}")
        print("\n-- Operation Metrics --")
        print(f"{'avg_operations_exact_match':<30}: {avg_results.get('avg_operations_exact_match', 'N/A'):.4f}")
        print(f"{'avg_operations_length_match':<30}: {avg_results.get('avg_operations_length_match', 'N/A'):.4f}")
        print(f"{'avg_operations_type_jaccard':<30}: {avg_results.get('avg_operations_type_jaccard', 'N/A'):.4f}")
        print(f"{'avg_op_type_precision':<30}: {avg_results.get('avg_op_type_precision', 'N/A'):.4f}")
        print(f"{'avg_op_type_recall':<30}: {avg_results.get('avg_op_type_recall', 'N/A'):.4f}")
        print(f"{'avg_op_type_f1':<30}: {avg_results.get('avg_op_type_f1', 'N/A'):.4f}")
        print("\nEvaluation step completed successfully.")

    metric_value = avg_results.get(metric_to_optimize)
    if metric_value is None or not isinstance(metric_value, (int, float)):
         print(f"Warning: Metric '{metric_to_optimize}' not found or non-numeric.", file=sys.stderr)
         return None
    return metric_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate k-NN model.')
    # Add arguments to specify dirs if needed for standalone run flexibility
    parser.add_argument('--feature_dir', type=str, default=DEFAULT_FEATURIZED_DATA_DIR, help='Directory containing featurized data')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, help='Directory containing trained model artifacts')
    args = parser.parse_args()

    print(f"Running evaluate.py standalone...")
    final_metric = run_evaluate(
        featurized_dir=args.feature_dir,
        model_dir=args.model_dir,
        metric_to_optimize=DEFAULT_METRIC_TO_OPTIMIZE, # Use default metric for standalone print
        verbose=True
    )
    if final_metric is None:
        sys.exit(1)
    print(f"\nStandalone run finished. Metric ({DEFAULT_METRIC_TO_OPTIMIZE}): {final_metric:.4f}")
