# 4_evaluate_knn_generic.py
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import json
import argparse
from common_utils import calculate_set_prf1 # Use common P/R/F1

# --- Configuration ---
TEST_SIZE = 0.25
RANDOM_STATE = 42

def compare_recipes_detailed_knn(pred_recipe, true_recipe):
    """Compares predicted recipe to the true one (k-NN specific)."""
    # (This function can be identical to the one in your previous evaluate.py for k-NN)
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
    pred_prec_formulas = set(p['material_formula'] for p in pred_precursors_list if isinstance(p, dict) and p.get('material_formula'))
    true_prec_formulas = set(p['material_formula'] for p in true_precursors_list if isinstance(p, dict) and p.get('material_formula'))
    if pred_prec_formulas and true_prec_formulas and pred_prec_formulas == true_prec_formulas: results["precursors_exact_match"] = True
    intersection_prec = len(pred_prec_formulas.intersection(true_prec_formulas))
    union_prec = len(pred_prec_formulas.union(true_prec_formulas))
    if union_prec > 0: results["precursors_formula_jaccard"] = intersection_prec / union_prec
    prec_p, prec_r, prec_f1 = calculate_set_prf1(pred_prec_formulas, true_prec_formulas)
    results["precursors_precision"], results["precursors_recall"], results["precursors_f1"] = prec_p, prec_r, prec_f1
    # Operations
    pred_ops = pred_recipe.get('operations', [])
    true_ops = true_recipe.get('operations', [])
    try:
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
    op_p, op_r, op_f1 = calculate_set_prf1(pred_op_types, true_op_types)
    results["op_type_precision"], results["op_type_recall"], results["op_type_f1"] = op_p, op_r, op_f1
    results["operations_length_match"] = (len(pred_ops) == len(true_ops))
    return results

def run_evaluate_knn(featurized_input_dir, model_input_dir, verbose=True):
    if verbose: print(f"--- Running Generic k-NN Evaluation ---")
    if verbose: print(f"Reading features from: {featurized_input_dir}")
    if verbose: print(f"Loading model artifacts from: {model_input_dir}")

    try:
        X = np.load(os.path.join(featurized_input_dir, 'features.npy'))
        with open(os.path.join(featurized_input_dir, 'recipes.json'), 'r') as f: all_recipes = json.load(f)
        with open(os.path.join(featurized_input_dir, 'targets.json'), 'r') as f: all_targets = json.load(f)
        with open(os.path.join(featurized_input_dir, 'original_ids.json'), 'r') as f: all_original_ids = json.load(f)
        if not (len(X) == len(all_recipes) == len(all_targets) == len(all_original_ids)):
             raise ValueError("Length mismatch in featurized data.")
        if verbose: print(f"Loaded {len(X)} total samples for evaluation context.")
    except Exception as e: print(f"Error loading featurized data: {e}", file=sys.stderr); return None

    model_artifacts = {}
    try:
        model_artifacts['scaler'] = joblib.load(os.path.join(model_input_dir, 'scaler.joblib'))
        model_artifacts['knn_model'] = joblib.load(os.path.join(model_input_dir, 'knn_model.joblib'))
        model_artifacts['train_indices'] = joblib.load(os.path.join(model_input_dir, 'train_indices.joblib'))
        with open(os.path.join(model_input_dir, 'model_info.json'), 'r') as f: model_info = json.load(f)
        if verbose: print(f"Model info: {model_info}")
    except Exception as e: print(f"Error loading model artifacts: {e}", file=sys.stderr); return None

    if len(X) < 2: print("Not enough data for split."); return {}
    indices = np.arange(len(X))
    _, X_test, _, test_indices_in_full_set = train_test_split( # test_indices are for the full loaded X
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_test_recipes = [all_recipes[i] for i in test_indices_in_full_set]
    y_test_targets = [all_targets[i] for i in test_indices_in_full_set]
    # test_original_ids = [all_original_ids[i] for i in test_indices_in_full_set]

    # train_indices from model artifact are indices into the *original full set*
    train_indices_from_model = model_artifacts['train_indices']
    y_train_recipes_for_lookup = [all_recipes[i] for i in train_indices_from_model]

    if len(X_test) == 0: print("No test data after split."); return {}

    scaler = model_artifacts['scaler']
    X_test_scaled = scaler.transform(X_test)
    # X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0) # if using StandardScaler

    predictions = []
    knn_model = model_artifacts['knn_model']
    distances_all, neighbor_idx_in_training_split = knn_model.kneighbors(X_test_scaled)

    for i in range(len(X_test_scaled)):
        # neighbor_idx_in_training_split is an index into the X_train_scaled that knn was fit on.
        # We need to map this back to an index in y_train_recipes_for_lookup.
        # The y_train_recipes_for_lookup is already sliced based on train_indices_from_model.
        predicted_recipe = y_train_recipes_for_lookup[neighbor_idx_in_training_split[i][0]]
        predictions.append(predicted_recipe)

    evaluation_results_list = []
    for i in range(len(predictions)):
        eval_metrics = compare_recipes_detailed_knn(predictions[i], y_test_recipes[i])
        evaluation_results_list.append(eval_metrics)

    avg_results = {}
    if evaluation_results_list:
        metric_keys = list(evaluation_results_list[0].keys())
        for key in metric_keys:
            values = [res.get(key) for res in evaluation_results_list if res.get(key) is not None]
            if values:
                 avg_results[f"avg_{key}"] = np.mean([float(v) if isinstance(v, bool) else v for v in values])
            else: avg_results[f"avg_{key}"] = float('nan') # Or "N/A"

    if verbose:
        print("\n--- Evaluation Summary ---")
        print(f"Model: {model_info.get('model_type', 'k-NN')} from {model_info.get('featurized_input_dir', 'Unknown Source')}")
        print(f"Test Set Size: {len(predictions)}")
        # ... (print metrics as before) ...
        print("\n-- Precursor Metrics --")
        print(f"{'avg_precursors_exact_match':<30}: {avg_results.get('avg_precursors_exact_match', float('nan')):.4f}")
        # ... and so on for all metrics in avg_results ...
        print(f"{'avg_precursors_f1':<30}: {avg_results.get('avg_precursors_f1', float('nan')):.4f}")
        print("\n-- Operation Metrics --")
        print(f"{'avg_op_type_f1':<30}: {avg_results.get('avg_op_type_f1', float('nan')):.4f}")
        print("\nGeneric k-NN Evaluation step completed.")
    return avg_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic k-NN Evaluator.')
    parser.add_argument('--feature_dir', type=str, required=True, help='Directory containing featurized data used for this model')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained k-NN model artifacts')
    args = parser.parse_args()

    results = run_evaluate_knn(featurized_input_dir=args.feature_dir, model_input_dir=args.model_dir, verbose=True)
    if results:
        print("\nStandalone Run Avg Results:")
        for k, v in results.items(): print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        sys.exit(1)
