import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-baseline/")
MODEL_BASE_DIR = os.path.join(DATA_DIR, "models")

# Must match train.py
TEST_SIZE = 0.25
RANDOM_STATE = 42


def calculate_prf1(pred_set, true_set):
    """Calculates Precision, Recall, and F1 score for two sets."""
    if not isinstance(pred_set, set):
        pred_set = set(pred_set)
    if not isinstance(true_set, set):
        true_set = set(true_set)
    true_positives = len(pred_set.intersection(true_set))
    false_positives = len(pred_set.difference(true_set))
    false_negatives = len(true_set.difference(pred_set))
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else (1.0 if not true_set else 0.0)
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compare_recipes_detailed(pred_recipe, true_recipe, is_rf=False):
    """Compares predicted recipe to the true one using detailed metrics."""
    results = {
        "precursors_exact_match": False,
        "precursors_formula_jaccard": 0.0,
        "precursors_precision": 0.0,
        "precursors_recall": 0.0,
        "precursors_f1": 0.0,
        "operations_exact_match": False,
        "operations_type_jaccard": 0.0,
        "op_type_precision": 0.0,
        "op_type_recall": 0.0,
        "op_type_f1": 0.0,
        "operations_length_match": False,
    }
    pred_recipe = pred_recipe if isinstance(pred_recipe, dict) else {}
    true_recipe = true_recipe if isinstance(true_recipe, dict) else {}

    # Precursors Comparison
    pred_precursors_list = pred_recipe.get("precursors", [])
    true_precursors_list = true_recipe.get("precursors", [])
    if not isinstance(pred_precursors_list, list):
        pred_precursors_list = []
    if not isinstance(true_precursors_list, list):
        true_precursors_list = []
    pred_prec_formulas = set(
        p["material_formula"]
        for p in pred_precursors_list
        if isinstance(p, dict) and p.get("material_formula")
    )
    true_prec_formulas = set(
        p["material_formula"]
        for p in true_precursors_list
        if isinstance(p, dict) and p.get("material_formula")
    )
    if (
        pred_prec_formulas
        and true_prec_formulas
        and pred_prec_formulas == true_prec_formulas
    ):
        results["precursors_exact_match"] = True
    intersection_prec = len(pred_prec_formulas.intersection(true_prec_formulas))
    union_prec = len(pred_prec_formulas.union(true_prec_formulas))
    if union_prec > 0:
        results["precursors_formula_jaccard"] = intersection_prec / union_prec
    prec_p, prec_r, prec_f1 = calculate_prf1(pred_prec_formulas, true_prec_formulas)
    (
        results["precursors_precision"],
        results["precursors_recall"],
        results["precursors_f1"],
    ) = (prec_p, prec_r, prec_f1)

    # Operations Comparison
    pred_ops = pred_recipe.get("operations", [])
    true_ops = true_recipe.get("operations", [])
    if not isinstance(pred_ops, list):
        pred_ops = []
    if not isinstance(true_ops, list):
        true_ops = []
    # Exact Match (only meaningful for k-NN type retrieval)
    if not is_rf:
        try:
            sorted_pred_ops = sorted(
                pred_ops, key=lambda x: json.dumps(x, sort_keys=True)
            )
            sorted_true_ops = sorted(
                true_ops, key=lambda x: json.dumps(x, sort_keys=True)
            )
            if json.dumps(sorted_pred_ops, sort_keys=True) == json.dumps(
                sorted_true_ops, sort_keys=True
            ):
                results["operations_exact_match"] = True
        except Exception:
            pass
    # Get Operation Types sets
    pred_op_types = set(
        op["type"] for op in pred_ops if isinstance(op, dict) and op.get("type")
    )
    true_op_types = set(
        op["type"] for op in true_ops if isinstance(op, dict) and op.get("type")
    )
    intersection_ops = len(pred_op_types.intersection(true_op_types))
    union_ops = len(pred_op_types.union(true_op_types))
    if union_ops > 0:
        results["operations_type_jaccard"] = intersection_ops / union_ops
    op_p, op_r, op_f1 = calculate_prf1(pred_op_types, true_op_types)
    results["op_type_precision"], results["op_type_recall"],
    results["op_type_f1"] = (
        op_p,
        op_r,
        op_f1,
    )
    # Length Match (less meaningful for RF predicting only types)
    results["operations_length_match"] = (
        (len(pred_ops) == len(true_ops)) if not is_rf else False
    )

    return results


def run_evaluate(model_type="knn"):
    """Loads specified model, scaler, test data, evaluates predictions."""
    print(f"--- Running Evaluation Step (Model: {model_type.upper()}) ---")
    MODEL_DIR = os.path.join(MODEL_BASE_DIR, f"{model_type}-model")
    print(f"Loading model artifacts from: {MODEL_DIR}")
    print(f"\nLoading featurized data from: {FEATURIZED_DATA_DIR}")
    try:
        X = np.load(os.path.join(FEATURIZED_DATA_DIR, "features.npy"))
        with open(os.path.join(FEATURIZED_DATA_DIR, "recipes.json"), "r") as f:
            all_recipes = json.load(f)
        with open(os.path.join(FEATURIZED_DATA_DIR, "targets.json"), "r") as f:
            all_targets = json.load(f)
        with open(os.path.join(FEATURIZED_DATA_DIR, "original_indices.json"), "r") as f:
            all_original_indices = json.load(f)
        if not (
            len(X) == len(all_recipes) == len(all_targets) == len(all_original_indices)
        ):
            raise ValueError(
                "Mismatch between loaded featurized \
                             data lengths."
            )
        print(f"Loaded {len(X)} total samples.")
    except FileNotFoundError:
        print(
            f"Error: Featurized data not found in \
                {FEATURIZED_DATA_DIR}. Run 'featurize'.",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"Error loading featurized data: {e}", file=sys.stderr)
        return False

    model_artifacts = {}
    try:
        model_artifacts["scaler"] = joblib.load(
            os.path.join(MODEL_DIR, "scaler.joblib")
        )
        if model_type == "knn":
            model_artifacts["knn_model"] = joblib.load(
                os.path.join(MODEL_DIR, "knn_model.joblib")
            )
            model_artifacts["train_indices"] = joblib.load(
                os.path.join(MODEL_DIR, "train_indices.joblib")
            )
        elif model_type == "rf":
            if os.path.exists(os.path.join(MODEL_DIR, "rf_prec_model.joblib")):
                model_artifacts["rf_prec_model"] = joblib.load(
                    os.path.join(MODEL_DIR, "rf_prec_model.joblib")
                )
                model_artifacts["prec_binarizer"] = joblib.load(
                    os.path.join(MODEL_DIR, "prec_binarizer.joblib")
                )
            if os.path.exists(os.path.join(MODEL_DIR, "rf_op_type_model.joblib")):
                model_artifacts["rf_op_type_model"] = joblib.load(
                    os.path.join(MODEL_DIR, "rf_op_type_model.joblib")
                )
                model_artifacts["op_type_binarizer"] = joblib.load(
                    os.path.join(MODEL_DIR, "op_type_binarizer.joblib")
                )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}' during artifact loading."
            )

    except FileNotFoundError:
        print(
            f"Error: Model artifacts not found in {MODEL_DIR}. \
                Run 'train --model_type {model_type}'.",
            file=sys.stderr,
        )
        return False
    except Exception as e:
        print(f"Error loading model artifacts: {e}", file=sys.stderr)
        return False
    print(
        f"\nRecreating train/test split (Test size: {TEST_SIZE}, \
            Random State: {RANDOM_STATE})..."
    )
    if len(X) < 2:
        print("Error: Not enough data points (< 2) for split.", file=sys.stderr)
        return False

    indices = np.arange(len(X))
    _, X_test, _, test_orig_indices_all = train_test_split(
        X, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_test_recipes = [all_recipes[i] for i in test_orig_indices_all]
    y_test_targets = [all_targets[i] for i in test_orig_indices_all]
    test_original_ids = [all_original_indices[i] for i in test_orig_indices_all]

    print(f"Test set size: {len(X_test)}")
    if len(X_test) == 0:
        print(
            "Warning: Test set has size 0 after split. \
                Evaluation cannot proceed.",
            file=sys.stderr,
        )
        return True

    print("\nScaling test features using loaded scaler...")
    start_scale_time = time.time()
    scaler = model_artifacts["scaler"]
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    end_scale_time = time.time()
    print(f"Scaling took {end_scale_time - start_scale_time:.2f} seconds.")
    print(
        f"\nPredicting recipes for the test set using \
          {model_type.upper()} model..."
    )
    predictions = []
    start_predict_time = time.time()

    if model_type == "knn":
        knn_model = model_artifacts["knn_model"]
        train_indices = model_artifacts["train_indices"]
        y_train_recipes = [all_recipes[i] for i in train_indices]

        distances_all, neighbor_idx_in_train_all = knn_model.kneighbors(X_test_scaled)

        for i in range(len(X_test_scaled)):
            nearest_train_list_index = neighbor_idx_in_train_all[i][0]
            predicted_recipe = y_train_recipes[nearest_train_list_index]
            predictions.append(predicted_recipe)

    elif model_type == "rf":
        pred_prec_multi = None
        pred_op_type_multi = None

        # Predict precursors if model exists
        if "rf_prec_model" in model_artifacts:
            print("  Predicting precursors...")
            rf_prec_model = model_artifacts["rf_prec_model"]
            pred_prec_multi = rf_prec_model.predict(X_test_scaled)
        else:
            print("  Skipping precursor prediction (model not found).")

        # Predict operation types if model exists
        if "rf_op_type_model" in model_artifacts:
            print("  Predicting operation types...")
            rf_op_type_model = model_artifacts["rf_op_type_model"]
            pred_op_type_multi = rf_op_type_model.predict(X_test_scaled)
        else:
            print("  Skipping operation type prediction (model not found).")

        # Convert multi-label predictions back to sets of strings
        prec_binarizer = model_artifacts.get("prec_binarizer")
        op_type_binarizer = model_artifacts.get("op_type_binarizer")

        for i in range(len(X_test_scaled)):
            pred_prec_set = set()
            if pred_prec_multi is not None and prec_binarizer:
                if i < len(pred_prec_multi):
                    try:
                        pred_prec_tuple = prec_binarizer.inverse_transform(
                            pred_prec_multi[i].reshape(1, -1)
                        )
                        if pred_prec_tuple:
                            pred_prec_set = set(pred_prec_tuple[0])
                    except ValueError as e:
                        print(
                            f"Warning: Error transforming precursor \
                                prediction for sample {i}: {e}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"Warning: Missing precursor prediction for \
                            sample {i}.",
                        file=sys.stderr,
                    )

            pred_op_type_set = set()
            if pred_op_type_multi is not None and op_type_binarizer:
                if i < len(pred_op_type_multi):
                    try:
                        pred_op_type_tuple = op_type_binarizer.inverse_transform(
                            pred_op_type_multi[i].reshape(1, -1)
                        )
                        if pred_op_type_tuple:
                            pred_op_type_set = set(pred_op_type_tuple[0])
                    except ValueError as e:
                        print(
                            f"Warning: Error transforming op_type \
                                prediction for sample {i}: {e}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"Warning: Missing op_type prediction \
                            for sample {i}.",
                        file=sys.stderr,
                    )
            predictions.append(
                {
                    "precursors": [{"material_formula": p} for p in pred_prec_set],
                    "operations": [{"type": op} for op in pred_op_type_set],
                }
            )
    else:
        print(
            f"Error: Unknown model_type '{model_type}' for prediction.", file=sys.stderr
        )
        return False
    end_predict_time = time.time()
    print(
        f"Prediction took {end_predict_time - start_predict_time:.2f} \
          seconds."
    )
    print("\nEvaluating predictions with detailed metrics...")
    start_eval_time = time.time()
    evaluation_results = []
    is_rf = model_type == "rf"
    for i in range(len(predictions)):
        pred_recipe = predictions[i]
        true_recipe = y_test_recipes[i]
        eval_metrics = compare_recipes_detailed(pred_recipe, true_recipe, is_rf=is_rf)
        evaluation_results.append(eval_metrics)
    end_eval_time = time.time()
    print(
        f"Evaluation calculation took \
          {end_eval_time - start_eval_time:.2f} seconds."
    )
    avg_results = {}
    if evaluation_results:
        metric_keys = list(evaluation_results[0].keys())
        for key in metric_keys:
            values = [res.get(key, None) for res in evaluation_results]
            valid_values = [v for v in values if v is not None]
            if valid_values:
                if isinstance(valid_values[0], bool):
                    avg_results[f"avg_{key}"] = np.mean(
                        [float(v) for v in valid_values]
                    )
                elif isinstance(valid_values[0], (int, float)):
                    avg_results[f"avg_{key}"] = np.mean(valid_values)
                else:
                    avg_results[f"avg_{key}"] = "N/A (non-numeric)"
            else:
                avg_results[f"avg_{key}"] = "N/A (no valid data)"

    print("\n--- Evaluation Summary ---")
    print(f"Model Type: {model_type.upper()}")
    print(f"Test Set Size: {len(predictions)}")
    print("\n-- Precursor Metrics --")
    print(
        f"{'avg_precursors_exact_match':<30}: \
            {avg_results.get('avg_precursors_exact_match', 'N/A'):.4f}"
    )
    print(
        f"{'avg_precursors_formula_jaccard':<30}: \
            {avg_results.get('avg_precursors_formula_jaccard', 'N/A'):.4f}"
    )
    print(
        f"{'avg_precursors_precision':<30}: \
            {avg_results.get('avg_precursors_precision', 'N/A'):.4f}"
    )
    print(
        f"{'avg_precursors_recall':<30}: \
            {avg_results.get('avg_precursors_recall', 'N/A'):.4f}"
    )
    print(
        f"{'avg_precursors_f1':<30}: \
            {avg_results.get('avg_precursors_f1', 'N/A'):.4f}"
    )

    print("\n-- Operation Metrics --")
    exact_match_note = "(Not applicable for RF)" if is_rf else ""
    length_match_note = "(Not applicable for RF)" if is_rf else ""
    print(
        f"{'avg_operations_exact_match':<30}: \
            {avg_results.get('avg_operations_exact_match', 'N/A'):.4f} \
                {exact_match_note}"
    )
    print(
        f"{'avg_operations_length_match':<30}: \
            {avg_results.get('avg_operations_length_match', 'N/A'):.4f} \
                {length_match_note}"
    )
    print(
        f"{'avg_operations_type_jaccard':<30}: \
            {avg_results.get('avg_operations_type_jaccard', 'N/A'):.4f}"
    )
    print(
        f"{'avg_op_type_precision':<30}: \
            {avg_results.get('avg_op_type_precision', 'N/A'):.4f}"
    )
    print(
        f"{'avg_op_type_recall':<30}: \
            {avg_results.get('avg_op_type_recall', 'N/A'):.4f}"
    )
    print(
        f"{'avg_op_type_f1':<30}: \
          {avg_results.get('avg_op_type_f1', 'N/A'):.4f}"
    )

    if model_type == "knn":
        print("\n-- Note on Top-K Metrics --")
        print("k-NN uses k=1, retrieving only the single " "most similar recipe.")
        print(
            "P/R/F1 scores evaluate the accuracy of the "
            "retrieved sets from that neighbor."
        )

    print("\n--- Example Prediction ---")
    if len(predictions) > 0:
        example_idx_in_test = 0
        print(
            f"Test Entry Original Identifier: \
                {test_original_ids[example_idx_in_test]}"
        )
        print(
            f"Test Target Formula         : \
              {y_test_targets[example_idx_in_test]}"
        )
        print("\nPredicted Recipe/Components:")
        print(json.dumps(predictions[example_idx_in_test], indent=2))
        print("\nTrue Recipe:")
        print(json.dumps(y_test_recipes[example_idx_in_test], indent=2))
        print("\nEval Metrics for this Example:")
        print(json.dumps(evaluation_results[example_idx_in_test], indent=2))

    print("\nEvaluation step completed successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate k-NN or RF model.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="knn",
        choices=["knn", "rf"],
        help="Type of model to evaluate (knn or rf)",
    )
    args = parser.parse_args()

    if not run_evaluate(model_type=args.model_type):
        sys.exit(1)
