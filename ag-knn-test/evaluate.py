import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
FEATURIZED_DATA_DIR_AG = os.path.join(DATA_DIR,
                                      "featurized-data-actiongraph/")
MODEL_AG_BASE_DIR = os.path.join(DATA_DIR, "models_ag")

TEST_SIZE = 0.25
RANDOM_STATE = 42


def calculate_prf1(pred_set, true_set):
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

    if not true_set:
        recall = 1.0 if not pred_set else 1.0
        recall = 1.0 if not pred_set else 0.0
        recall = 1.0
    elif (true_positives + false_negatives) == 0:
        recall = 1.0
        recall = 0.0
    else:
        recall = true_positives / (true_positives + false_negatives)

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compare_recipes_detailed(pred_recipe, true_recipe, is_rf=False):
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
    pred_recipe_dict = pred_recipe if isinstance(pred_recipe, dict) else {}
    true_recipe_dict = true_recipe if isinstance(true_recipe, dict) else {}

    pred_precursors_list = pred_recipe_dict.get("precursors", [])
    true_precursors_list = true_recipe_dict.get("precursors", [])
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

    if pred_prec_formulas == true_prec_formulas and pred_prec_formulas:
        results["precursors_exact_match"] = True

    intersection_prec = \
        len(pred_prec_formulas.intersection(true_prec_formulas))
    union_prec = len(pred_prec_formulas.union(true_prec_formulas))
    if union_prec > 0:
        results["precursors_formula_jaccard"] = \
            intersection_prec / union_prec

    prec_p, prec_r, prec_f1 = calculate_prf1(pred_prec_formulas,
                                             true_prec_formulas)
    (
        results["precursors_precision"],
        results["precursors_recall"],
        results["precursors_f1"],
    ) = (prec_p, prec_r, prec_f1)

    pred_ops_list = pred_recipe_dict.get("operations", [])
    true_ops_list = true_recipe_dict.get("operations", [])
    if not isinstance(pred_ops_list, list):
        pred_ops_list = []
    if not isinstance(true_ops_list, list):
        true_ops_list = []

    if not is_rf:
        try:
            sorted_pred_ops_str = sorted(
                [
                    json.dumps(op, sort_keys=True)
                    for op in pred_ops_list
                    if isinstance(op, dict)
                ]
            )
            sorted_true_ops_str = sorted(
                [
                    json.dumps(op, sort_keys=True)
                    for op in true_ops_list
                    if isinstance(op, dict)
                ]
            )
            if sorted_pred_ops_str == sorted_true_ops_str and \
                sorted_pred_ops_str:
                results["operations_exact_match"] = True
        except Exception:
            pass

    pred_op_types = set(
        op["type"] for op in pred_ops_list if isinstance(op, dict) \
            and op.get("type")
    )
    true_op_types = set(
        op["type"] for op in true_ops_list if isinstance(op, dict) \
            and op.get("type")
    )

    intersection_op_types = len(pred_op_types.intersection(true_op_types))
    union_op_types = len(pred_op_types.union(true_op_types))
    if union_op_types > 0:
        results["operations_type_jaccard"] = \
            intersection_op_types / union_op_types

    op_type_p, op_type_r, op_type_f1 = calculate_prf1(pred_op_types, \
                                                      true_op_types)
    results["op_type_precision"], results["op_type_recall"], \
        results["op_type_f1"] = (
        op_type_p,
        op_type_r,
        op_type_f1,
    )

    if not is_rf:
        results["operations_length_match"] = \
            len(pred_ops_list) == len(true_ops_list)
    return results


def run_evaluate_ag(model_type="knn"):
    if model_type != "knn":
        print(
            f"Error: This script is configured for k-NN with ActionGraph \
                features. Model type '{model_type}' not supported here.",
            file=sys.stderr,
        )
        return None

    print(
        f"--- Running Evaluation Step (Model: {model_type.upper()} \
            with ActionGraph Features) ---"
    )

    MODEL_DIR = os.path.join(MODEL_AG_BASE_DIR, f"{model_type}-model")
    print(f"Loading AG model artifacts from: {MODEL_DIR}")

    print(f"\nLoading AG-featurized data from: {FEATURIZED_DATA_DIR_AG}")
    try:
        all_X_ag = np.load(os.path.join(FEATURIZED_DATA_DIR_AG, \
                                        "features_ag.npy"))
        with open(
            os.path.join(FEATURIZED_DATA_DIR_AG, \
                         "recipes_mp_style_for_ag.json"), "r"
        ) as f:
            all_recipes_mp_style = json.load(f)
        with open(
            os.path.join(FEATURIZED_DATA_DIR_AG, \
                         "targets_ag_output_formula.json"), "r"
        ) as f:
            all_targets_ag_formula = json.load(f)
        with open(
            os.path.join(FEATURIZED_DATA_DIR_AG, \
                         "original_ag_ids.json"), "r"
        ) as f:
            all_original_ag_ids = json.load(f)

        if not (
            len(all_X_ag)
            == len(all_recipes_mp_style)
            == len(all_targets_ag_formula)
            == len(all_original_ag_ids)
        ):
            raise ValueError("Mismatch between loaded \
                             AG-featurized data lengths.")
        print(f"Loaded {len(all_X_ag)} total AG-featurized samples.")
    except FileNotFoundError:
        print(
            f"Error: Featurized AG data not found in \
                {FEATURIZED_DATA_DIR_AG}. Run 'featurize.py'.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Error loading featurized AG data: {e}",
              file=sys.stderr)
        return None

    model_artifacts_ag = {}
    try:
        model_artifacts_ag["scaler_ag"] = joblib.load(
            os.path.join(MODEL_DIR, "scaler_ag.joblib")
        )
        model_artifacts_ag["knn_model_ag"] = joblib.load(
            os.path.join(MODEL_DIR, "knn_model_ag.joblib")
        )
        model_artifacts_ag["train_original_indices_ag"] = joblib.load(
            os.path.join(MODEL_DIR, "train_original_indices_ag.joblib")
        )
    except FileNotFoundError:
        print(
            f"Error: AG Model artifacts not found in {MODEL_DIR}. \
                Run 'train.py --model_type {model_type}'.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Error loading AG model artifacts: {e}", file=sys.stderr)
        return None

    print(
        f"\nRecreating train/test split for AG data (Test size: {TEST_SIZE}, \
            Random State: {RANDOM_STATE})..."
    )
    if len(all_X_ag) < 2:
        print("Error: Not enough AG data points (< 2) for split.",
              file=sys.stderr)
        return None

    original_indices_arr = np.arange(len(all_X_ag))
    _X_train_ag_feat, X_test_ag_feat, _train_orig_idx, \
        test_orig_idx = train_test_split(
        all_X_ag, original_indices_arr, test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    y_test_recipes_mp_style = [all_recipes_mp_style[i] for i in test_orig_idx]
    y_test_targets_formula = [all_targets_ag_formula[i] for i in test_orig_idx]
    test_set_original_ag_ids = [all_original_ag_ids[i] for i in test_orig_idx]

    print(f"Test set size (AG features): {len(X_test_ag_feat)}")
    if len(X_test_ag_feat) == 0:
        print(
            "Warning: Test set (AG features) has size 0 after split. \
                Evaluation may not be meaningful.",
            file=sys.stderr,
        )
        return {"avg_precursors_f1": 0.0, "avg_op_type_f1": 0.0}

    print("\nScaling AG test features using loaded scaler...")
    scaler_ag = model_artifacts_ag["scaler_ag"]
    X_test_ag_scaled = scaler_ag.transform(X_test_ag_feat)
    X_test_ag_scaled = np.nan_to_num(X_test_ag_scaled, nan=0.0,
                                     posinf=0.0, neginf=0.0)

    print(
        f"\nPredicting recipes for the AG test set using \
            {model_type.upper()} model..."
    )
    predicted_recipes_mp_style = []

    if model_type == "knn":
        knn_model_ag = model_artifacts_ag["knn_model_ag"]
        train_original_indices_ag = np.array(
            model_artifacts_ag["train_original_indices_ag"]
        )
        _distances_all, neighbor_indices_in_train_subset = \
            knn_model_ag.kneighbors(
            X_test_ag_scaled
        )

        for i in range(len(X_test_ag_scaled)):
            nearest_neighbor_train_subset_idx = \
                neighbor_indices_in_train_subset[i][0]
            original_dataset_idx_of_neighbor = train_original_indices_ag[
                nearest_neighbor_train_subset_idx
            ]
            predicted_recipe = \
                all_recipes_mp_style[original_dataset_idx_of_neighbor]
            predicted_recipes_mp_style.append(predicted_recipe)
    else:
        print(
            f"Error: Prediction logic for model_type \
                '{model_type}' with AGs not implemented.",
            file=sys.stderr,
        )
        return None

    if (
        len(predicted_recipes_mp_style) != len(y_test_recipes_mp_style)
        and len(X_test_ag_feat) > 0
    ):
        print(
            f"Error: Mismatch in number of predictions \
                ({len(predicted_recipes_mp_style)}) and test \
                    samples ({len(y_test_recipes_mp_style)}).",
            file=sys.stderr,
        )
        return None

    print("\nEvaluating AG-based predictions with detailed metrics...")
    evaluation_results_list = []
    for i in range(len(predicted_recipes_mp_style)):
        pred_recipe = predicted_recipes_mp_style[i]
        true_recipe = y_test_recipes_mp_style[i]
        eval_metrics = compare_recipes_detailed(pred_recipe, \
                                                true_recipe, is_rf=False)
        evaluation_results_list.append(eval_metrics)

    avg_results = {}
    expected_metric_keys = [
        "precursors_exact_match",
        "precursors_formula_jaccard",
        "precursors_precision",
        "precursors_recall",
        "precursors_f1",
        "operations_exact_match",
        "operations_type_jaccard",
        "op_type_precision",
        "op_type_recall",
        "op_type_f1",
        "operations_length_match",
    ]

    if evaluation_results_list:
        for key in expected_metric_keys:
            current_key_values = [
                res.get(key)
                for res in evaluation_results_list
                if res.get(key) is not None
            ]
            if current_key_values:
                first_val = current_key_values[0]
                if isinstance(first_val, bool):
                    avg_results[f"avg_{key}"] = np.mean(
                        [float(v) for v in current_key_values]
                    )
                elif isinstance(first_val, (int, float)):
                    avg_results[f"avg_{key}"] = np.mean(current_key_values)
                else:
                    avg_results[f"avg_{key}"] = "N/A (non-numeric)"
            else:
                avg_results[f"avg_{key}"] = 0.0
    else:
        print("Warning: No evaluation results to aggregate.", file=sys.stderr)
        for key_template in expected_metric_keys:
            avg_results[f"avg_{key_template}"] = 0.0

    def format_metric_value(value):
        if isinstance(value, (float, np.floating, np.float64)):
            return f"{value:.4f}"
        return str(value)

    print("\n--- Evaluation Summary (ActionGraph Features) ---")
    print(f"Model Type: {model_type.upper()}")
    print(f"Test Set Size: {len(predicted_recipes_mp_style)}")

    print("\n-- Precursor Metrics --")
    print(
        f"{'avg_precursors_exact_match':<30}: \
            {format_metric_value(avg_results.get(
                'avg_precursors_exact_match', 0.0))}"
    )
    print(
        f"{'avg_precursors_formula_jaccard':<30}: \
            {format_metric_value(avg_results.get(
                'avg_precursors_formula_jaccard', 0.0))}"
    )
    print(
        f"{'avg_precursors_precision':<30}: {format_metric_value(
            avg_results.get('avg_precursors_precision', 0.0))}"
    )
    print(
        f"{'avg_precursors_recall':<30}: {format_metric_value(avg_results.get(
            'avg_precursors_recall', 0.0))}"
    )
    print(
        f"{'avg_precursors_f1':<30}: {format_metric_value(avg_results.get(
            'avg_precursors_f1', 0.0))}"
    )

    print("\n-- Operation Metrics --")
    print(
        f"{'avg_operations_exact_match':<30}: \
            {format_metric_value(avg_results.get(
                'avg_operations_exact_match', 0.0))}"
    )
    print(
        f"{'avg_operations_length_match':<30}: \
            {format_metric_value(avg_results.get(
                'avg_operations_length_match', 0.0))}"
    )
    print(
        f"{'avg_operations_type_jaccard':<30}: \
            {format_metric_value(avg_results.get(\
                'avg_operations_type_jaccard', 0.0))}"
    )
    print(
        f"{'avg_op_type_precision':<30}: \
            {format_metric_value(avg_results.get(
                'avg_op_type_precision', 0.0))}"
    )
    print(
        f"{'avg_op_type_recall':<30}: \
            {format_metric_value(avg_results.get(
                'avg_op_type_recall', 0.0))}"
    )
    print(
        f"{'avg_op_type_f1':<30}: {format_metric_value(avg_results.get(
            'avg_op_type_f1', 0.0))}"
    )

    print("\nAG Evaluation step completed successfully.")

    prec_f1_val = avg_results.get("avg_precursors_f1", 0.0)
    op_f1_val = avg_results.get("avg_op_type_f1", 0.0)

    return_metrics = {
        "avg_precursors_f1": (
            float(prec_f1_val)
            if isinstance(prec_f1_val, (int, float, np.number))
            else 0.0
        ),
        "avg_op_type_f1": (
            float(op_f1_val) if isinstance(op_f1_val,
                                           (int, float, np.number)) else 0.0
        ),
    }
    return return_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate k-NN model trained on ActionGraph features."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="knn",
        choices=["knn"],
        help="Type of model to evaluate (fixed to knn for AG features)",
    )
    args = parser.parse_args()

    results = run_evaluate_ag(model_type=args.model_type)
    if results is None:
        print("Evaluation script failed to produce results.")
        sys.exit(1)
    else:
        print("\nReturned Metrics for Scripting:")
        print(json.dumps(results, indent=2))
