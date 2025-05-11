# 5_pipeline_knn_comparison.py
import os
import sys
import time
import argparse
import json

try:
    from common_utils import ELEMENT_PROPS, NUM_ELEMENT_PROPS # For featurizer default
    # Import specific run functions with clearer names if needed
    from p1_ag_converter import run_ag_conversion_and_dedup # Renamed for clarity
    from p2a_featurize_baseline_knn import run_featurize_baseline
    from p2b_featurize_ag_cout_knn import run_featurize_ag_cout
    from p2c_featurize_ag_graph_knn import run_featurize_ag_graph
    from p3_train_knn_generic import run_train_knn
    from p4_evaluate_knn_generic import run_evaluate_knn
except ImportError as e:
    print(f"Error importing pipeline modules: {e}. Ensure all pX_*.py files are present.", file=sys.stderr)
    sys.exit(1)

# --- Define Paths Based on common_utils.DATA_DIR ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
RAW_MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
ACTION_GRAPH_RAW_DIR = os.path.join(DATA_DIR, "filtered-ag-data/")

FEATURIZED_BASELINE_KNN_DIR = os.path.join(DATA_DIR, "featurized-baseline-knn/")
MODEL_BASELINE_KNN_DIR = os.path.join(DATA_DIR, "models/baseline-knn-model/")

FEATURIZED_AG_COUT_KNN_DIR = os.path.join(DATA_DIR, "featurized-ag-cout-knn/")
MODEL_AG_COUT_KNN_DIR = os.path.join(DATA_DIR, "models/ag-cout-knn-model/")

FEATURIZED_AG_GRAPH_KNN_DIR = os.path.join(DATA_DIR, "featurized-ag-graph-knn/")
MODEL_AG_GRAPH_KNN_DIR = os.path.join(DATA_DIR, "models/ag-graph-knn-model/")


def main():
    parser = argparse.ArgumentParser(description='k-NN Comparison Pipeline for Synthesis Prediction')
    parser.add_argument(
        '--steps', nargs='+',
        choices=[
            'convert_ag',
            'featurize_baseline', 'train_baseline', 'evaluate_baseline',
            'featurize_ag_cout', 'train_ag_cout', 'evaluate_ag_cout',
            'featurize_ag_graph', 'train_ag_graph', 'evaluate_ag_graph',
            'all_baseline', 'all_ag_cout', 'all_ag_graph', 'all'
        ],
        default=['all'],
        help='Steps to run in the pipeline'
    )
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help="Distance metric for k-NN training.")

    args = parser.parse_args()
    steps_to_run = args.steps
    knn_metric = args.metric

    start_pipeline_time = time.time()
    print("--- Starting k-NN Comparison Pipeline ---")
    print(f"k-NN Distance Metric: {knn_metric}")
    print(f"Steps to run: {', '.join(steps_to_run)}")
    overall_success = True

    def run_step(step_name, function_to_run, *func_args, **func_kwargs):
        nonlocal overall_success
        if not overall_success: # Skip if previous critical step failed
            print(f"Skipping step '{step_name}' due to previous failure.")
            return False
        print(f"\n>>> Executing Step: {step_name} <<<")
        step_start_time = time.time()
        success = function_to_run(*func_args, **func_kwargs)
        if not success:
            print(f"Step '{step_name}' FAILED.", file=sys.stderr)
            overall_success = False # Mark overall failure
        else:
            print(f"Step '{step_name}' finished in {time.time() - step_start_time:.2f}s.")
        return success

    # --- Common Step: Convert MP JSON to ActionGraph JSONs ---
    if any(s in steps_to_run for s in ['all', 'all_ag_cout', 'all_ag_graph', 'convert_ag',
                                       'featurize_ag_cout', 'featurize_ag_graph']):
        run_step("Convert MP to ActionGraphs", run_ag_conversion_and_dedup)

    # --- Path 1: Baseline k-NN (Target-only features from MP JSON) ---
    if any(s in steps_to_run for s in ['all', 'all_baseline', 'featurize_baseline']):
        run_step("Featurize Baseline k-NN", run_featurize_baseline)
    if any(s in steps_to_run for s in ['all', 'all_baseline', 'train_baseline']):
        run_step("Train Baseline k-NN", run_train_knn,
                 FEATURIZED_BASELINE_KNN_DIR, MODEL_BASELINE_KNN_DIR, knn_metric)
    if any(s in steps_to_run for s in ['all', 'all_baseline', 'evaluate_baseline']):
        run_step("Evaluate Baseline k-NN", run_evaluate_knn,
                 FEATURIZED_BASELINE_KNN_DIR, MODEL_BASELINE_KNN_DIR, True)


    # --- Path 2: ActionGraph-Informed k-NN (C_out node features from AG) ---
    if any(s in steps_to_run for s in ['all', 'all_ag_cout', 'featurize_ag_cout']):
        if not overall_success and 'convert_ag' not in steps_to_run: print("Skipping AG C_out featurization as AG conversion might not have run.")
        else: run_step("Featurize AG C_out for k-NN", run_featurize_ag_cout)
    if any(s in steps_to_run for s in ['all', 'all_ag_cout', 'train_ag_cout']):
        run_step("Train AG C_out k-NN", run_train_knn,
                 FEATURIZED_AG_COUT_KNN_DIR, MODEL_AG_COUT_KNN_DIR, knn_metric)
    if any(s in steps_to_run for s in ['all', 'all_ag_cout', 'evaluate_ag_cout']):
        run_step("Evaluate AG C_out k-NN", run_evaluate_knn,
                 FEATURIZED_AG_COUT_KNN_DIR, MODEL_AG_COUT_KNN_DIR, True)

    # --- Path 3: ActionGraph-Informed k-NN (Simple Graph Features from AG) ---
    if any(s in steps_to_run for s in ['all', 'all_ag_graph', 'featurize_ag_graph']):
        if not overall_success and 'convert_ag' not in steps_to_run: print("Skipping AG Graph featurization as AG conversion might not have run.")
        else: run_step("Featurize AG Graph Stats for k-NN", run_featurize_ag_graph)
    if any(s in steps_to_run for s in ['all', 'all_ag_graph', 'train_ag_graph']):
        run_step("Train AG Graph Stats k-NN", run_train_knn,
                 FEATURIZED_AG_GRAPH_KNN_DIR, MODEL_AG_GRAPH_KNN_DIR, knn_metric)
    if any(s in steps_to_run for s in ['all', 'all_ag_graph', 'evaluate_ag_graph']):
        run_step("Evaluate AG Graph Stats k-NN", run_evaluate_knn,
                 FEATURIZED_AG_GRAPH_KNN_DIR, MODEL_AG_GRAPH_KNN_DIR, True)


    end_pipeline_time = time.time()
    print(f"\n--- Full Comparison Pipeline Finished (Overall Success: {overall_success}) ---")
    print(f"Total execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds.")
    if not overall_success:
        sys.exit(1)

if __name__ == "__main__":
    main()
