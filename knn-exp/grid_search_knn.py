# grid_search_knn.py

import numpy as np
import itertools
import os
import sys
import time
import shutil
import argparse
import json

# Import run functions
try:
    from featurize import run_featurize, NUM_ELEMENT_PROPS, ELEMENT_PROPS
    from train import run_train
    from evaluate import run_evaluate
except ImportError as e:
    print(f"Error importing pipeline modules: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
# Final output directories for the *best* model
BEST_FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-weighted/")
BEST_MODEL_DIR = os.path.join(DATA_DIR, "models/knn-weighted-model/")
# Temporary directories for grid search intermediate files
GRID_SEARCH_TEMP_BASE_DIR = os.path.join(DATA_DIR, "grid_search_temp") # Base temp dir

# Grid Search Parameters
# Define the possible values for each weight in the z vector
GRID_VALUES_DEFAULT = np.linspace(0, 1, 5)
# Metric to maximize
METRIC_TO_OPTIMIZE_DEFAULT = 'avg_precursors_f1'

def run_grid_search(grid_values, metric_to_optimize):
    """Performs grid search over z_weights for ELEMENT_PROPS."""
    print("--- Starting k-NN Grid Search for Feature Weights ---")
    print(f"Grid values per weight: {grid_values}")
    print(f"Metric to optimize: {metric_to_optimize}")
    print(f"Number of properties to weight: {NUM_ELEMENT_PROPS}")

    if NUM_ELEMENT_PROPS <= 0:
        print("Error: NUM_ELEMENT_PROPS is zero.", file=sys.stderr)
        return False

    # Generate weight combinations
    weight_combinations = list(itertools.product(grid_values, repeat=NUM_ELEMENT_PROPS))
    num_combinations = len(weight_combinations)
    print(f"Generated {num_combinations} weight combinations.")

    best_metric = -float('inf')
    best_z_weights = None
    results = []

    # Create unique temporary subdirectories for this run
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    temp_feature_dir = os.path.join(GRID_SEARCH_TEMP_BASE_DIR, f"features_{run_timestamp}")
    temp_model_dir = os.path.join(GRID_SEARCH_TEMP_BASE_DIR, f"model_{run_timestamp}")
    os.makedirs(temp_feature_dir, exist_ok=True)
    os.makedirs(temp_model_dir, exist_ok=True)
    print(f"Using temporary feature directory: {temp_feature_dir}")
    print(f"Using temporary model directory: {temp_model_dir}")


    start_grid_time = time.time()
    for i, z_weights_tuple in enumerate(weight_combinations):
        current_z = np.array(z_weights_tuple)
        print(f"\n[{i+1}/{num_combinations}] Testing z_weights = {current_z.tolist()}")
        iter_start_time = time.time()

        status = 'unknown_fail'
        current_metric = None

        # 1. Featurize
        print(f"  [{i+1}.1] Running featurization...")
        if run_featurize(z_weights=current_z, output_dir=temp_feature_dir):
            # 2. Train
            print(f"  [{i+1}.2] Running training...")
            if run_train(featurized_dir=temp_feature_dir, model_out_dir=temp_model_dir):
                # 3. Evaluate
                print(f"  [{i+1}.3] Running evaluation...")
                current_metric = run_evaluate(
                    featurized_dir=temp_feature_dir,
                    model_dir=temp_model_dir,
                    metric_to_optimize=metric_to_optimize,
                    verbose=False # Keep logs clean
                )
                if current_metric is not None:
                    status = 'success'
                    print(f"    Metric ({metric_to_optimize}): {current_metric:.4f}")
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_z_weights = current_z
                        print(f"    >>> New best metric found: {best_metric:.4f}")
                else:
                    status = 'evaluate_fail'
                    print(f"    Evaluation failed for z={current_z.tolist()}.")
            else:
                status = 'train_fail'
                print(f"    Training failed for z={current_z.tolist()}.")
        else:
            status = 'featurize_fail'
            print(f"    Featurization failed for z={current_z.tolist()}.")

        results.append({'z': current_z.tolist(), 'metric': current_metric, 'status': status})
        iter_end_time = time.time()
        print(f"  Iteration Time: {iter_end_time - iter_start_time:.2f}s")

    end_grid_time = time.time()
    print(f"\n--- Grid Search Finished ---")
    print(f"Total Grid Search Time: {(end_grid_time - start_grid_time)/60:.2f} minutes.")

    # Save grid search results summary
    try:
        summary_path = os.path.join(GRID_SEARCH_TEMP_BASE_DIR, f"grid_search_summary_{run_timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump({
                "metric_optimized": metric_to_optimize,
                "grid_values": grid_values,
                "best_metric": best_metric,
                "best_z_weights": best_z_weights.tolist() if best_z_weights is not None else None,
                "all_results": results
            }, f, indent=2)
        print(f"Saved grid search summary to {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save grid search summary: {e}", file=sys.stderr)


    if best_z_weights is None:
        print("\nError: No successful runs completed in the grid search.")
        success = False
    else:
        print(f"\nBest Overall {metric_to_optimize}: {best_metric:.4f}")
        print(f"Achieved with z_weights: {best_z_weights.tolist()}")
        print(f"Weights correspond to properties: {ELEMENT_PROPS}")

        print("\nRe-running pipeline with best weights to save final artifacts...")
        print("  Running final featurization...")
        if not run_featurize(z_weights=best_z_weights, output_dir=BEST_FEATURIZED_DATA_DIR):
            print("Error: Final featurization failed.", file=sys.stderr)
            success = False
        else:
            print("  Running final training...")
            if not run_train(featurized_dir=BEST_FEATURIZED_DATA_DIR, model_out_dir=BEST_MODEL_DIR):
                 print("Error: Final training failed.", file=sys.stderr)
                 success = False
            else:
                 print("  Running final evaluation (verbose)...")
                 final_metric_check = run_evaluate(
                     featurized_dir=BEST_FEATURIZED_DATA_DIR,
                     model_dir=BEST_MODEL_DIR,
                     metric_to_optimize=metric_to_optimize,
                     verbose=True # Show full output
                 )
                 if final_metric_check is None:
                      print("Error: Final evaluation failed.", file=sys.stderr)
                      success = False
                 else:
                      print(f"\nFinal Evaluation with Best Weights ({metric_to_optimize}): {final_metric_check:.4f}")
                      success = True

    # Clean up temporary directories for this specific run
    temp_dirs_to_remove = [temp_feature_dir, temp_model_dir]
    for temp_dir in temp_dirs_to_remove:
        try:
             if os.path.exists(temp_dir):
                  print(f"\nCleaning up temporary directory: {temp_dir}")
                  shutil.rmtree(temp_dir)
        except Exception as e: print(f"Warning: Could not remove temp dir {temp_dir}: {e}")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search for k-NN feature weights.')
    parser.add_argument('--grid_values', type=float, nargs='+', default=GRID_VALUES_DEFAULT,
                        help=f'List of values to test for each weight (default: {GRID_VALUES_DEFAULT})')
    parser.add_argument('--metric', type=str, default=METRIC_TO_OPTIMIZE_DEFAULT,
                        help=f'Metric from evaluate.py to optimize (default: {METRIC_TO_OPTIMIZE_DEFAULT})')
    args = parser.parse_args()

    if not run_grid_search(grid_values=args.grid_values, metric_to_optimize=args.metric):
        sys.exit(1)
