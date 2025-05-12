# run_pca_experiment.py

import argparse  # For command-line arguments
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Ensure pipeline modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from evaluate import run_evaluate_ag
from featurize import run_featurize_ag
from train import run_train_ag

# --- Configuration for the Experiment ---
PCA_COMPONENTS_RANGE = range(10, 31)
RESULTS_DIR = os.path.abspath(
    os.path.join(current_dir, "..", "Data", "pca_experiment_results_pub/")
)
DEFAULT_RESULTS_JSON_FILENAME = "pca_experiment_f1_scores.json"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication Style Settings
plt.style.use("seaborn-v0_8-whitegrid")
PUB_FONT_SIZE_SMALL = 12
PUB_FONT_SIZE_MEDIUM = 14
PUB_FIGURE_DPI = 300

plt.rcParams.update(
    {
        "font.size": PUB_FONT_SIZE_MEDIUM,
        "axes.labelsize": PUB_FONT_SIZE_MEDIUM,
        "xtick.labelsize": PUB_FONT_SIZE_SMALL,
        "ytick.labelsize": PUB_FONT_SIZE_SMALL,
        "legend.fontsize": PUB_FONT_SIZE_SMALL,
    }
)


# --- Plotting Function (can be called standalone) ---
def plot_f1_vs_pca_from_data(
    pca_values, precursor_f1_scores, operation_f1_scores, output_dir
):
    """Generates and saves the F1 vs.

    PCA components plot.
    """
    print("\nPlotting F1 scores vs. PCA Components (Publication Style)...")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory for plot exists

    plt.figure(figsize=(8, 5.5))

    # Filter out NaN values for plotting if any iterations failed or data is incomplete
    # Convert to numpy arrays for safe boolean indexing
    pca_values_arr = np.array(pca_values)
    precursor_f1_scores_arr = np.array(precursor_f1_scores)
    operation_f1_scores_arr = np.array(operation_f1_scores)

    valid_indices = ~np.isnan(precursor_f1_scores_arr) & ~np.isnan(
        operation_f1_scores_arr
    )

    plotted_pca_values = pca_values_arr[valid_indices]
    plotted_prec_f1 = precursor_f1_scores_arr[valid_indices]
    plotted_ops_f1 = operation_f1_scores_arr[valid_indices]

    if len(plotted_pca_values) > 0:
        plt.plot(
            plotted_pca_values,
            plotted_prec_f1,
            marker="o",
            linestyle="-",
            color="royalblue",
            label="Precursors F1 Score",
            markersize=6,
            linewidth=1.5,
        )
        plt.plot(
            plotted_pca_values,
            plotted_ops_f1,
            marker="s",
            linestyle="--",
            color="orangered",
            label="Operations Type F1 Score",
            markersize=6,
            linewidth=1.5,
        )

        plt.xlabel("Number of PCA Components")
        plt.ylabel("Average F1 Score")

        if len(plotted_pca_values) > 1:
            min_x, max_x = min(plotted_pca_values), max(plotted_pca_values)
            tick_step = max(1, (max_x - min_x) // 10 if (max_x - min_x) > 0 else 1)
            if len(plotted_pca_values) <= 10:
                tick_step = 1
            plt.xticks(ticks=np.arange(min_x, max_x + tick_step, step=tick_step))
        elif len(plotted_pca_values) == 1:
            plt.xticks(ticks=plotted_pca_values)

        plt.legend(loc="best")
        plt.grid(True, linestyle=":", alpha=0.7)

        all_plotted_f1_scores = np.concatenate((plotted_prec_f1, plotted_ops_f1))
        if len(all_plotted_f1_scores) > 0:
            min_f1 = np.nanmin(all_plotted_f1_scores)
            max_f1 = np.nanmax(all_plotted_f1_scores)

            padding = (max_f1 - min_f1) * 0.10
            if padding < 0.01 and max_f1 - min_f1 < 0.02:
                padding = 0.01  # Ensure minimal padding if range is tiny

            plot_bottom = max(0, min_f1 - padding)
            plot_top = min(1, max_f1 + padding)

            if (
                plot_bottom >= plot_top
            ):  # Handle cases where all values are identical or very close to 0 or 1
                if max_f1 > 0.01:  # If scores are not all zero
                    plot_bottom = max(0, max_f1 - 0.02)
                    plot_top = min(1, max_f1 + 0.02)
                else:  # All scores are essentially zero
                    plot_bottom = 0
                    plot_top = 0.1
                if plot_bottom >= plot_top:  # Final fallback for identical values
                    plot_bottom = max_f1 - 0.01 if max_f1 > 0.01 else 0
                    plot_top = max_f1 + 0.01 if max_f1 < 0.99 else 1.0

            plt.ylim(bottom=plot_bottom, top=plot_top)
        else:
            plt.ylim(bottom=0, top=1)

        plt.tight_layout()

        plot_save_path = os.path.join(output_dir, "pca_components_vs_f1_pub.png")
        plt.savefig(plot_save_path, dpi=PUB_FIGURE_DPI)
        print(f"Plot saved to {plot_save_path}")
        # plt.show() # Commented out for batch runs
    else:
        print("No valid data points to plot.")


def plot_results_from_json(json_filepath, output_dir_for_plot):
    """Loads experiment results from a JSON file and generates the plot."""
    print(f"Attempting to plot results from: {json_filepath}")
    if not os.path.exists(json_filepath):
        print(f"Error: Results JSON file not found at {json_filepath}", file=sys.stderr)
        return False

    try:
        with open(json_filepath, "r") as f:
            results_data = json.load(f)
    except Exception as e:
        print(
            f"Error reading or parsing JSON file {json_filepath}: {e}", file=sys.stderr
        )
        return False

    pca_components = results_data.get("pca_components")
    precursor_f1 = results_data.get("precursor_f1_scores")
    operation_f1 = results_data.get("operation_f1_scores")

    if pca_components is None or precursor_f1 is None or operation_f1 is None:
        print(
            "Error: JSON file is missing one or more required keys ('pca_components', 'precursor_f1_scores', 'operation_f1_scores').",
            file=sys.stderr,
        )
        return False

    plot_f1_vs_pca_from_data(
        pca_components, precursor_f1, operation_f1, output_dir_for_plot
    )
    return True


# --- Main Experiment Loop ---
def run_full_experiment():
    print("--- Starting Full PCA Components Experiment ---")

    pca_values_loop = list(PCA_COMPONENTS_RANGE)
    precursor_f1_scores_collected = []
    operation_f1_scores_collected = []

    for n_components_loop in pca_values_loop:
        print(
            f"\n----- Running pipeline for PCA Components = {n_components_loop} -----"
        )

        print(f"\nStep 1: Featurizing with {n_components_loop} PCA components...")
        start_time_step = time.time()
        if not run_featurize_ag(pca_n_components_arg=n_components_loop):
            print(
                f"Featurization failed for {n_components_loop} components. Appending NaN and skipping."
            )
            precursor_f1_scores_collected.append(np.nan)
            operation_f1_scores_collected.append(np.nan)
            continue
        print(f"Featurization took {time.time() - start_time_step:.2f}s")

        print("\nStep 2: Training model...")
        start_time_step = time.time()
        if not run_train_ag(model_type="knn"):
            print(
                f"Training failed for {n_components_loop} components. Appending NaN and skipping."
            )
            precursor_f1_scores_collected.append(np.nan)
            operation_f1_scores_collected.append(np.nan)
            continue
        print(f"Training took {time.time() - start_time_step:.2f}s")

        print("\nStep 3: Evaluating model...")
        start_time_step = time.time()
        eval_metrics = run_evaluate_ag(model_type="knn")
        if not eval_metrics or not isinstance(eval_metrics, dict):
            print(
                f"Evaluation failed or returned invalid metrics for {n_components_loop} components. Appending NaN."
            )
            precursor_f1_scores_collected.append(np.nan)
            operation_f1_scores_collected.append(np.nan)
            continue

        current_prec_f1 = eval_metrics.get(
            "avg_precursors_f1", np.nan
        )  # Use NaN if key missing
        current_ops_f1 = eval_metrics.get(
            "avg_op_type_f1", np.nan
        )  # Use NaN if key missing

        print(f"Evaluation took {time.time() - start_time_step:.2f}s")
        print(
            f"  Results for {n_components_loop} PCA: Precursor F1 = {current_prec_f1:.4f}, Operations F1 = {current_ops_f1:.4f}"
        )

        precursor_f1_scores_collected.append(current_prec_f1)
        operation_f1_scores_collected.append(current_ops_f1)

        time.sleep(0.2)  # Short pause, e.g. for file systems to catch up if needed

    results_data_to_save = {
        "pca_components": pca_values_loop,
        "precursor_f1_scores": precursor_f1_scores_collected,
        "operation_f1_scores": operation_f1_scores_collected,
    }
    results_json_path = os.path.join(RESULTS_DIR, DEFAULT_RESULTS_JSON_FILENAME)
    with open(results_json_path, "w") as f:
        json.dump(results_data_to_save, f, indent=2)
    print(f"\nRaw F1 scores saved to {results_json_path}")

    # Plot the results from the just-completed experiment
    plot_f1_vs_pca_from_data(
        pca_values_loop,
        precursor_f1_scores_collected,
        operation_f1_scores_collected,
        RESULTS_DIR,
    )

    print("\n--- Full PCA Components Experiment Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PCA component experiment or plot results from JSON."
    )
    parser.add_argument(
        "--plot_from_json",
        type=str,
        default=None,
        help=f"Path to the results JSON file to plot. If provided, skips the experiment. Defaults to checking for '{DEFAULT_RESULTS_JSON_FILENAME}' in RESULTS_DIR if only flag is present.",
    )
    parser.add_argument(
        "--run_experiment",
        action="store_true",
        help="Flag to run the full PCA experiment.",
    )

    args = parser.parse_args()

    if args.plot_from_json:
        json_file_to_plot = args.plot_from_json
        if not os.path.isabs(json_file_to_plot) and not os.path.exists(
            json_file_to_plot
        ):
            # If relative path and not found, try constructing from RESULTS_DIR
            json_file_to_plot = os.path.join(RESULTS_DIR, json_file_to_plot)

        if not plot_results_from_json(json_file_to_plot, RESULTS_DIR):
            print(
                f"Could not plot from {json_file_to_plot}. Try running the experiment first."
            )
            sys.exit(1)
    elif args.run_experiment:
        run_full_experiment()
    else:
        # Default behavior: try to plot from default JSON if it exists, otherwise suggest running experiment
        default_json_path = os.path.join(RESULTS_DIR, DEFAULT_RESULTS_JSON_FILENAME)
        if os.path.exists(default_json_path):
            print(
                f"No specific action requested. Attempting to plot from default results file: {default_json_path}"
            )
            plot_results_from_json(default_json_path, RESULTS_DIR)
        else:
            print("No action specified and no default results JSON found.")
            print(f"Use --run_experiment to run the experiment and generate results.")
            print(
                f"Or use --plot_from_json <path_to_your_results.json> to plot existing results."
            )
            parser.print_help()
