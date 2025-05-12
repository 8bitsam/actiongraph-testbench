# pipeline_ag.py

import argparse
import sys
import time

# Ensure imports work correctly from the AG-specific versions
try:
    from evaluate import run_evaluate_ag
    from featurize import run_featurize_ag
    from train import run_train_ag
except ImportError as e:
    print(f"Error importing ActionGraph pipeline modules: {e}", file=sys.stderr)
    print(
        "Ensure featurize_ag.py, train_ag.py, evaluate_ag.py are accessible.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Inorganic Synthesis Prediction Pipeline (ActionGraph Version)"
    )
    # For this AG strategy, model_type is effectively fixed to 'knn'
    parser.add_argument(
        "--model_type",
        type=str,
        default="knn",
        choices=["knn"],
        help="Type of model to use (currently fixed to knn for AG pipeline)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["featurize_ag", "train_ag", "evaluate_ag", "all_ag"],
        default=["all_ag"],
        help="Steps to run in the AG pipeline",
    )

    args = parser.parse_args()
    steps_to_run = args.steps
    model_type = args.model_type  # Will be 'knn'

    if model_type != "knn":
        print(
            f"Warning: This pipeline version is optimized for 'knn' with ActionGraph features. "
            f"Specified model_type '{model_type}' will proceed as 'knn'.",
            file=sys.stderr,
        )
        model_type = "knn"

    start_pipeline_time = time.time()
    print(f"--- Starting ActionGraph Pipeline (Model: {model_type.upper()}) ---")
    print(f"Steps to run: {', '.join(steps_to_run)}")

    success = True  # Track overall success

    # Featurization (ActionGraph specific)
    if "all_ag" in steps_to_run or "featurize_ag" in steps_to_run:
        print("\n>>> Executing Featurize AG Step <<<")
        step_start_time = time.time()
        if (
            not run_featurize_ag()
        ):  # run_featurize_ag is specific, doesn't need model_type
            print("Featurize AG step failed. Aborting pipeline.", file=sys.stderr)
            success = False
        else:
            print(
                f"Featurize AG step finished in {time.time() - step_start_time:.2f} seconds."
            )

    # Training (ActionGraph specific, k-NN model)
    if success and ("all_ag" in steps_to_run or "train_ag" in steps_to_run):
        print(f"\n>>> Executing Train AG Step ({model_type.upper()}) <<<")
        step_start_time = time.time()
        if not run_train_ag(model_type=model_type):  # Pass model_type (knn)
            print("Train AG step failed. Aborting subsequent steps.", file=sys.stderr)
            success = False
        else:
            print(
                f"Train AG step finished in {time.time() - step_start_time:.2f} seconds."
            )

    # Evaluation (ActionGraph specific, k-NN model)
    if success and ("all_ag" in steps_to_run or "evaluate_ag" in steps_to_run):
        print(f"\n>>> Executing Evaluate AG Step ({model_type.upper()}) <<<")
        step_start_time = time.time()
        if not run_evaluate_ag(model_type=model_type):  # Pass model_type (knn)
            print("Evaluate AG step failed.", file=sys.stderr)
            success = False
        else:
            print(
                f"Evaluate AG step finished in {time.time() - step_start_time:.2f} seconds."
            )

    end_pipeline_time = time.time()
    print(f"\n--- ActionGraph Pipeline Finished (Success: {success}) ---")
    print(
        f"Total AG pipeline execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds."
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
