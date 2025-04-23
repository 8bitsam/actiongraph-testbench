# pipeline.py

import argparse
import sys
import time

# Ensure imports work correctly
try:
    from featurize import run_featurize
    from train import run_train
    from evaluate import run_evaluate
except ImportError as e:
    print(f"Error importing pipeline modules: {e}", file=sys.stderr)
    print("Ensure featurize.py, train.py, evaluate.py are accessible.", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Inorganic Synthesis Prediction Pipeline')
    parser.add_argument('--model_type', type=str, default='knn', choices=['knn', 'rf'],
                        help='Type of model to use (knn or rf)')
    parser.add_argument('--steps', nargs='+',
                        choices=['featurize', 'train', 'evaluate', 'all'],
                        default=['all'], help='Steps to run in the pipeline')

    args = parser.parse_args()
    steps_to_run = args.steps
    model_type = args.model_type

    start_pipeline_time = time.time()
    print(f"--- Starting Pipeline (Model: {model_type.upper()}) ---")
    print(f"Steps to run: {', '.join(steps_to_run)}")

    success = True # Track overall success

    # Featurization is model-independent
    if 'all' in steps_to_run or 'featurize' in steps_to_run:
        print("\n>>> Executing Featurize Step <<<")
        step_start_time = time.time()
        # run_featurize doesn't depend on model_type
        if not run_featurize():
            print("Featurize step failed. Aborting pipeline.", file=sys.stderr)
            success = False
        else:
            print(f"Featurize step finished in {time.time() - step_start_time:.2f} seconds.")

    # Training depends on model_type
    if success and ('all' in steps_to_run or 'train' in steps_to_run):
        print(f"\n>>> Executing Train Step ({model_type.upper()}) <<<")
        step_start_time = time.time()
        # Pass model_type to run_train
        if not run_train(model_type=model_type):
            print("Train step failed. Aborting subsequent steps.", file=sys.stderr)
            success = False
        else:
             print(f"Train step finished in {time.time() - step_start_time:.2f} seconds.")

    # Evaluation depends on model_type
    if success and ('all' in steps_to_run or 'evaluate' in steps_to_run):
        print(f"\n>>> Executing Evaluate Step ({model_type.upper()}) <<<")
        step_start_time = time.time()
        # Pass model_type to run_evaluate
        if not run_evaluate(model_type=model_type):
            print("Evaluate step failed.", file=sys.stderr)
            success = False
        else:
             print(f"Evaluate step finished in {time.time() - step_start_time:.2f} seconds.")

    end_pipeline_time = time.time()
    print(f"\n--- Pipeline Finished (Success: {success}) ---")
    print(f"Total pipeline execution time: {end_pipeline_time - start_pipeline_time:.2f} seconds.")

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
