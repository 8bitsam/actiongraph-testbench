import os
import argparse

def run_pipeline(steps):
    """Run the full pipeline or selected steps."""
    if 'filter' in steps or 'all' in steps:
        print("Step 1: Filtering synthesis data...")
        from filter_synthesis_data import filter_synthesis_data
        filter_synthesis_data()
    
    if 'convert' in steps or 'all' in steps:
        print("Step 2: Converting to action graphs...")
        from convert_to_action_graphs import convert_to_action_graphs
        convert_to_action_graphs()
    
    if 'featurize' in steps or 'all' in steps:
        print("Step 3: Featurizing action graphs...")
        from featurize_action_graphs import featurize_action_graphs
        featurize_action_graphs()
    
    if 'train' in steps or 'all' in steps:
        print("Step 4: Training GNN model...")
        from train_gnn import train_model
        try:
            train_model()
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
    
    if 'evaluate' in steps or 'all' in steps:
        print("Step 5: Evaluating model...")
        from evaluate_model import evaluate_model
        try:
            evaluate_model()
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inorganic Synthesis Prediction Pipeline')
    parser.add_argument('--steps', nargs='+', 
                        choices=['filter', 'convert', 'featurize', 'train', 'evaluate', 'all'], 
                        default=['all'], help='Steps to run in the pipeline')
    
    args = parser.parse_args()
    run_pipeline(args.steps)
