# ag_evaluate_gatv2.py
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F # Though not directly used here, often used with model outputs
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # For type hinting mainly
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib # For loading model/scaler if they were saved with joblib (usually .pt for torch)

# Import model definition and ActionGraph class
try:
    from ag_train_gatv2 import GATInversePredictor # Model definition
    from actiongraph import ActionGraph # For qualitative comparison
    # from ag_featurizer_gatv2 import ELEMENT_PROPS, OP_TYPES_ENUM # Not strictly needed here
except ImportError as e:
    print(f"Error importing necessary modules (GATInversePredictor, ActionGraph): {e}", file=sys.stderr)
    print("Ensure ag_train_gatv2.py and actiongraph.py are accessible.", file=sys.stderr)
    # Define a placeholder if direct import fails for GATInversePredictor
    if 'GATInversePredictor' not in locals(): GATInversePredictor = torch.nn.Module
    if 'ActionGraph' not in locals(): ActionGraph = object # Placeholder
    # sys.exit(1) # Allow script to run further to catch other errors if needed

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
FEATURIZED_AG_DIR = os.path.join(DATA_DIR, "ag-data-featurized-gatv2/")
TEST_DATA_DIR = os.path.join(FEATURIZED_AG_DIR, "testing-data") # Where test .pt files are
ACTION_GRAPH_RAW_DIR = os.path.join(DATA_DIR, "filtered-ag-data") # For qualitative comparison source JSONs
MODEL_DIR = os.path.join(DATA_DIR, "models/gat-model-v2/")
PLOTS_DIR = os.path.join(MODEL_DIR, "evaluation_plots_gatv2") # Subdir for plots
BATCH_SIZE = 64 # Batch size for evaluation

# --- Helper: P/R/F1 for set comparison (if needed for a different type of evaluation) ---
# Not directly used for the current node classification output, but kept for reference
def calculate_set_prf1(pred_set, true_set):
    if not isinstance(pred_set, set): pred_set = set(pred_set)
    if not isinstance(true_set, set): true_set = set(true_set)
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set.difference(true_set))
    fn = len(true_set.difference(pred_set))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not true_set else 0.0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

# --- Main Evaluation Function ---
def run_ag_evaluate_gatv2():
    print("--- Running GATv2 Model Evaluation for Masked Prediction ---")
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create plots directory {PLOTS_DIR}: {e}", file=sys.stderr)


    if not os.path.isdir(MODEL_DIR):
        print(f"Error: Model directory not found: {MODEL_DIR}", file=sys.stderr); return False
    if not os.path.isdir(TEST_DATA_DIR):
        print(f"Error: Test data directory not found: {TEST_DATA_DIR}", file=sys.stderr); return False
    if not os.path.isdir(FEATURIZED_AG_DIR):
        print(f"Error: Main featurized directory not found: {FEATURIZED_AG_DIR}", file=sys.stderr); return False


    print("Loading metadata, train_config, and test file list...")
    try:
        metadata_path = os.path.join(FEATURIZED_AG_DIR, 'metadata_gatv2.json')
        with open(metadata_path, 'r') as f: metadata = json.load(f)
        gat_input_dim = metadata['gat_input_feature_dim']
        num_op_type_classes = metadata['num_op_type_classes']
        cin_target_feature_dim = metadata['chem_feature_dim']
        op_types_list_str = metadata.get('op_types_list_str') # Key from featurizer
        if op_types_list_str is None: raise KeyError("'op_types_list_str' not found in metadata.")
        class_names_for_report = op_types_list_str

        train_config_path = os.path.join(MODEL_DIR, 'train_config.json')
        with open(train_config_path, 'r') as f: train_config = json.load(f)
        hidden_channels = train_config['hidden_channels']
        num_heads = train_config['num_heads']
        gat_layers = train_config.get('gat_layers', 2) # Default if not in older config

        test_list_path = os.path.join(FEATURIZED_AG_DIR, 'gatv2_test_files.json') # Path where featurizer saved it
        with open(test_list_path, 'r') as f: test_files_basenames = json.load(f)
        test_files = [os.path.join(TEST_DATA_DIR, basename) for basename in test_files_basenames]

        if not test_files: print("Warning: No test files specified in list."); return True # Nothing to eval
        print(f"Found {len(test_files)} test graph files to evaluate.")

    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e}. Ensure featurization and training were run.", file=sys.stderr)
        return False
    except KeyError as e:
        print(f"Error: Missing key {e} in metadata_gatv2.json or train_config.json.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error loading configuration files: {e}", file=sys.stderr)
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if GATInversePredictor == torch.nn.Module and 'GATInversePredictor' not in sys.modules.get('ag_train_gatv2', {}):
         print("Error: GATInversePredictor class definition not properly imported/available.", file=sys.stderr)
         return False

    model = GATInversePredictor(
        in_channels=gat_input_dim, hidden_channels=hidden_channels,
        num_op_type_classes=num_op_type_classes,
        cin_target_feature_dim=cin_target_feature_dim,
        gat_layers=gat_layers, heads=num_heads, dropout=0.0 # Dropout is 0.0 for evaluation
    ).to(device)
    model_path = os.path.join(MODEL_DIR, 'gat_v2_masked_best.pt')
    if not os.path.exists(model_path): model_path = os.path.join(MODEL_DIR, 'gat_v2_masked_final.pt')
    if not os.path.exists(model_path): print(f"Error: Model file not found at {model_path}.", file=sys.stderr); return False

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from {model_path}.")
    except Exception as e: print(f"Error loading model state dict: {e}", file=sys.stderr); return False

    class GraphDataset(torch.utils.data.Dataset):
        def __init__(self, fl): self.file_list = fl
        def __len__(self): return len(self.file_list)
        def __getitem__(self, i):
            try: return torch.load(self.file_list[i], weights_only=False)
            except Exception as e: print(f"Warning: Error loading {self.file_list[i]}: {e}"); return None
    def collate_fn(b):
        b = [d for d in b if d is not None]; return DataLoader.collate(b) if b else None

    test_dataset = GraphDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    all_true_cin_feats, all_pred_cin_feats = [], []
    all_true_op_types, all_pred_op_types_logits = [], []

    print("\nEvaluating model on the test set...")
    start_eval_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Batches"):
            if batch is None: continue
            batch = batch.to(device)
            node_embeddings = model(batch)

            cin_mask = batch.y_role == 0
            if torch.any(cin_mask):
                pred_cin_f = model.cin_head(node_embeddings[cin_mask])
                true_cin_f = batch.y_target_cin_features[cin_mask]
                if pred_cin_f.numel() > 0: # Ensure prediction is not empty
                    all_pred_cin_feats.append(pred_cin_f.cpu())
                    all_true_cin_feats.append(true_cin_f.cpu())

            op_mask = batch.y_role == 1
            if torch.any(op_mask):
                pred_op_l = model.op_head(node_embeddings[op_mask])
                true_op_t = batch.y_target_op_types[op_mask]
                valid_op_targets = true_op_t != -1 # Filter out placeholder -1 labels
                if torch.any(valid_op_targets) and pred_op_l[valid_op_targets].numel() > 0:
                    all_pred_op_types_logits.append(pred_op_l[valid_op_targets].cpu())
                    all_true_op_types.append(true_op_t[valid_op_targets].cpu())
    end_eval_time = time.time()
    print(f"Evaluation prediction loop took {end_eval_time - start_eval_time:.2f} seconds.")

    if not all_true_cin_feats and not all_true_op_types:
        print("No valid targets found across the entire test set for evaluation.", file=sys.stderr); return False

    # --- Process C_in Feature Regression Results ---
    test_mae_cin = float('nan')
    if all_true_cin_feats:
        y_true_cin_all = torch.cat(all_true_cin_feats, dim=0).numpy()
        y_pred_cin_all = torch.cat(all_pred_cin_feats, dim=0).numpy()
        if y_true_cin_all.shape == y_pred_cin_all.shape and y_true_cin_all.size > 0:
             test_mae_cin = mean_absolute_error(y_true_cin_all, y_pred_cin_all)
             print(f"\nC_in Feature Regression MAE: {test_mae_cin:.4f}")
        else: print("\nC_in regression: True and Pred shapes mismatch or empty.")
    else: print("\nNo C_in nodes with valid targets found for MAE calculation.")

    # --- Process Operation Type Classification Results ---
    if all_true_op_types:
        y_true_op_cat_all = torch.cat(all_true_op_types, dim=0).numpy()
        y_pred_op_logits_cat_all = torch.cat(all_pred_op_types_logits, dim=0)
        y_pred_op_cat_all = y_pred_op_logits_cat_all.argmax(dim=1).numpy()

        if len(y_true_op_cat_all) > 0:
            test_acc_op = accuracy_score(y_true_op_cat_all, y_pred_op_cat_all)
            print(f"\nOperation Type Classification Accuracy: {test_acc_op:.4f}")

            # Ensure labels for report are within bounds of target_names
            report_labels = list(range(len(class_names_for_report)))
            op_report = classification_report(y_true_op_cat_all, y_pred_op_cat_all,
                                              labels=report_labels,
                                              target_names=class_names_for_report,
                                              digits=4, zero_division=0)
            print("Operation Type Classification Report:\n", op_report)

            print("\nOperation Type Top-K Accuracy:")
            for k_val in [1, 3, 5, 10]:
                if k_val > y_pred_op_logits_cat_all.shape[1]: continue
                _, top_k_preds = y_pred_op_logits_cat_all.topk(k_val, dim=1)
                # Unsqueeze y_true_op_cat_all to enable broadcasting for comparison
                correct_top_k = torch.sum(top_k_preds == torch.from_numpy(y_true_op_cat_all).unsqueeze(1)).item()
                top_k_acc = correct_top_k / len(y_true_op_cat_all)
                print(f"  Top-{k_val} Accuracy: {top_k_acc:.4f}")

            try:
                cm = confusion_matrix(y_true_op_cat_all, y_pred_op_cat_all, labels=report_labels)
                plt.figure(figsize=(max(8, len(class_names_for_report)), max(6, len(class_names_for_report)*0.8)))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names_for_report, yticklabels=class_names_for_report)
                plt.xlabel('Predicted Label'); plt.ylabel('True Label')
                plt.title('Op Type Confusion Matrix'); plt.tight_layout()
                cm_path = os.path.join(PLOTS_DIR, "op_type_confusion_matrix.png")
                plt.savefig(cm_path); print(f"Saved Op Type CM to {cm_path}"); plt.close()
            except Exception as e: print(f"Warning: Could not plot confusion matrix: {e}")
        else: print("No valid operation type predictions to report on.")
    else: print("\nNo Operation nodes with valid targets found for classification metrics.")

    history_path = os.path.join(MODEL_DIR, 'training_history_gatv2.json')
    if os.path.exists(history_path):
        print("\nPlotting training history...")
        try:
            with open(history_path, 'r') as f: history = json.load(f)
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            if epochs:
                fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                axs[0].plot(epochs, history.get('train_loss', [float('nan')]*len(epochs)), 'b-', label='Train Loss')
                axs[0].plot(epochs, history.get('val_loss', [float('nan')]*len(epochs)), 'r-', label='Val Loss')
                axs[0].set_ylabel('Combined Loss'); axs[0].legend(); axs[0].grid(True); axs[0].set_title('Training & Validation Loss')
                axs[1].plot(epochs, history.get('train_mae_cin', [float('nan')]*len(epochs)), 'b-o', label='Train C_in MAE', markersize=3)
                axs[1].plot(epochs, history.get('val_mae_cin', [float('nan')]*len(epochs)), 'r-o', label='Val C_in MAE', markersize=3)
                axs[1].set_ylabel('C_in Feat. MAE'); axs[1].legend(); axs[1].grid(True); axs[1].set_title('C_in Feature Regression MAE')
                axs[2].plot(epochs, history.get('train_acc_op', [float('nan')]*len(epochs)), 'b-s', label='Train Op Acc.', markersize=3)
                axs[2].plot(epochs, history.get('val_acc_op', [float('nan')]*len(epochs)), 'r-s', label='Val Op Acc.', markersize=3)
                axs[2].set_xlabel('Epochs'); axs[2].set_ylabel('Op Type Accuracy'); axs[2].legend(); axs[2].grid(True); axs[2].set_title('Operation Type Classification Accuracy')
                plt.tight_layout(); hist_plot_path = os.path.join(PLOTS_DIR, "training_curves_gatv2.png")
                plt.savefig(hist_plot_path); print(f"Saved training curves to {hist_plot_path}"); plt.close(fig)
            else: print("  No epoch data in history to plot.")
        except Exception as e: print(f"Warning: Could not plot training history: {e}")
    else: print("Training history file not found.")

    # --- Qualitative Comparison ---
    if test_files and ActionGraph != object: # Check if ActionGraph was properly imported
        print("\n--- Example Qualitative Prediction (First Test Sample) ---")
        try:
            example_pyg_path = test_files[0]
            example_ag_filename = os.path.basename(example_pyg_path).replace('.pt', '.json')
            example_ag_path = os.path.join(ACTION_GRAPH_RAW_DIR, example_ag_filename)
            print(f"Attempting to load original AG JSON from: {example_ag_path}")

            if os.path.exists(example_ag_path):
                with open(example_ag_path, 'r') as f: ag_dict = json.load(f)
                ag_example = ActionGraph.deserialize(ag_dict)
                data_example = torch.load(example_pyg_path).to(device)
                with torch.no_grad(): node_embeds = model(data_example)

                print(f"Example Graph: {example_ag_filename}")
                print("Node Roles (True): ", data_example.y_role.cpu().numpy().tolist())

                if torch.any(data_example.y_role == 0): # If C_in nodes exist
                    pred_cin_feats_ex = model.cin_head(node_embeds[data_example.y_role == 0])
                    print("\nPredicted C_in Features (sample from first C_in node):")
                    print("  Pred:", pred_cin_feats_ex[0, :5].cpu().numpy().round(3).tolist(), "...")
                    print("  True:", data_example.y_target_cin_features[data_example.y_role == 0][0, :5].cpu().numpy().round(3).tolist(), "...")

                if torch.any(data_example.y_role == 1): # If O nodes exist
                    pred_op_logits_ex = model.op_head(node_embeds[data_example.y_role == 1])
                    pred_op_types_ex_indices = pred_op_logits_ex.argmax(dim=1).cpu().numpy()
                    true_op_types_ex_indices = data_example.y_target_op_types[data_example.y_role == 1].cpu().numpy()
                    op_node_graph_indices = (data_example.y_role == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    print("\nPredicted Operation Types (for O nodes):")
                    for i, op_node_graph_idx in enumerate(op_node_graph_indices):
                        node_id_in_ag = list(ag_example.nodes())[op_node_graph_idx]
                        pred_type_idx = pred_op_types_ex_indices[i] if i < len(pred_op_types_ex_indices) else -1
                        pred_type_str = class_names_for_report[pred_type_idx] if 0 <= pred_type_idx < len(class_names_for_report) else "N/A"
                        true_type_idx = true_op_types_ex_indices[i]
                        true_type_str = class_names_for_report[true_type_idx] if 0 <= true_type_idx < len(class_names_for_report) else "N/A (or invalid)"
                        print(f"  AG Node ID '{node_id_in_ag}': Pred Type = {pred_type_str} (idx {pred_type_idx}), True Type = {true_type_str} (idx {true_type_idx})")
            else: print(f"Original ActionGraph JSON not found for example: {example_ag_path}")
        except Exception as e: print(f"Error during qualitative example: {e}", file=sys.stderr)
    elif ActionGraph == object: print("Skipping qualitative example: ActionGraph class not imported.")


    print("\nEvaluation step completed successfully.")
    return True

if __name__ == "__main__":
    if not run_ag_evaluate_gatv2():
        sys.exit(1)
