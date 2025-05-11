import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv # removed global_mean_pool as it wasn't used
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from serialization_utils import safe_load, safe_save
import pandas as pd
import numpy as np

class PrecursorPredictionGNN(nn.Module):
    """GNN model for predicting precursors given a target material."""
    def __init__(self, node_features, hidden_channels, num_layers, dropout=0.1):
        super(PrecursorPredictionGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        # Ensure input layer matches the actual node feature dimension
        self.embedding = nn.Linear(node_features, hidden_channels)
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_channels, out_channels=hidden_channels,
                heads=4, concat=False, dropout=dropout # Changed heads to 4 to match original
            ) for _ in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x_res = x
            x = self.conv_layers[i](x, edge_index)
            x = x_res + x # Residual connection
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        scores = self.predictor(x).squeeze(-1)

        # Clamp logits for numerical stability with BCEWithLogitsLoss
        clamped_scores = torch.clamp(scores, min=-15.0, max=15.0)
        return clamped_scores


def train_model():
    """Train the GNN model for precursor prediction with train/val/test split."""
    # --- Configuration ---
    data_path = "Data/ag-data/featurized/all_data.pt"
    model_dir = "Data/models/"
    loss_file = os.path.join(model_dir, "loss-vs-epochs.csv") # Save loss in model dir
    os.makedirs(model_dir, exist_ok=True)
    # --- Hyperparameters --- (Match these with evaluate_model.py if hardcoded there)
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.0001
    weight_decay = 1e-5
    hidden_channels = 32 # Make sure this matches evaluate_model.py
    num_layers = 3       # Make sure this matches evaluate_model.py
    dropout = 0.3
    # --- Split Ratios ---
    validation_split_ratio = 0.15
    test_split_ratio = 0.15
    random_state = 42
    use_weighted_loss = True
    gradient_clip_norm = 1.0
    patience = 15

    # --- Data Loading & Validation ---
    print("Loading and validating data...")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    try:
        # Load the dictionary containing config and data
        loaded_obj = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False) # Explicitly set weights_only=False        all_data = loaded_obj['data']
        config = loaded_obj.get('config', {}) # Get config if it exists
        node_feature_dim = config.get('max_feature_len') # Get feature dimension from config

        if node_feature_dim is None:
             # Fallback: Infer from first data object if config is missing
             if all_data and hasattr(all_data[0], 'x') and all_data[0].x is not None:
                 node_feature_dim = all_data[0].x.shape[1]
                 print(f"Warning: Feature dimension not found in config, inferred as {node_feature_dim}")
             else:
                 print("Error: Cannot determine node feature dimension from data file.")
                 return
        else:
             print(f"Node feature dimension loaded from config: {node_feature_dim}")

        print(f"Successfully loaded {len(all_data)} graphs.")
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        import traceback
        traceback.print_exc()
        return

    if not all_data:
        print("Error: No graphs loaded from the data file.")
        return

    # --- Data Validation Loop ---
    valid_data = []
    total_nodes = 0
    total_target_nodes = 0 # Renamed from total_input_nodes for clarity
    print("Validating loaded graphs...")
    skipped_mismatch = 0
    skipped_zero_nodes = 0
    skipped_no_targets = 0
    skipped_no_outputs = 0
    for i, d in enumerate(all_data):
        valid = True
        if not hasattr(d, 'x') or not hasattr(d, 'edge_index') or not hasattr(d, 'y') or not hasattr(d, 'is_output'):
             valid = False; skipped_mismatch+=1
        elif d.x is None or d.edge_index is None or d.y is None or d.is_output is None:
             valid = False; skipped_mismatch+=1
        elif d.x.shape[0] != d.y.shape[0] or d.x.shape[0] != d.is_output.shape[0]:
             valid = False; skipped_mismatch+=1
        elif d.num_nodes == 0:
             valid = False; skipped_zero_nodes+=1

        if valid:
             # Check feature dimension consistency
             if d.x.shape[1] != node_feature_dim:
                 print(f"Warning: Graph {i} has inconsistent feature dimension ({d.x.shape[1]} vs {node_feature_dim}). Skipping.")
                 valid = False; skipped_mismatch+=1

        if valid:
            num_targets = d.y.sum().item()
            num_outputs = d.is_output.sum().item()
            # Require at least one target (precursor) and one output node for training
            if num_targets > 0 and num_outputs > 0:
                 valid_data.append(d)
                 total_nodes += d.num_nodes
                 total_target_nodes += num_targets
            else:
                 if num_targets == 0: skipped_no_targets += 1
                 if num_outputs == 0: skipped_no_outputs += 1 # May overlap with no targets
                 valid = False # Skip graphs missing targets or outputs for training

    print(f"Validation Summary:")
    print(f"  Input Graphs: {len(all_data)}")
    print(f"  Valid Graphs for Training/Splitting: {len(valid_data)}")
    print(f"  Skipped (Attribute/Shape Mismatch): {skipped_mismatch}")
    print(f"  Skipped (Zero Nodes): {skipped_zero_nodes}")
    print(f"  Skipped (No Target Nodes 'y'=1): {skipped_no_targets}")
    print(f"  Skipped (No Output Nodes 'is_output'=1): {skipped_no_outputs}")


    if not valid_data:
        print("Error: No valid graphs remaining after validation. Cannot train.")
        return
    print(f"Using {len(valid_data)} valid graphs with avg {total_nodes/len(valid_data):.2f} nodes and avg {total_target_nodes/len(valid_data):.2f} target nodes.")
    # --------------------------------------------------

    # --- Train/Validation/Test Split ---
    # Ensure enough data for splitting
    if len(valid_data) < 3: # Need at least one sample for train, val, test potentially
        print("Error: Not enough valid data points for train/validation/test split.")
        return
    try:
        train_val_data, test_data = train_test_split(
            valid_data, test_size=test_split_ratio, random_state=random_state, stratify=[d.y.sum().item() > 0 for d in valid_data] if len(set(d.y.sum().item() for d in valid_data)) > 1 else None # Stratify if possible
        )
        # Adjust val ratio for the remaining data
        if len(train_val_data) < 2:
             print("Warning: Not enough data for separate train/val split after test split. Using all remaining for training.")
             train_data, val_data = train_val_data, []
        else:
             relative_val_ratio = validation_split_ratio / (1.0 - test_split_ratio)
             train_data, val_data = train_test_split(
                 train_val_data, test_size=relative_val_ratio, random_state=random_state, stratify=[d.y.sum().item() > 0 for d in train_val_data] if len(set(d.y.sum().item() for d in train_val_data)) > 1 else None
             )
        print(f"Splitting into: {len(train_data)} Train, {len(val_data)} Validation, {len(test_data)} Test graphs")
        if not train_data:
             print("Error: No training data after split.")
             return
    except ValueError as e:
         print(f"Error during splitting (potentially too few samples for stratification?): {e}")
         # Fallback to non-stratified split
         train_val_data, test_data = train_test_split(valid_data, test_size=test_split_ratio, random_state=random_state)
         relative_val_ratio = validation_split_ratio / (1.0 - test_split_ratio)
         if len(train_val_data) > 1:
             train_data, val_data = train_test_split(train_val_data, test_size=relative_val_ratio, random_state=random_state)
         else:
              train_data, val_data = train_val_data, [] # Assign all to train if only one left
         print(f"Splitting (fallback): {len(train_data)} Train, {len(val_data)} Validation, {len(test_data)} Test graphs")
         if not train_data:
              print("Error: No training data after split.")
              return


    # --- Data Loaders ---
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # Only create val_loader if val_data exists
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False) if test_data else None
    # ------------------

    # --- Model Initialization ---
    print(f"\nInitializing model with {node_feature_dim} node features...")
    model = PrecursorPredictionGNN(
        node_features=node_feature_dim, hidden_channels=hidden_channels,
        num_layers=num_layers, dropout=dropout
    )
    # -----------------------------------------

    # --- Optimizer, Device, Loss ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Calculate pos_weight based ONLY on training data
    pos_weight = None
    if use_weighted_loss:
        train_nodes = sum(d.num_nodes for d in train_data)
        train_target_nodes = sum(d.y.sum().item() for d in train_data)
        print(f"Training set: {train_nodes} total nodes, {train_target_nodes} target nodes.")
        if train_nodes > 0 and train_target_nodes > 0 and train_target_nodes < train_nodes:
            # Calculate weight for the positive class (targets)
            pos_weight_val = (train_nodes - train_target_nodes) / train_target_nodes
            # Cap the weight to prevent extreme values
            pos_weight_val = min(pos_weight_val, 100.0)
            pos_weight = torch.tensor([pos_weight_val], device=device)
            print(f"Using weighted BCE loss (pos_weight={pos_weight.item():.4f})")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print("Weighted loss requested but prerequisites not met (e.g., no targets or all nodes are targets). Using standard BCE loss.")
            criterion = nn.BCEWithLogitsLoss()
            use_weighted_loss = False # Ensure flag matches reality
    else:
         print("Using standard BCE loss.")
         criterion = nn.BCEWithLogitsLoss()
    # --------------------------------------------------------------------------------

    # --- Training Loop with Validation and Early Stopping ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    loss_history = []

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        processed_batches = 0
        for batch in train_loader:
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                # Ensure batch has nodes before proceeding
                if batch.num_nodes == 0: continue

                scores = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.float() # Ensure target is float

                # Check for shape mismatch before calculating loss
                if scores.shape != target.shape:
                     print(f"Warning: Shape mismatch in training batch! Scores: {scores.shape}, Target: {target.shape}. Skipping batch.")
                     continue

                # Check for NaN/Inf in scores before loss calculation
                if not torch.isfinite(scores).all():
                    print(f"Warning: NaN/Inf detected in model scores during training epoch {epoch+1}. Skipping batch.")
                    continue

                loss = criterion(scores, target)

                # Check for NaN/Inf in loss
                if not torch.isfinite(loss):
                     print(f"Warning: NaN/Inf detected in loss during training epoch {epoch+1}. Skipping batch.")
                     continue

                loss.backward()
                if gradient_clip_norm is not None:
                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                total_train_loss += loss.item()
                processed_batches += 1
            except Exception as e:
                print(f"Error processing training batch: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                continue # Skip batch on error

        train_loss = total_train_loss / processed_batches if processed_batches > 0 else 0.0
        # --------------------

        # --- Validation Phase ---
        val_loss = float('nan') # Default if no validation
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            processed_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        batch = batch.to(device)
                        if batch.num_nodes == 0: continue # Skip empty batches

                        scores = model(batch.x, batch.edge_index, batch.batch)
                        target = batch.y.float()

                        if scores.shape != target.shape:
                            print(f"Warning: Shape mismatch in validation batch! Scores: {scores.shape}, Target: {target.shape}. Skipping batch.")
                            continue

                        # No need to check scores for NaN here typically, but can add if needed
                        # Use the same criterion for validation loss calculation
                        loss = criterion(scores, target)

                        if not torch.isfinite(loss):
                             print(f"Warning: NaN/Inf detected in validation loss during epoch {epoch+1}. Skipping batch.")
                             continue

                        total_val_loss += loss.item()
                        processed_val_batches += 1
                    except Exception as e:
                        print(f"Error processing validation batch: {e}")
                        continue # Skip batch on error
            val_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else 0.0
        # --------------------

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        # Store Val loss (use 'Val Loss' key for clarity)
        loss_history.append({'Epoch': epoch + 1, 'Train Loss': train_loss, 'Val Loss': val_loss})

        # --- Checkpoint Best Model based on Validation Loss (only if val_loader exists) ---
        if val_loader:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(model_dir, "best_model.pt")
                print(f"  New best validation loss: {best_val_loss:.6f}. Saving model to {best_model_path}")
                # Save model state and also the config used for this model
                model_save_obj = {
                    'state_dict': model.state_dict(),
                    'config': { # Store key hyperparams used for this model
                        'node_features': node_feature_dim,
                        'hidden_channels': hidden_channels,
                        'num_layers': num_layers,
                        'dropout': dropout # Save dropout used during training
                    }
                }
                safe_save(model_save_obj, best_model_path)
                epochs_no_improve = 0 # Reset counter
            else:
                epochs_no_improve += 1

            # --- Early Stopping ---
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation loss improvement.")
                break
        # -------------------- (End Checkpoint/Early Stopping block)

    # --- Save Final Model (regardless of validation performance) ---
    final_model_path = os.path.join(model_dir, "final_model.pt")
    print(f"\nSaving final model state to {final_model_path}")
    final_model_save_obj = {
        'state_dict': model.state_dict(),
         'config': { # Include config here too
            'node_features': node_feature_dim,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'dropout': dropout
        }
    }
    safe_save(final_model_save_obj, final_model_path)

    # --- Save Test Data (if it exists) ---
    if test_data:
        test_data_path = os.path.join(model_dir, "test_data.pt")
        print(f"Saving test data ({len(test_data)} graphs) to {test_data_path}")
        # Save test data along with the config used during its creation/training run
        test_save_obj = {'config': {'max_feature_len': node_feature_dim}, 'data': test_data}
        torch.save(test_save_obj, test_data_path)
    else:
         print("No test data generated, skipping save.")
    # --------------------------------------

    # --- Save Loss History ---
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(loss_file, index=False)
    print(f"Loss history (Train/Val) saved to {loss_file}")
    # ------------------------

    print("\nTraining completed.")

if __name__ == "__main__":
    train_model()
