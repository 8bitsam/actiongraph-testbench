import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from serialization_utils import safe_load, safe_save # Keep for model saving
import pandas as pd # For saving loss history
import numpy as np # For weight calculation

# --- GNN Model Definition (keep as before) ---
class PrecursorPredictionGNN(nn.Module):
    """GNN model for predicting precursors given a target material."""
    def __init__(self, node_features, hidden_channels, num_layers, dropout=0.1):
        super(PrecursorPredictionGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Linear(node_features, hidden_channels)
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_channels, out_channels=hidden_channels,
                heads=4, concat=False, dropout=dropout
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
            x = x_res + x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        scores = self.predictor(x).squeeze(-1)

        # --- Add Logit Clamping ---
        # Clamp logits to a reasonable range, e.g., [-15, 15]
        # sigmoid(-15) is very close to 0, sigmoid(15) is very close to 1
        # Adjust the range if needed based on experimentation
        clamped_scores = torch.clamp(scores, min=-15.0, max=15.0)
        # --------------------------

        # return scores # Return original scores if needed elsewhere?
        return clamped_scores # Return clamped scores for loss calculation
# ---------------------------------------------

def train_model():
    """Train the GNN model for precursor prediction with train/val/test split."""
    # --- Configuration ---
    data_path = "Data/ag-data/featurized/all_data.pt"
    model_dir = "Data/models/"
    loss_file = "loss-vs-epochs.csv"
    os.makedirs(model_dir, exist_ok=True)
    batch_size = 32
    num_epochs = 100 # Increase epochs?
    learning_rate = 0.0001 # Keep reduced LR
    weight_decay = 1e-5
    hidden_channels = 32
    num_layers = 3
    dropout = 0.3 # Slightly increased dropout
    # --- Split Ratios ---
    validation_split_ratio = 0.15 # e.g., 15% for validation
    test_split_ratio = 0.15      # e.g., 15% for test (adjust as needed, train = 1 - val - test)
    # --------------------
    random_state = 42
    use_weighted_loss = True
    gradient_clip_norm = 1.0
    patience = 15 # Early stopping patience

    # --- Data Loading & Validation (keep as before) ---
    print("Loading and validating data...")
    if not os.path.exists(data_path): # ... (error handling) ...
        return
    try:
        all_data = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False)
        print(f"Successfully loaded {len(all_data)} featurized graphs.")
    except Exception as e: # ... (error handling) ...
        return
    if not all_data: # ... (error handling) ...
        return

    valid_data = []
    total_nodes = 0
    total_input_nodes = 0
    # ... (keep the validation loop from previous version) ...
    for i, d in enumerate(all_data):
        valid = True
        # ... (attribute checks) ...
        if valid:
             if d.x.shape[0] != d.y.shape[0] or d.x.shape[0] != d.is_output.shape[0]: valid = False
             elif d.num_nodes == 0: valid = False
        if valid:
            num_inputs = d.y.sum().item()
            num_outputs = d.is_output.sum().item()
            if num_inputs > 0 and num_outputs > 0:
                 valid_data.append(d)
                 total_nodes += d.num_nodes
                 total_input_nodes += num_inputs
    # ... (print stats) ...
    if not valid_data: # ... (error handling) ...
        return
    print(f"Using {len(valid_data)} valid graphs for splitting.")
    # --------------------------------------------------

    # --- Train/Validation/Test Split ---
    train_val_data, test_data = train_test_split(
        valid_data, test_size=test_split_ratio, random_state=random_state
    )
    # Adjust val ratio for the remaining data
    relative_val_ratio = validation_split_ratio / (1.0 - test_split_ratio)
    train_data, val_data = train_test_split(
        train_val_data, test_size=relative_val_ratio, random_state=random_state # Use same random state
    )
    print(f"Splitting into: {len(train_data)} Train, {len(val_data)} Validation, {len(test_data)} Test graphs")
    # ----------------------------------

    # --- Data Loaders ---
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) # No shuffle for val/test
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # ------------------

    # --- Model Initialization (keep as before) ---
    node_feature_dim = valid_data[0].x.shape[1]
    model = PrecursorPredictionGNN(
        node_features=node_feature_dim, hidden_channels=hidden_channels,
        num_layers=num_layers, dropout=dropout
    )
    # -----------------------------------------

    # --- Optimizer, Device, Loss (keep weighted loss calculation based on train_data) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if use_weighted_loss:
        train_nodes = sum(d.num_nodes for d in train_data)
        train_input_nodes = sum(d.y.sum().item() for d in train_data)
        if train_nodes > 0 and train_input_nodes > 0 and train_input_nodes < train_nodes:
            pos_weight_val = min((train_nodes - train_input_nodes) / train_input_nodes, 100.0) # Keep cap
            pos_weight = torch.tensor([pos_weight_val], device=device)
            print(f"Using weighted BCE loss (pos_weight={pos_weight.item():.4f})")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            print("Weighted loss requested but invalid calculation; using standard BCE.")
            criterion = nn.BCEWithLogitsLoss()
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
                scores = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.float()
                if scores.shape != target.shape: continue # Skip bad batches

                loss = criterion(scores, target)
                loss.backward()
                if gradient_clip_norm is not None:
                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()
                total_train_loss += loss.item()
                processed_batches += 1
            except Exception as e:
                print(f"Error processing training batch: {e}")
                continue
        train_loss = total_train_loss / processed_batches if processed_batches > 0 else 0.0
        # --------------------

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        processed_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(device)
                    scores = model(batch.x, batch.edge_index, batch.batch)
                    target = batch.y.float()
                    if scores.shape != target.shape: continue

                    loss = criterion(scores, target) # Use same criterion
                    total_val_loss += loss.item()
                    processed_val_batches += 1
                except Exception as e:
                    print(f"Error processing validation batch: {e}")
                    continue
        val_loss = total_val_loss / processed_val_batches if processed_val_batches > 0 else 0.0
        # --------------------

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        loss_history.append({'Epoch': epoch + 1, 'Train Loss': train_loss, 'Test Loss': val_loss}) # Store Val loss as 'Test Loss' for plotting script

        # --- Checkpoint Best Model based on Validation Loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_dir, "best_model.pt")
            print(f"  New best validation loss: {best_val_loss:.6f}. Saving model.")
            safe_save(model.state_dict(), best_model_path)
            epochs_no_improve = 0 # Reset counter
        else:
            epochs_no_improve += 1

        # --- Early Stopping ---
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without validation loss improvement.")
            break
        # --------------------

    # --- Save Final Model and Test Data ---
    final_model_path = os.path.join(model_dir, "final_model.pt")
    safe_save(model.state_dict(), final_model_path) # Save the last epoch model too

    test_data_path = os.path.join(model_dir, "test_data.pt")
    print(f"\nSaving test data ({len(test_data)} graphs) to {test_data_path}")
    torch.save(test_data, test_data_path)
    # --------------------------------------

    # --- Save Loss History ---
    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(loss_file, index=False)
    print(f"Loss history (Train/Val) saved to {loss_file}")
    # ------------------------

    print("\nTraining completed.")

if __name__ == "__main__":
    train_model()
