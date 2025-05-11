# ag_train_gatv2.py
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data # Explicitly import Data for type hinting
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import mean_absolute_error # Used for F.l1_loss reference
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# import matplotlib.pyplot as plt # Not directly used for plotting within train script

# --- Configuration ---
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
FEATURIZED_AG_DIR = os.path.join(DATA_DIR, "ag-data-featurized-gatv2/")
TRAIN_DATA_DIR = os.path.join(FEATURIZED_AG_DIR, "training-data")
MODEL_DIR = os.path.join(DATA_DIR, "models/gat-model-v2/")

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 100
BATCH_SIZE = 32
HIDDEN_CHANNELS = 64 # Number of hidden units in GAT layers and MLP heads
GAT_LAYERS = 2 # Number of GAT layers
NUM_HEADS = 4 # Number of attention heads
DROPOUT_RATE = 0.3
# Loss weights
ALPHA_CIN_REG = 1.0  # Weight for C_in feature regression loss
ALPHA_OP_CLS = 1.0   # Weight for O type classification loss
VAL_SPLIT_SIZE_FROM_TRAIN = 0.15 # 15% of training files for validation
RANDOM_STATE = 42
EARLY_STOPPING_PATIENCE = 10

# --- GAT Model with Prediction Heads ---
class GATInversePredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels,
                 num_op_type_classes, cin_target_feature_dim,
                 gat_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout_val = dropout
        self.gat_layers_list = nn.ModuleList() # Renamed for clarity
        current_dim = in_channels

        for i in range(gat_layers):
            # Each GAT layer will output hidden_channels in total (after head concatenation)
            # So, each head within a GAT layer outputs hidden_channels / heads
            conv = GATv2Conv(current_dim, hidden_channels // heads,
                             heads=heads, concat=True, dropout=dropout, add_self_loops=True)
            self.gat_layers_list.append(conv)
            current_dim = hidden_channels # Output of GAT layer (heads * (hidden_channels // heads))

        # Prediction head for C_in (precursor) feature regression
        # Input to this head is the output of the last GAT layer (current_dim = hidden_channels)
        self.cin_head = nn.Sequential(
            nn.Linear(current_dim, hidden_channels), # Can keep or reduce intermediate dim
            nn.ReLU(),
            nn.Dropout(dropout), # Apply dropout in MLP head as well
            nn.Linear(hidden_channels, cin_target_feature_dim)
        )

        # Prediction head for O (operation) type classification
        self.op_head = nn.Sequential(
            nn.Linear(current_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_op_type_classes)
        )

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.gat_layers_list):
            # Apply dropout before each GAT layer's message passing
            x = F.dropout(x, p=self.dropout_val, training=self.training)
            x = layer(x, edge_index)
            # Apply activation after each GAT layer
            x = F.elu(x)
        # The final 'x' is the node embeddings after all GAT layers
        return x


# --- Training Function ---
def run_ag_train_gatv2():
    print("--- Running GATv2 Model Training for Masked Prediction ---")

    if not os.path.isdir(TRAIN_DATA_DIR):
        print(f"Error: Training data dir not found: {TRAIN_DATA_DIR}", file=sys.stderr)
        return False

    print("Loading metadata...")
    try:
        with open(os.path.join(FEATURIZED_AG_DIR, 'metadata_gatv2.json'), 'r') as f:
            metadata = json.load(f)
        gat_input_dim = metadata['gat_input_feature_dim']
        num_op_type_classes = metadata['num_op_type_classes']
        cin_target_feature_dim = metadata['chem_feature_dim']
        print(f"  GAT Input Dim: {gat_input_dim}, OpType Classes: {num_op_type_classes}, C_in Target Dim: {cin_target_feature_dim}")
    except Exception as e:
        print(f"Error loading metadata: {e}", file=sys.stderr); return False

    print("Loading training graph file paths...")
    try:
        all_train_val_files = [os.path.join(TRAIN_DATA_DIR, f)
                           for f in os.listdir(TRAIN_DATA_DIR) if f.endswith('.pt')]
        if not all_train_val_files:
             print("Error: No training '.pt' files found in TRAIN_DATA_DIR.", file=sys.stderr); return False

        train_files, val_files = train_test_split(
            all_train_val_files, test_size=VAL_SPLIT_SIZE_FROM_TRAIN, random_state=RANDOM_STATE
        )
        print(f"  Actual Train set size: {len(train_files)}")
        print(f"  Validation set size: {len(val_files)}")

        # Save the test file list for the evaluation script
        # This should ideally be done when splitting the *entire* dataset,
        # not just the training portion. Assuming test_files.json is created by featurizer.
        # If not, and TRAIN_DATA_DIR contains all non-test files:
        if not os.path.exists(os.path.join(MODEL_DIR, 'test_files.json')):
             print("Warning: test_files.json not found in model directory. Evaluation might use different test set if not managed externally.", file=sys.stderr)

    except Exception as e:
        print(f"Error listing/splitting training files: {e}", file=sys.stderr); return False

    class GraphDataset(torch.utils.data.Dataset):
        def __init__(self, file_list): self.file_list = file_list
        def __len__(self): return len(self.file_list)
        def __getitem__(self, idx):
            try: return torch.load(self.file_list[idx], weights_only=False)
            except Exception as e:
                 # print(f"Warning: Error loading graph {self.file_list[idx]}: {e}", file=sys.stderr) # Can be very verbose
                 return None
    def collate_fn(batch):
        batch = [data for data in batch if data is not None] # Filter out None from failed loads
        return DataLoader.collate(batch) if batch else None # Return None if entire batch failed

    train_dataset = GraphDataset(train_files)
    val_dataset = GraphDataset(val_files)
    # Consider num_workers > 0 if data loading is a bottleneck and not causing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = GATInversePredictor(
        in_channels=gat_input_dim, hidden_channels=HIDDEN_CHANNELS,
        num_op_type_classes=num_op_type_classes,
        cin_target_feature_dim=cin_target_feature_dim,
        gat_layers=GAT_LAYERS, heads=NUM_HEADS, dropout=DROPOUT_RATE
    ).to(device)
    print("\nModel Architecture:"); print(model)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    cin_criterion = nn.MSELoss()
    op_criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'val_loss': [], 'train_mae_cin': [], 'val_mae_cin': [], 'train_acc_op': [], 'val_acc_op': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    actual_epochs_run = 0 # To store the number of epochs actually run

    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        actual_epochs_run = epoch
        start_epoch_time = time.time()
        model.train()
        epoch_train_loss_sum, total_cin_nodes_train, epoch_train_cin_mae_sum, epoch_train_op_correct, epoch_train_op_total = 0, 0, 0, 0, 0
        num_train_batches_with_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            if batch is None: continue # Skip if collate_fn returned None (all graphs in batch failed load)
            batch = batch.to(device)
            optimizer.zero_grad()
            node_embeddings = model(batch) # Get embeddings for all nodes

            loss_this_batch = torch.tensor(0.0, device=device) # Initialize loss for this batch
            has_loss_component = False

            # C_in Feature Regression Loss
            cin_mask = batch.y_role == 0
            if torch.any(cin_mask):
                pred_cin_feats = model.cin_head(node_embeddings[cin_mask])
                true_cin_feats = batch.y_target_cin_features[cin_mask]
                if pred_cin_feats.numel() > 0 and true_cin_feats.numel() > 0: # Ensure tensors are not empty
                    loss_cin = cin_criterion(pred_cin_feats, true_cin_feats)
                    loss_this_batch += ALPHA_CIN_REG * loss_cin
                    epoch_train_cin_mae_sum += F.l1_loss(pred_cin_feats, true_cin_feats, reduction='sum').item()
                    total_cin_nodes_train += cin_mask.sum().item()
                    has_loss_component = True

            # Operation Type Classification Loss
            op_mask = batch.y_role == 1
            if torch.any(op_mask):
                pred_op_logits = model.op_head(node_embeddings[op_mask])
                true_op_types = batch.y_target_op_types[op_mask]
                valid_op_target_mask = true_op_types != -1
                if torch.any(valid_op_target_mask) and pred_op_logits[valid_op_target_mask].numel() > 0:
                    loss_op = op_criterion(pred_op_logits[valid_op_target_mask], true_op_types[valid_op_target_mask])
                    loss_this_batch += ALPHA_OP_CLS * loss_op
                    preds_op = pred_op_logits[valid_op_target_mask].argmax(dim=1)
                    epoch_train_op_correct += (preds_op == true_op_types[valid_op_target_mask]).sum().item()
                    epoch_train_op_total += valid_op_target_mask.sum().item()
                    has_loss_component = True
            
            if has_loss_component: # Only backward/step if a loss was computed
                loss_this_batch.backward()
                optimizer.step()
                epoch_train_loss_sum += loss_this_batch.item()
                num_train_batches_with_loss += 1
        
        avg_train_loss = epoch_train_loss_sum / num_train_batches_with_loss if num_train_batches_with_loss > 0 else float('nan')
        avg_train_mae_cin = epoch_train_cin_mae_sum / total_cin_nodes_train if total_cin_nodes_train > 0 else float('nan')
        avg_train_acc_op = epoch_train_op_correct / epoch_train_op_total if epoch_train_op_total > 0 else float('nan')
        history['train_loss'].append(avg_train_loss)
        history['train_mae_cin'].append(avg_train_mae_cin)
        history['train_acc_op'].append(avg_train_acc_op)

        # Validation
        model.eval()
        epoch_val_loss_sum, total_cin_nodes_val, epoch_val_cin_mae_sum, epoch_val_op_correct, epoch_val_op_total = 0, 0, 0, 0, 0
        num_val_batches_with_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]  "):
                if batch is None: continue
                batch = batch.to(device)
                node_embeddings = model(batch)
                val_loss_this_batch = torch.tensor(0.0, device=device)
                has_val_loss_component = False

                cin_mask_val = batch.y_role == 0
                if torch.any(cin_mask_val):
                    pred_cin_feats_val = model.cin_head(node_embeddings[cin_mask_val])
                    true_cin_feats_val = batch.y_target_cin_features[cin_mask_val]
                    if pred_cin_feats_val.numel() > 0 and true_cin_feats_val.numel() > 0:
                        val_loss_cin = cin_criterion(pred_cin_feats_val, true_cin_feats_val)
                        val_loss_this_batch += ALPHA_CIN_REG * val_loss_cin
                        epoch_val_cin_mae_sum += F.l1_loss(pred_cin_feats_val, true_cin_feats_val, reduction='sum').item()
                        total_cin_nodes_val += cin_mask_val.sum().item()
                        has_val_loss_component = True
                
                op_mask_val = batch.y_role == 1
                if torch.any(op_mask_val):
                    pred_op_logits_val = model.op_head(node_embeddings[op_mask_val])
                    true_op_types_val = batch.y_target_op_types[op_mask_val]
                    valid_op_target_mask_val = true_op_types_val != -1
                    if torch.any(valid_op_target_mask_val) and pred_op_logits_val[valid_op_target_mask_val].numel() > 0:
                        val_loss_op = op_criterion(pred_op_logits_val[valid_op_target_mask_val], true_op_types_val[valid_op_target_mask_val])
                        val_loss_this_batch += ALPHA_OP_CLS * val_loss_op
                        preds_op_val = pred_op_logits_val[valid_op_target_mask_val].argmax(dim=1)
                        epoch_val_op_correct += (preds_op_val == true_op_types_val[valid_op_target_mask_val]).sum().item()
                        epoch_val_op_total += valid_op_target_mask_val.sum().item()
                        has_val_loss_component = True
                
                if has_val_loss_component:
                     epoch_val_loss_sum += val_loss_this_batch.item()
                     num_val_batches_with_loss +=1

        avg_val_loss = epoch_val_loss_sum / num_val_batches_with_loss if num_val_batches_with_loss > 0 else float('inf')
        avg_val_mae_cin = epoch_val_cin_mae_sum / total_cin_nodes_val if total_cin_nodes_val > 0 else float('nan')
        avg_val_acc_op = epoch_val_op_correct / epoch_val_op_total if epoch_val_op_total > 0 else float('nan')
        history['val_loss'].append(avg_val_loss)
        history['val_mae_cin'].append(avg_val_mae_cin)
        history['val_acc_op'].append(avg_val_acc_op)
        end_epoch_time = time.time()

        print(f"Epoch {epoch:03d}/{EPOCHS} | Time: {end_epoch_time - start_epoch_time:.2f}s | "
              f"Train L: {avg_train_loss:.4f} MAE_cin: {avg_train_mae_cin:.4f} Acc_op: {avg_train_acc_op:.4f} | "
              f"Val L: {avg_val_loss:.4f} MAE_cin: {avg_val_mae_cin:.4f} Acc_op: {avg_val_acc_op:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'gat_v2_masked_best.pt'))
            print(f"  -> Val loss improved! Saved model.")
        else:
            epochs_no_improve += 1
            print(f"  -> Val loss did not improve. ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print("\nTraining finished.")
    final_model_path = os.path.join(MODEL_DIR, 'gat_v2_masked_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model state to {final_model_path}")

    # Save training configuration details
    train_config = {
        'learning_rate': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY,
        'epochs_run': actual_epochs_run,
        'batch_size': BATCH_SIZE,
        'hidden_channels': HIDDEN_CHANNELS, 'num_heads': NUM_HEADS,
        'dropout_rate': DROPOUT_RATE, 'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'gat_layers': GAT_LAYERS,
        'alpha_cin_reg': ALPHA_CIN_REG,
        'alpha_op_cls': ALPHA_OP_CLS
    }
    try:
         os.makedirs(MODEL_DIR, exist_ok=True)
         config_path = os.path.join(MODEL_DIR, 'train_config.json')
         with open(config_path, 'w') as f:
             json.dump(train_config, f, indent=2)
         print(f"Saved training configuration to {config_path}")
    except Exception as e:
         print(f"Warning: Could not save training configuration: {e}", file=sys.stderr)

    # Save history
    try:
        history_path = os.path.join(MODEL_DIR, 'training_history_gatv2.json')
        with open(history_path, 'w') as f:
            # Convert potential NaNs in history to None for JSON serializability
            json_serializable_history = {}
            for key, values in history.items():
                json_serializable_history[key] = [None if isinstance(v, float) and np.isnan(v) else v for v in values]
            json.dump(json_serializable_history, f, indent=2)
        print(f"Saved training history to {history_path}")
    except Exception as e: print(f"Warning: Could not save history: {e}")

    return True

if __name__ == "__main__":
    if not run_ag_train_gatv2():
        sys.exit(1)
