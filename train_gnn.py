import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from serialization_utils import safe_load, safe_save

class PrecursorPredictionGNN(nn.Module):
    """GNN model for predicting precursors given a target material."""
    def __init__(self, node_features, hidden_channels, num_layers, dropout=0.1):
        super(PrecursorPredictionGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial embedding layer
        self.embedding = nn.Linear(node_features, hidden_channels)
        
        # Graph attention layers with concat=False to maintain dimensionality
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=4,
                concat=False,  # Average outputs from different heads instead of concatenating
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
    
    def forward(self, x, edge_index, batch):
        """Forward pass through the network."""
        # Initial embedding
        x = self.embedding(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply graph attention layers
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level prediction scores
        scores = self.predictor(x).squeeze(-1)
        
        return scores

def prepare_batch_data(batch, device):
    """Prepare training data from a batch."""
    batch = batch.to(device)
    
    # Create target tensor (1 for precursor nodes, 0 for others)
    target = torch.zeros(batch.x.size(0), dtype=torch.float, device=device)
    
    # Get number of graphs in the batch
    num_graphs = batch.batch.max().item() + 1
    
    # Calculate node offsets for each graph in the batch
    node_offsets = [0]
    current_offset = 0
    
    # For each graph, find its nodes and mark input nodes as positive examples
    for graph_idx in range(num_graphs):
        # Find which nodes belong to this graph
        graph_mask = (batch.batch == graph_idx)
        graph_nodes = torch.nonzero(graph_mask).squeeze(1)
        
        # Store offset for next graph
        current_offset += len(graph_nodes)
        if graph_idx < num_graphs - 1:
            node_offsets.append(current_offset)
        
        # Get input nodes for this graph and apply offset
        if hasattr(batch, 'input_nodes') and isinstance(batch.input_nodes, list):
            if graph_idx < len(batch.input_nodes):
                for input_idx in batch.input_nodes[graph_idx]:
                    if input_idx < len(graph_nodes):
                        # Mark as a positive example
                        node_idx = graph_nodes[input_idx].item()
                        target[node_idx] = 1.0
    
    return {
        'x': batch.x,
        'edge_index': batch.edge_index,
        'batch': batch.batch,
        'target': target
    }

def train_model():
    """Train the GNN model for precursor prediction with enhanced diagnostics."""
    # Load the featurized dataset
    data_path = "Data/ag-data/featurized/all_data.pt"
    os.makedirs("Data/models/", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"No featurized data found at {data_path}. Please run the featurization step first.")
        return
    
    try:
        all_data = safe_load(data_path)
        print(f"Successfully loaded {len(all_data)} featurized action graphs")
    except Exception as e:
        print(f"Error loading featurized data: {e}")
        return
    
    if not all_data:
        print("No data available for training. Please check the featurization step.")
        return
    
    # Detailed diagnostics on loaded data
    has_input_nodes = sum(1 for d in all_data if hasattr(d, 'input_nodes'))
    has_output_nodes = sum(1 for d in all_data if hasattr(d, 'output_nodes'))
    has_non_empty_inputs = sum(1 for d in all_data if hasattr(d, 'input_nodes') and len(d.input_nodes) > 0)
    
    print(f"Diagnostics on loaded data:")
    print(f"  - Total samples: {len(all_data)}")
    print(f"  - Samples with input_nodes attribute: {has_input_nodes}")
    print(f"  - Samples with output_nodes attribute: {has_output_nodes}")
    print(f"  - Samples with non-empty input_nodes: {has_non_empty_inputs}")
    
    # Sample inspection (first few samples)
    print("\nInspecting first samples:")
    for i, data in enumerate(all_data[:3]):
        print(f"  Sample {i}:")
        print(f"    - Node features shape: {data.x.shape}")
        print(f"    - Has input_nodes: {hasattr(data, 'input_nodes')}")
        if hasattr(data, 'input_nodes'):
            print(f"    - Input nodes count: {len(data.input_nodes)}")
            print(f"    - Input nodes: {data.input_nodes}")
        print(f"    - Has output_nodes: {hasattr(data, 'output_nodes')}")
        if hasattr(data, 'output_nodes'):
            print(f"    - Output nodes count: {len(data.output_nodes)}")
    
    # Filter data to ensure all graphs have input and output nodes
    valid_data = [d for d in all_data if hasattr(d, 'input_nodes') and 
                 hasattr(d, 'output_nodes') and len(d.input_nodes) > 0]
    
    if not valid_data:
        print("\nNo valid data available for training after filtering. Please check the featurization step.")
        return
    
    print(f"\nTraining on {len(valid_data)} valid graphs out of {len(all_data)} total")
    
    # Split into training and test sets
    train_data, test_data = train_test_split(valid_data, test_size=0.2, random_state=42)
    print(f"Training on {len(train_data)} graphs, testing on {len(test_data)} graphs")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model
    node_features = valid_data[0].x.shape[1]
    model = PrecursorPredictionGNN(
        node_features=node_features, 
        hidden_channels=64, 
        num_layers=3
    )
    
    # Define optimizer and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training loop
    num_epochs = 100
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            try:
                batch_data = prepare_batch_data(batch, device)
                optimizer.zero_grad()
                
                # Forward pass
                scores = model(batch_data['x'], batch_data['edge_index'], batch_data['batch'])
                
                # Binary cross entropy loss
                loss = F.binary_cross_entropy_with_logits(scores, batch_data['target'])
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                print(f"Error processing batch: {e}")
                continue
        
        train_loss = total_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_data = prepare_batch_data(batch, device)
                scores = model(batch_data['x'], batch_data['edge_index'], batch_data['batch'])
                loss = F.binary_cross_entropy_with_logits(scores, batch_data['target'])
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            safe_save(model.state_dict(), "Data/models/best_model.pt")
    
    # Save test data for evaluation
    safe_save(test_data, "Data/models/test_data.pt")
    print("Training completed and model saved successfully.")
