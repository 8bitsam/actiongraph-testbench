import torch
from pymatgen.core import Composition
from torch_geometric.data import Data
from train_gnn import PrecursorPredictionGNN
from featurize_action_graphs import get_composition_features

def predict_precursors(target_formula, top_k=5):
    """Predict precursors for a given target material."""
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load a reference dataset to get feature dimensions
    all_data = torch.load("Data/ag-data/featurized/all_data.pt", map_location=device)
    node_features = all_data[0].x.shape[1]
    
    # Create a simple graph with just the target material
    target_features = get_composition_features(target_formula)
    
    # Add placeholder nodes for potential precursors (based on common precursors)
    common_precursors = ["Li2CO3", "CoO", "Fe2O3", "MnO2", "NiO", "CaO", "TiO2", 
                         "Al2O3", "SiO2", "ZnO", "MgO", "Na2CO3", "K2CO3"]
    
    # Create a graph with the target and potential precursors
    node_features = [target_features]  # First node is the target
    
    for precursor in common_precursors:
        try:
            features = get_composition_features(precursor)
            node_features.append(features)
        except:
            pass
    
    # Create a simple graph structure
    num_nodes = len(node_features)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Connect all precursor candidates to the target
    edge_index = []
    for i in range(1, num_nodes):
        edge_index.append([i, 0])  # Precursor -> Target
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)
    data.output_nodes = [0]  # Target is the output node
    
    # Load the model
    model = PrecursorPredictionGNN(node_features=x.shape[1], hidden_channels=64, num_layers=3)
    model.load_state_dict(torch.load("Data/models/best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    # Make prediction
    data = data.to(device)
    with torch.no_grad():
        scores = model(data.x, data.edge_index, data.batch)
    
    # Get scores for precursor candidates (exclude the target)
    precursor_scores = scores[1:].cpu()
    
    # Sort by score
    sorted_indices = torch.argsort(precursor_scores, descending=True)
    
    # Return top-k precursors and their scores
    top_precursors = []
    for i in range(min(top_k, len(sorted_indices))):
        idx = sorted_indices[i].item() + 1  # +1 because we excluded the target
        precursor = common_precursors[idx-1]
        score = torch.sigmoid(precursor_scores[idx-1]).item()
        top_precursors.append((precursor, score))
    
    return top_precursors

if __name__ == "__main__":
    target = "LiCoO2"
    print(f"Predicting precursors for {target}:")
    precursors = predict_precursors(target)
    
    for precursor, score in precursors:
        print(f"  {precursor}: {score:.4f}")
