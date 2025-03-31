import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from train_gnn import PrecursorPredictionGNN, prepare_batch_data
from serialization_utils import safe_load

def compute_similarity(true_precursors, pred_precursors, sim_type='jaccard'):
    """Compute similarity between true and predicted precursor sets."""
    if not true_precursors and not pred_precursors:
        return 1.0  # Both empty sets are considered identical
    
    if not true_precursors or not pred_precursors:
        return 0.0  # One empty, one non-empty = no similarity
    
    true_set = set(true_precursors)
    pred_set = set(pred_precursors)
    
    if sim_type == 'jaccard':
        # |A ∩ B| / |A ∪ B|
        intersection = len(true_set.intersection(pred_set))
        union = len(true_set.union(pred_set))
        return intersection / union
    
    elif sim_type == 'dice':
        # 2 * |A ∩ B| / (|A| + |B|)
        intersection = len(true_set.intersection(pred_set))
        return 2 * intersection / (len(true_set) + len(pred_set))
    
    elif sim_type == 'overlap':
        # |A ∩ B| / min(|A|, |B|)
        intersection = len(true_set.intersection(pred_set))
        return intersection / min(len(true_set), len(pred_set))
    
    else:
        raise ValueError(f"Unsupported similarity type: {sim_type}")

def evaluate_model(top_k_values=[1, 3, 5, 10]):
    """Evaluate model performance on test data."""
    # Load test data and model
    test_data = safe_load("Data/models/test_data.pt")
    test_loader = DataLoader(test_data, batch_size=1)  # Use batch size of 1 for evaluation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_features = test_data[0].x.shape[1]
    
    model = PrecursorPredictionGNN(
        node_features=node_features, 
        hidden_channels=64, 
        num_layers=3
    )
    model.load_state_dict(safe_load("Data/models/best_model.pt"))
    model.to(device)
    model.eval()
    
    # Metrics containers
    top_k_hits = {k: 0 for k in top_k_values}
    similarities = {'jaccard': [], 'dice': [], 'overlap': []}
    
    total_graphs = 0
    precisions, recalls, f1s = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            total_graphs += 1
            batch = batch.to(device)
            
            # Get model predictions
            scores = model(batch.x, batch.edge_index, batch.batch)
            
            # For evaluation, handle each graph individually
            # Since we're using batch_size=1, all nodes belong to the same graph
            graph_mask = (batch.batch == 0)  # All nodes in the first (and only) graph
            graph_nodes = torch.nonzero(graph_mask).squeeze(1)
            
            # Create target tensor
            target = torch.zeros(len(graph_nodes), dtype=torch.float, device=device)
            
            # Mark input nodes as positive examples
            true_precursors = []
            for idx in batch.input_nodes[0]:  # Access by list index, not by tensor index
                if idx < len(graph_nodes):
                    node_idx = graph_nodes[idx].item()
                    target[idx] = 1.0
                    true_precursors.append(idx)
            
            # Get output nodes to exclude from prediction candidates
            output_nodes = batch.output_nodes[0]  # Access by list index
            
            # Get candidate nodes (exclude output nodes)
            candidates = [i for i in range(len(graph_nodes)) if i not in output_nodes]
            candidate_scores = scores[candidates]
            
            # Get top-k predictions
            _, indices = torch.sort(candidate_scores, descending=True)
            
            # Calculate metrics
            for k in top_k_values:
                if k <= len(indices):
                    top_k_preds = [candidates[idx.item()] for idx in indices[:k]]
                    # Check if any true precursor is in top-k
                    if any(p in true_precursors for p in top_k_preds):
                        top_k_hits[k] += 1
            
            # Calculate similarity metrics
            max_k = max(top_k_values)
            if max_k <= len(indices):
                top_preds = [candidates[idx.item()] for idx in indices[:max_k]]
                
                for sim_type in similarities.keys():
                    sim = compute_similarity(true_precursors, top_preds, sim_type)
                    similarities[sim_type].append(sim)
            
            # Binary classification metrics
            predicted_labels = (scores > 0).float()
            precision = precision_score(target.cpu(), predicted_labels.cpu(), zero_division=0)
            recall = recall_score(target.cpu(), predicted_labels.cpu(), zero_division=0)
            f1 = f1_score(target.cpu(), predicted_labels.cpu(), zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
    
    # Print results
    print("\nEvaluation Results:")
    print("\nTop-k Accuracy:")
    for k in top_k_values:
        accuracy = top_k_hits[k] / total_graphs if total_graphs > 0 else 0
        print(f"  Top-{k}: {accuracy:.4f}")
    
    print("\nSimilarity Metrics:")
    for sim_type, values in similarities.items():
        avg_sim = np.mean(values) if values else 0
        print(f"  {sim_type.capitalize()}: {avg_sim:.4f}")
    
    print("\nClassification Metrics:")
    print(f"  Precision: {np.mean(precisions):.4f}")
    print(f"  Recall: {np.mean(recalls):.4f}")
    print(f"  F1 Score: {np.mean(f1s):.4f}")

if __name__ == "__main__":
    evaluate_model()
