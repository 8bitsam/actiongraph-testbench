import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support # Use this for clarity
from train_gnn import PrecursorPredictionGNN # Model definition
from serialization_utils import safe_load # For model state_dict
import pandas as pd # For result saving

def compute_similarity(true_precursors, pred_precursors, sim_type='jaccard'):
    """Compute similarity between true and predicted precursor sets (indices)."""
    true_set = set(true_precursors)
    pred_set = set(pred_precursors)

    if not true_set and not pred_set:
        return 1.0  # Both empty sets are considered identical

    if not true_set or not pred_set:
        # If only one set is empty, similarity depends on the metric
        if sim_type == 'overlap':
            return 0.0 # No intersection possible
        # For Jaccard and Dice, if union or sum of sizes is non-zero, result is 0
        elif sim_type == 'jaccard':
             union = len(true_set.union(pred_set))
             return 0.0 if union > 0 else 1.0 # Should only be 1.0 if both were empty initially
        elif sim_type == 'dice':
             denominator = len(true_set) + len(pred_set)
             return 0.0 if denominator > 0 else 1.0 # Should only be 1.0 if both were empty initially


    intersection = len(true_set.intersection(pred_set))

    if sim_type == 'jaccard':
        # |A ∩ B| / |A ∪ B|
        union = len(true_set.union(pred_set))
        return intersection / union if union > 0 else 1.0 # Handle case where union is 0 (both empty)

    elif sim_type == 'dice':
        # 2 * |A ∩ B| / (|A| + |B|)
        denominator = len(true_set) + len(pred_set)
        return (2 * intersection) / denominator if denominator > 0 else 1.0 # Handle case where both empty

    elif sim_type == 'overlap':
        # |A ∩ B| / min(|A|, |B|)
        denominator = min(len(true_set), len(pred_set))
        return intersection / denominator if denominator > 0 else 1.0 # Handle if min size is 0

    else:
        raise ValueError(f"Unsupported similarity type: {sim_type}")

def evaluate_model(top_k_values=[1, 3, 5, 10], threshold=0.5):
    """Evaluate model performance on test data."""
    # --- Configuration ---
    model_dir = "Data/models/"
    test_data_path = os.path.join(model_dir, "test_data.pt")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    results_file = "evaluation_results.csv"

    # --- Load Test Data ---
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}. Run training first.")
        return
    try:
        # Load using standard torch.load BUT SPECIFY weights_only=False
        # Also add map_location for flexibility
        test_data = torch.load(test_data_path, map_location=torch.device('cpu'), weights_only=False)
        print(f"Loaded {len(test_data)} test graphs from {test_data_path}")
    except Exception as e:
         print(f"Error loading test data: {e}")
         return

    if not test_data:
        print("No test data loaded.")
        return

    # Use batch_size=1 for evaluation to handle graphs individually
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # --- Load Model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get feature dimension from the first test sample
    node_feature_dim = test_data[0].x.shape[1]

    # Initialize model architecture
    model = PrecursorPredictionGNN(
        node_features=node_feature_dim,
        hidden_channels=32, # Match training configuration
        num_layers=3,       # Match training configuration
        dropout=0.0         # Set dropout to 0 for evaluation
    )

    if not os.path.exists(best_model_path):
        print(f"Model file not found at {best_model_path}. Run training first.")
        return
    try:
        # Load state dict using safe_load
        state_dict = safe_load(best_model_path)
        model.load_state_dict(state_dict)
        print(f"Loaded model state from {best_model_path}")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- Metrics Containers ---
    recall_at_k = {k: [] for k in top_k_values}
    precision_at_k = {k: [] for k in top_k_values}
    similarities = {'jaccard': [], 'dice': [], 'overlap': []}
    all_preds = []
    all_targets = []

    # --- Evaluation Loop ---
    print("Starting evaluation...")
    total_graphs = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            total_graphs += 1
            batch = batch.to(device)

            # --- Get Predictions ---
            scores = model(batch.x, batch.edge_index, batch.batch) # Get raw scores (logits)
            probs = torch.sigmoid(scores) # Convert to probabilities if needed for thresholding/ranking

            # --- Identify True Precursors and Candidates ---
            # Since batch_size=1, all nodes belong to the same graph (index 0)
            target_nodes = batch.y.cpu().numpy() # 1.0 for precursors, 0.0 otherwise
            is_output_nodes = batch.is_output.cpu().numpy() # 1.0 for output nodes

            # Indices of true precursors
            true_precursor_indices = np.where(target_nodes == 1.0)[0].tolist()
            # Indices of output nodes
            output_node_indices = np.where(is_output_nodes == 1.0)[0].tolist()
            # Indices of all nodes
            all_node_indices = list(range(batch.num_nodes))

            # Candidate nodes for prediction are all nodes *except* the output nodes
            candidate_indices = [idx for idx in all_node_indices if idx not in output_node_indices]

            if not candidate_indices:
                 # print(f"Graph {i}: No candidate nodes found after excluding outputs. Skipping metrics for this graph.")
                 # Handle metrics appropriately, e.g., append 0 or skip
                 for k in top_k_values:
                      recall_at_k[k].append(0.0) # Or handle based on metric definition
                      precision_at_k[k].append(0.0)
                 for sim_type in similarities:
                     similarities[sim_type].append(0.0) # Or handle based on metric definition
                 continue # Skip if no candidates

            # Get scores for candidate nodes ONLY
            candidate_scores = scores[candidate_indices].cpu()

            # --- Calculate Top-k Metrics (Recall@k, Precision@k) ---
            # Sort candidates by score (descending)
            num_candidates = len(candidate_indices)
            sorted_candidate_indices_local = torch.argsort(candidate_scores, descending=True)

            # Map sorted local indices back to global graph indices
            sorted_candidate_indices_global = [candidate_indices[idx.item()] for idx in sorted_candidate_indices_local]

            true_precursors_set = set(true_precursor_indices)
            num_true_precursors = len(true_precursors_set)

            for k in top_k_values:
                # Get top-k predicted global indices
                top_k_preds_global = sorted_candidate_indices_global[:min(k, num_candidates)]
                top_k_preds_set = set(top_k_preds_global)

                # Calculate intersection
                intersection = true_precursors_set.intersection(top_k_preds_set)
                num_intersect = len(intersection)

                # Recall@k = |Intersection| / |True Precursors|
                recall = num_intersect / num_true_precursors if num_true_precursors > 0 else 1.0 if not top_k_preds_set else 0.0
                recall_at_k[k].append(recall)

                # Precision@k = |Intersection| / k
                precision = num_intersect / k if k > 0 else 0.0
                precision_at_k[k].append(precision)


            # --- Calculate Similarity Metrics ---
            # Use the top-N predictions, where N is typically the number of true precursors
            # Or use a fixed k like max(top_k_values) - let's use num_true_precursors
            num_to_predict = max(1, num_true_precursors) # Predict at least 1
            top_n_preds_global = sorted_candidate_indices_global[:min(num_to_predict, num_candidates)]

            for sim_type in similarities:
                sim = compute_similarity(true_precursor_indices, top_n_preds_global, sim_type)
                similarities[sim_type].append(sim)

            # --- Collect for Classification Metrics ---
            # Use scores of *all* nodes for overall classification metrics
            # Apply threshold to raw scores (logits)
            predicted_labels = (scores.cpu().numpy() > 0).astype(int) # Threshold logits at 0
            # Or threshold probabilities: predicted_labels = (probs.cpu().numpy() > threshold).astype(int)

            all_targets.extend(target_nodes.astype(int))
            all_preds.extend(predicted_labels)

    # --- Aggregate and Print Results ---
    print("\n--- Evaluation Results ---")

    # Top-k Metrics
    print("\nTop-k Metrics (Averaged over graphs):")
    results_summary = {}
    for k in top_k_values:
        avg_recall_k = np.mean(recall_at_k[k]) if recall_at_k[k] else 0
        avg_precision_k = np.mean(precision_at_k[k]) if precision_at_k[k] else 0
        print(f"  Recall@{k}:    {avg_recall_k:.4f}")
        print(f"  Precision@{k}: {avg_precision_k:.4f}")
        results_summary[f'Recall@{k}'] = avg_recall_k
        results_summary[f'Precision@{k}'] = avg_precision_k


    # Similarity Metrics
    print("\nSimilarity Metrics (Averaged over graphs):")
    for sim_type, values in similarities.items():
        avg_sim = np.mean(values) if values else 0
        print(f"  {sim_type.capitalize()}: {avg_sim:.4f}")
        results_summary[f'Similarity_{sim_type.capitalize()}'] = avg_sim

    # Overall Classification Metrics (Calculated across all nodes in the test set)
    print("\nOverall Classification Metrics (Calculated across all nodes):")
    if all_targets and all_preds:
         # Use zero_division=0 to avoid warnings when a class is not predicted or not present
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )
        # Calculate metrics specifically for the positive class (precursors)
        pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
             all_targets, all_preds, average=None, labels=[1], zero_division=0
        )
        # Handle cases where label 1 might not exist
        pos_precision = pos_precision[0] if len(pos_precision) > 0 else 0.0
        pos_recall = pos_recall[0] if len(pos_recall) > 0 else 0.0
        pos_f1 = pos_f1[0] if len(pos_f1) > 0 else 0.0


        print(f"  Binary Precision (Overall): {precision:.4f}")
        print(f"  Binary Recall (Overall):    {recall:.4f}")
        print(f"  Binary F1 Score (Overall):  {f1:.4f}")
        print(f"  Precision (Precursor Class): {pos_precision:.4f}")
        print(f"  Recall (Precursor Class):    {pos_recall:.4f}")
        print(f"  F1 Score (Precursor Class):  {pos_f1:.4f}")

        results_summary['Precision_Overall'] = precision
        results_summary['Recall_Overall'] = recall
        results_summary['F1_Overall'] = f1
        results_summary['Precision_Precursor'] = pos_precision
        results_summary['Recall_Precursor'] = pos_recall
        results_summary['F1_Precursor'] = pos_f1
    else:
        print("  Could not calculate classification metrics (no predictions/targets).")

    # --- Save Results Summary ---
    try:
         results_df = pd.DataFrame([results_summary])
         results_df.to_csv(results_file, index=False)
         print(f"\nEvaluation summary saved to {results_file}")
    except Exception as e:
         print(f"\nError saving evaluation results: {e}")


if __name__ == "__main__":
    evaluate_model()
