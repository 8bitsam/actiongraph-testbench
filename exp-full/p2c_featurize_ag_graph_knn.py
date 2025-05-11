# 2c_featurize_ag_graph_knn.py
import json
import os
import sys
import time
import numpy as np
from actiongraph import ActionGraph, OperationTypeEnum # Import OperationTypeEnum
from common_utils import load_json_data_from_directory, save_featurized_data
from tqdm import tqdm
from pymatgen.core import Composition # Import Composition

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
ACTION_GRAPH_RAW_DIR = os.path.join(DATA_DIR, "filtered-ag-data/") # Input
FEATURIZED_AG_GRAPH_KNN_DIR = os.path.join(DATA_DIR, "featurized-ag-graph-knn/") # Output

# Fixed list of operation types for one-hot encoding presence
OP_TYPES_FOR_FEATURES = list(OperationTypeEnum)

# --- Helper Function (copied from 2b, or move to common_utils if used by many) ---
def convert_node_attrs_to_json_serializable(node_attrs_list):
    """
    Converts a list of node attribute dictionaries to be JSON serializable.
    Specifically handles Pymatgen Composition and OperationTypeEnum.
    """
    serializable_list = []
    for attrs in node_attrs_list:
        s_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, Composition):
                s_attrs[key] = value.formula # Store formula string
            elif isinstance(value, OperationTypeEnum):
                s_attrs[key] = value.value # Store the string value of the enum
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                s_attrs[key] = value
            else:
                s_attrs[key] = str(value) # Fallback for other types
        serializable_list.append(s_attrs)
    return serializable_list

def featurize_ag_graph_stats(ag: ActionGraph):
    """Extracts simple graph-level statistics from an ActionGraph."""
    features = []
    features.append(len(ag.input_nodes))
    features.append(len(ag.operation_nodes))

    op_types_present_in_ag = set()
    for op_node_id in ag.operation_nodes:
        # Ensure op_type attribute exists and is an enum member
        op_node_data = ag.nodes[op_node_id]
        op_type_enum = op_node_data.get('op_type') if isinstance(op_node_data, dict) else None
        if isinstance(op_type_enum, OperationTypeEnum):
            op_types_present_in_ag.add(op_type_enum)

    for op_enum_member in OP_TYPES_FOR_FEATURES:
        features.append(1.0 if op_enum_member in op_types_present_in_ag else 0.0)

    avg_elements_in_cin = 0.0
    if ag.input_nodes:
        total_cin_elements = 0
        num_cin_valid_comp = 0
        for cin_id in ag.input_nodes:
            node_data = ag.nodes[cin_id]
            comp = node_data.get('composition') if isinstance(node_data, dict) else None
            if isinstance(comp, Composition): # Check if it's actually a Composition object
                total_cin_elements += len(comp.elements)
                num_cin_valid_comp +=1
        if num_cin_valid_comp > 0 : avg_elements_in_cin = total_cin_elements / num_cin_valid_comp
    features.append(avg_elements_in_cin)
    features.append(ag.number_of_edges())
    return np.array(features)


def run_featurize_ag_graph():
    print("--- Running AG Graph-Level k-NN Featurization ---")
    serialized_ag_entries = load_json_data_from_directory(ACTION_GRAPH_RAW_DIR, "Loading Serialized AGs")
    if not serialized_ag_entries: return False

    print("\nExtracting graph-level features from ActionGraphs...")
    features_list, recipes_list, targets_list, original_ids_list = [], [], [], []
    processed_count, skipped_count = 0, 0
    feature_dim = -1 # To check consistency

    for ag_data in tqdm(serialized_ag_entries, desc="Featurizing AG graph stats"):
        source_id = ag_data.get('_source_file', f"ag_{processed_count}") # If _source_file was added during conversion
        try:
            ag = ActionGraph.deserialize(ag_data) # Deserialize first
            if not ag.nodes() or not ag.output_nodes:
                skipped_count += 1
                continue

            feature_vector = featurize_ag_graph_stats(ag) # Pass the AG instance
            if feature_vector is None: # featurize_ag_graph_stats might return None on error
                skipped_count +=1
                continue

            if feature_dim == -1:
                feature_dim = len(feature_vector)
            elif len(feature_vector) != feature_dim:
                # print(f"Warning: Feature dim mismatch for {source_id} ({len(feature_vector)} vs {feature_dim}). Skipping.")
                skipped_count +=1
                continue

            features_list.append(feature_vector)

            # --- MODIFICATION HERE ---
            # Get raw node attributes
            raw_recipe_precursors = [ag.nodes[nid] for nid in ag.input_nodes if nid in ag.nodes]
            raw_recipe_operations = [ag.nodes[nid] for nid in ag.operation_nodes if nid in ag.nodes]

            # Convert them to be JSON serializable
            serializable_precursors = convert_node_attrs_to_json_serializable(raw_recipe_precursors)
            serializable_operations = convert_node_attrs_to_json_serializable(raw_recipe_operations)

            recipes_list.append({
                "precursors": serializable_precursors,
                "operations": serializable_operations
            })
            # --- END OF MODIFICATION ---

            # Target is the formula of the first C_out node (ensure C_out exists)
            if ag.output_nodes and ag.output_nodes[0] in ag.nodes:
                 targets_list.append(ag.nodes[ag.output_nodes[0]].get('formula', 'UnknownTarget'))
            else:
                 targets_list.append('UnknownTarget_NoValidOutputNode')

            original_ids_list.append(source_id)
            processed_count += 1

        except Exception as e:
            # print(f"Error processing AG {source_id} for graph stats: {e}")
            skipped_count += 1

    if not features_list:
        print("Error: No AGs successfully featurized for graph stats.", file=sys.stderr)
        return False

    print(f"Processed {processed_count} AGs for graph features. Skipped {skipped_count}.")
    features_arr = np.array(features_list)
    print(f"Final feature array shape: {features_arr.shape}")

    return save_featurized_data(
        FEATURIZED_AG_GRAPH_KNN_DIR, features_arr, recipes_list,
        targets_list, original_ids_list, {"note": "Element map not used for these graph-level features"}
    )

if __name__ == "__main__":
    if not run_featurize_ag_graph():
        sys.exit(1)
