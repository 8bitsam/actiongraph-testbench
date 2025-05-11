# 2b_featurize_ag_cout_knn.py
import json
import os
import sys
import time
import numpy as np
from actiongraph import ActionGraph, OperationTypeEnum # Import OperationTypeEnum
from common_utils import (
    load_json_data_from_directory, get_global_element_map,
    featurize_chemical_formula, save_featurized_data
)
from tqdm import tqdm
from pymatgen.core import Composition # Ensure Composition is imported if used directly

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
ACTION_GRAPH_RAW_DIR = os.path.join(DATA_DIR, "filtered-ag-data/") # Input: Serialized AGs
FEATURIZED_AG_COUT_KNN_DIR = os.path.join(DATA_DIR, "featurized-ag-cout-knn/") # Output

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
                # Or value.as_dict() if you need the full composition dict
            elif isinstance(value, OperationTypeEnum):
                s_attrs[key] = value.value # Store the string value of the enum
            elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                s_attrs[key] = value
            else:
                s_attrs[key] = str(value) # Fallback for other types
        serializable_list.append(s_attrs)
    return serializable_list

def run_featurize_ag_cout():
    print("--- Running AG C_out k-NN Featurization ---")
    serialized_ag_entries = load_json_data_from_directory(ACTION_GRAPH_RAW_DIR, "Loading Serialized AGs")
    if not serialized_ag_entries: return False

    all_cout_node_attrs_for_map = [] # For creating element map
    for ag_data in serialized_ag_entries:
        # Temporarily deserialize just to get C_out node formulas for element_map
        # This is a bit inefficient but ensures we use the deserialized AG's view
        try:
            ag_temp = ActionGraph.deserialize(ag_data)
            for out_node_id in ag_temp.output_nodes:
                if out_node_id in ag_temp.nodes:
                     all_cout_node_attrs_for_map.append(ag_temp.nodes[out_node_id])
        except Exception:
             pass # Ignore errors for map creation, main loop will handle errors again

    if not all_cout_node_attrs_for_map:
         print("Error: No C_out nodes found in any ActionGraphs for element map.", file=sys.stderr)
         return False

    element_map = get_global_element_map(all_cout_node_attrs_for_map, data_key='node_attrs', formula_key='formula')
    if not element_map: return False

    print("\nFeaturizing C_out nodes from ActionGraphs...")
    features_list, recipes_list, targets_list, original_ids_list = [], [], [], []
    processed_count, skipped_count = 0, 0

    for ag_data in tqdm(serialized_ag_entries, desc="Featurizing AG C_out nodes"):
        source_id = ag_data.get('_source_file', f"ag_{processed_count}") # Assuming _source_file was added
        try:
            ag = ActionGraph.deserialize(ag_data)
            if not ag.output_nodes:
                skipped_count += 1
                continue

            target_node_id = ag.output_nodes[0] # Assuming one C_out for simplicity
            target_node_attrs = ag.nodes[target_node_id]
            target_formula = target_node_attrs.get('formula')

            if not target_formula:
                skipped_count += 1
                continue

            feature_vector = featurize_chemical_formula(target_formula, element_map)

            if feature_vector is not None:
                features_list.append(feature_vector)

                # --- MODIFICATION HERE ---
                # Get raw node attributes
                raw_recipe_precursors = [ag.nodes[nid] for nid in ag.input_nodes]
                raw_recipe_operations = [ag.nodes[nid] for nid in ag.operation_nodes]

                # Convert them to be JSON serializable
                serializable_precursors = convert_node_attrs_to_json_serializable(raw_recipe_precursors)
                serializable_operations = convert_node_attrs_to_json_serializable(raw_recipe_operations)

                recipes_list.append({
                    "precursors": serializable_precursors,
                    "operations": serializable_operations
                })
                # --- END OF MODIFICATION ---

                targets_list.append(target_formula)
                original_ids_list.append(source_id)
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            # print(f"Error processing AG {source_id}: {e}")
            skipped_count += 1

    if not features_list:
        print("Error: No C_out nodes successfully featurized.", file=sys.stderr)
        return False

    print(f"Processed {processed_count} AGs for C_out features. Skipped {skipped_count}.")
    features_arr = np.array(features_list)
    print(f"Final feature array shape: {features_arr.shape}")

    return save_featurized_data(
        FEATURIZED_AG_COUT_KNN_DIR, features_arr, recipes_list,
        targets_list, original_ids_list, element_map
    )

if __name__ == "__main__":
    if not run_featurize_ag_cout():
        sys.exit(1)
