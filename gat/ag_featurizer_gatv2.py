# ag_featurizer_gatv2.py
import json
import os
import sys
import time
import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core import Composition, Element
from emmet.core.synthesis import OperationTypeEnum # Ensure this is correctly importable
from actiongraph import ActionGraph # Your ActionGraph class
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))

ACTION_GRAPH_INPUT_DIR = os.path.join(DATA_DIR, "filtered-ag-data/") # Input: Unique AG JSONs
FEATURIZED_AG_OUTPUT_DIR = os.path.join(DATA_DIR, "ag-data-featurized-gatv2/") # Base output
TRAIN_DATA_OUT_DIR = os.path.join(FEATURIZED_AG_OUTPUT_DIR, "training-data")
TEST_DATA_OUT_DIR = os.path.join(FEATURIZED_AG_OUTPUT_DIR, "testing-data")

RANDOM_STATE = 42
TEST_SPLIT_SIZE = 0.25

ELEMENT_PROPS = [
    'atomic_mass', 'atomic_radius', 'atomic_volume',
    'electronegativity', 'electron_affinity', 'ionization_energy'
]
NUM_ELEMENT_PROPS = len(ELEMENT_PROPS)
OP_TYPES_ENUM_LIST = sorted(list(OperationTypeEnum), key=lambda e: e.value) # Sort for consistent mapping

# --- Helper: Feature Extraction ---
def get_global_element_map(ag_json_file_list, ag_input_dir):
    print("Scanning all AGs to create a global element map...")
    all_elements = set()
    for filename in tqdm(ag_json_file_list, desc="Scanning elements"):
        filepath = os.path.join(ag_input_dir, filename)
        try:
            with open(filepath, 'r') as f: ag_data = json.load(f)
            # AG serialize stores Pymatgen composition as dict
            for node in ag_data.get('nodes', []):
                if node.get('type') == 'chemical':
                    comp_dict = node.get('composition') # This is from Pymatgen's as_dict()
                    if isinstance(comp_dict, dict):
                        all_elements.update(comp_dict.keys()) # Keys are element symbols
        except Exception: pass # Ignore errors in individual files for map creation
    try:
        common_elements = {el.symbol for el in Element}
        all_elements.update(common_elements)
    except Exception: pass
    sorted_elements = sorted(list(all_elements - {"dummy"})) # Pymatgen Composition might add "dummy"
    if not sorted_elements:
        raise ValueError("No valid elements found to create map.")
    print(f"Global element map created with {len(sorted_elements)} elements.")
    return {el: i for i, el in enumerate(sorted_elements)}

def extract_chemical_features(attrs, element_map):
    """Extracts [element_fractions, avg_element_properties] vector."""
    if not attrs or not isinstance(attrs, dict): return None
    pymatgen_comp = attrs.get('composition') # This should be a Pymatgen Composition object after AG.deserialize
    if not isinstance(pymatgen_comp, Composition):
        # Try to create from formula if 'composition' wasn't a Pymatgen object
        formula = attrs.get('formula')
        if not formula: return None
        try: pymatgen_comp = Composition(formula)
        except: return None

    if pymatgen_comp.num_atoms < 1: return None
    el_amt_dict = pymatgen_comp.get_el_amt_dict()
    total_atoms = pymatgen_comp.num_atoms

    comp_vector = np.zeros(len(element_map))
    for el_sym, amount in el_amt_dict.items():
        if el_sym in element_map:
            comp_vector[element_map[el_sym]] = amount / total_atoms

    prop_vector = np.zeros(NUM_ELEMENT_PROPS)
    for i, prop_name in enumerate(ELEMENT_PROPS):
        weighted_sum, total_wt_for_prop, prop_avail = 0.0, 0.0, False
        for element in pymatgen_comp.elements: # Iterate Pymatgen Element objects
            try:
                prop_val = getattr(element, prop_name, None)
                if prop_val is not None and isinstance(prop_val, (int, float)):
                    amount_in_comp = pymatgen_comp.get_atomic_fraction(element) * total_atoms # Get actual count
                    weighted_sum += prop_val * amount_in_comp
                    total_wt_for_prop += amount_in_comp
                    prop_avail = True
            except Exception: continue
        if prop_avail and total_wt_for_prop > 0:
            prop_vector[i] = weighted_sum / total_wt_for_prop

    return np.nan_to_num(np.concatenate((comp_vector, prop_vector)))

def extract_operation_features(attrs, op_types_map):
    """Extracts features for an operation node: [one_hot_type, condition_features]."""
    if not attrs or not isinstance(attrs, dict): return None
    op_features = []

    # 1. One-hot operation type
    op_type_one_hot = np.zeros(len(op_types_map))
    op_type_enum = attrs.get('op_type') # Should be OperationTypeEnum instance
    if isinstance(op_type_enum, OperationTypeEnum) and op_type_enum in op_types_map:
        op_type_one_hot[op_types_map[op_type_enum]] = 1.0
    op_features.extend(op_type_one_hot)

    # 2. Condition features (ensure these keys match ActionGraph._parse_conditions output)
    conditions = attrs.get('conditions', {})
    # Temperature_K: min, max, mean, count/presence (4 features)
    temps = conditions.get('temperature_K', [])
    op_features.extend([np.min(temps) if temps else 0.0,
                        np.max(temps) if temps else 0.0,
                        np.mean(temps) if temps else 0.0,
                        float(len(temps)>0)])
    # Time_s: min, max, mean, count/presence (4 features)
    times = conditions.get('time_s', [])
    op_features.extend([np.min(times) if times else 0.0,
                        np.max(times) if times else 0.0,
                        np.mean(times) if times else 0.0,
                        float(len(times)>0)])
    # Atmosphere: count of distinct atmospheres (or simple presence)
    op_features.append(float(len(set(conditions.get('atmosphere', []))) > 0))
    # Device/Media: presence flags
    op_features.append(1.0 if conditions.get('device') else 0.0)
    op_features.append(1.0 if conditions.get('media') else 0.0)
    # Other conditions count (simple proxy)
    op_features.append(float(len(conditions.get('other_conditions', {})) > 0))

    return np.array(op_features)

# --- Main Featurization Script ---
def run_ag_featurize_gatv2():
    print("--- Running ActionGraph Featurization for GATv2 (Masked Prediction Task) ---")
    if not os.path.isdir(ACTION_GRAPH_INPUT_DIR):
        print(f"Error: Input AG directory not found: {ACTION_GRAPH_INPUT_DIR}", file=sys.stderr)
        return False

    all_ag_json_filenames = [f for f in os.listdir(ACTION_GRAPH_INPUT_DIR) if f.lower().endswith('.json')]
    if not all_ag_json_filenames:
        print(f"No AG JSON files found in {ACTION_GRAPH_INPUT_DIR}", file=sys.stderr)
        return False

    element_map = get_global_element_map(all_ag_json_filenames, ACTION_GRAPH_INPUT_DIR)
    if not element_map: return False
    op_types_map = {op_type: i for i, op_type in enumerate(OP_TYPES_ENUM_LIST)}

    chem_feat_dim = len(element_map) + NUM_ELEMENT_PROPS
    # one-hot type + 4 temp + 4 time + 1 atm + 1 dev + 1 media + 1 other_cond_presence
    op_feat_dim_raw = len(op_types_map) + 4 + 4 + 1 + 1 + 1 + 1

    # For "Masked Node Property Prediction", input 'x' to GAT will be based on chem_feat_dim
    # C_in and O nodes will have their 'x' features masked (zeroed out).
    # C_out nodes will have their 'x' features as their chemical features.
    gat_input_x_dim = chem_feat_dim

    print(f"\nTarget Chemical Feature Dim (for C_in regression & GAT input x): {chem_feat_dim}")
    print(f"Raw Operation Feature Dim (not direct GAT input x): {op_feat_dim_raw}")
    print(f"Operation Type Classes (for O classification target): {len(op_types_map)}")

    os.makedirs(FEATURIZED_AG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DATA_OUT_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_OUT_DIR, exist_ok=True)

    train_json_files, test_json_files = train_test_split(
        all_ag_json_filenames, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nSplitting AG files: {len(train_json_files)} train, {len(test_json_files)} test.")

    # Save test file list (basenames) for evaluation script reference
    try:
        test_list_path = os.path.join(FEATURIZED_AG_OUTPUT_DIR, 'gatv2_test_files.json')
        with open(test_list_path, 'w') as f: json.dump(test_json_files, f)
        print(f"Saved test file list (basenames) to {test_list_path}")
    except Exception as e: print(f"Warning: Could not save test file list: {e}")

    for split_name, current_json_file_list, current_output_dir in [
        ("train", train_json_files, TRAIN_DATA_OUT_DIR),
        ("test", test_json_files, TEST_DATA_OUT_DIR)
    ]:
        print(f"\nProcessing {split_name} set ({len(current_json_file_list)} files) -> {current_output_dir}")
        errors_in_split = 0
        saved_in_split = 0
        for filename in tqdm(current_json_file_list, desc=f"Featurizing {split_name} AGs"):
            filepath = os.path.join(ACTION_GRAPH_INPUT_DIR, filename)
            try:
                with open(filepath, 'r') as f: ag_data_dict = json.load(f)
                ag = ActionGraph.deserialize(ag_data_dict)

                if not ag.nodes() or not ag.output_nodes or not ag.input_nodes: # Basic validity
                    # print(f"Skipping {filename}: Empty graph or missing essential node lists after deserialize.")
                    errors_in_split += 1; continue

                node_id_to_idx = {nid: i for i, nid in enumerate(ag.nodes())}
                num_graph_nodes = len(ag.nodes())

                # Initialize tensors for PyG Data object
                # x: input features to GAT (masked for C_in, O; full for C_out)
                x_features = torch.zeros((num_graph_nodes, gat_input_x_dim), dtype=torch.float)
                # y_role: 0=C_in, 1=O, 2=C_out
                y_node_roles = torch.full((num_graph_nodes,), -1, dtype=torch.long)
                # y_target_cin_features: Original chemical features for C_in nodes (regression target)
                y_cin_reg_targets = torch.zeros((num_graph_nodes, chem_feat_dim), dtype=torch.float)
                # y_target_op_types: Class index for O nodes (classification target)
                y_op_cls_targets = torch.full((num_graph_nodes,), -1, dtype=torch.long) # -1 for non-op nodes


                valid_node_processing = True
                for i, node_id in enumerate(ag.nodes()): # Iterate in the order of node_id_to_idx mapping
                    idx = node_id_to_idx[node_id] # Should be == i
                    attrs = ag.nodes[node_id]
                    node_type_attr = attrs.get('type')

                    if node_type_attr == 'chemical':
                        chem_feats = extract_chemical_features(attrs, element_map)
                        if chem_feats is None: valid_node_processing = False; break
                        
                        if node_id in ag.input_nodes:
                            y_node_roles[idx] = 0 # C_in
                            # x_features[idx, :] remains zeros (masked)
                            y_cin_reg_targets[idx, :] = torch.tensor(chem_feats, dtype=torch.float)
                        elif node_id in ag.output_nodes:
                            y_node_roles[idx] = 2 # C_out
                            x_features[idx, :] = torch.tensor(chem_feats, dtype=torch.float) # Unmasked
                        else: # Intermediate chemical - not expected for this task
                            # print(f"Warning: Intermediate chemical node {node_id} in {filename}. Skipping graph.")
                            valid_node_processing = False; break
                    
                    elif node_type_attr == 'operation':
                        y_node_roles[idx] = 1 # O
                        # x_features[idx, :] remains zeros (masked)
                        op_type_enum_attr = attrs.get('op_type')
                        op_type_idx_val = op_types_map.get(op_type_enum_attr, -1)
                        y_op_cls_targets[idx] = op_type_idx_val
                        if op_type_idx_val == -1 and op_type_enum_attr != OperationTypeEnum.unknown:
                            # print(f"Warning: Unmapped op_type {op_type_enum_attr} in {filename}")
                            pass
                    else: # Unknown node type
                        valid_node_processing = False; break
                
                if not valid_node_processing: errors_in_split += 1; continue

                # Edge index
                edge_sources, edge_targets = [], []
                for u, v in ag.edges():
                    if u in node_id_to_idx and v in node_id_to_idx:
                        edge_sources.append(node_id_to_idx[u])
                        edge_targets.append(node_id_to_idx[v])
                    else: valid_node_processing = False; break # Edge to non-existent mapped node
                if not valid_node_processing: errors_in_split += 1; continue
                edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long)


                pyg_data_instance = Data(x=x_features, edge_index=edge_index_tensor,
                                         y_role=y_node_roles,
                                         y_target_cin_features=y_cin_reg_targets,
                                         y_target_op_types=y_op_cls_targets)
                # pyg_data_instance.num_nodes = num_graph_nodes # PyG usually infers this

                output_pt_path = os.path.join(current_output_dir, filename.replace('.json', '.pt'))
                torch.save(pyg_data_instance, output_pt_path)
                saved_in_split += 1

            except Exception as e:
                # print(f"Error processing AG file {filename}: {e}", file=sys.stderr)
                # import traceback; traceback.print_exc() # For detailed debugging
                errors_in_split += 1
        
        print(f"  {split_name} set: Saved {saved_in_split} PyG Data objects. Errors/Skips: {errors_in_split}.")

    # Save metadata
    metadata_dict = {
        'element_list': element_list,
        'op_types_list_str': [e.value for e in OP_TYPES_ENUM_LIST],
        'chem_feature_dim': chem_feat_dim,
        'op_feature_dim_raw_from_conditions': op_feat_dim_raw,
        'num_op_type_classes': len(op_types_map),
        'gat_input_x_dim': gat_input_x_dim
    }
    try:
         metadata_path = os.path.join(FEATURIZED_AG_OUTPUT_DIR, 'metadata_gatv2.json')
         with open(metadata_path, 'w') as f: json.dump(metadata_dict, f, indent=2)
         print(f"\nSaved GATv2 featurization metadata to {metadata_path}")
    except Exception as e:
         print(f"Error saving metadata: {e}", file=sys.stderr); return False

    print("GATv2 featurization step completed.")
    return True

if __name__ == "__main__":
    if not run_ag_featurize_gatv2():
        sys.exit(1)
