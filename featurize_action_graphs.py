import os
import json
import numpy as np
import torch
from pymatgen.core import Composition, Element
from torch_geometric.data import Data
from actiongraph import ActionGraph

ELEMENT_PROPS = [
    'atomic_mass', 'atomic_radius', 'atomic_volume',
    'electronegativity', 'electron_affinity', 'ionization_energy'
]

CHEMICAL_BASE_FEATURES = len(ELEMENT_PROPS)
OP_TYPES = ['starting', 'mixing', 'shaping', 'drying', 'heating', 'quenching']
OPERATION_BASE_FEATURES = len(OP_TYPES) + 1 + 3 + 3 # op_onehot + temp(3) + time(3)
INDICATOR_FEATURES = 2 # is_chemical, is_operation
MAX_FEATURE_LEN = max(CHEMICAL_BASE_FEATURES, OPERATION_BASE_FEATURES) + INDICATOR_FEATURES

# Helper to create zero vector of correct final length
def get_zero_feature_vector():
    return np.zeros(MAX_FEATURE_LEN, dtype=np.float32)

def get_element_features(element_symbol):
    """Extract element-level features."""
    try:
        element = Element(element_symbol)
        features = []
        for prop in ELEMENT_PROPS:
            try:
                value = getattr(element, prop)
                # Handle potential None or non-numeric values gracefully
                if value is None or not isinstance(value, (int, float, np.number)):
                    features.append(0.0)
                else:
                    features.append(float(value))
            except AttributeError:
                features.append(0.0)
        return np.array(features, dtype=np.float32)
    except KeyError:
         print(f"Warning: Element {element_symbol} not found. Using zero vector for element.")
         # Return zero vector matching ELEMENT_PROPS length
         return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)
    except Exception as e:
        print(f"Warning: Error getting features for element {element_symbol}: {e}. Using zero vector for element.")
        return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)


def get_composition_features(formula):
    """Generate features for a chemical composition."""
    try:
        comp = Composition(formula)
        elements_present = comp.elements
        if not elements_present: # Handle empty composition if formula is invalid
             print(f"Warning: Could not parse composition from formula '{formula}'. Using zero vector.")
             return get_zero_feature_vector() # Use helper for correct length

        base_features = np.zeros(CHEMICAL_BASE_FEATURES, dtype=np.float32)
        total_amount = 0.0

        for element in elements_present:
            amount = comp.get_atomic_fraction(element) * comp.num_atoms # Use actual amount if possible, fallback to fraction
            if amount == 0: amount = comp.get_atomic_fraction(element) # Fallback if num_atoms is weird

            element_features = get_element_features(element.symbol)
            base_features += amount * element_features
            total_amount += amount

        if total_amount > 0:
            base_features /= total_amount  # Normalize by total amount

        # Pad base features and add indicators
        padded_features = np.zeros(MAX_FEATURE_LEN, dtype=np.float32)
        padded_features[:CHEMICAL_BASE_FEATURES] = base_features
        padded_features[MAX_FEATURE_LEN - INDICATOR_FEATURES] = 1.0 # is_chemical = 1
        padded_features[MAX_FEATURE_LEN - INDICATOR_FEATURES + 1] = 0.0 # is_operation = 0
        return padded_features

    except Exception as e:
        print(f"Error featurizing formula '{formula}': {e}")
        return get_zero_feature_vector() # Use helper for correct length

def get_operation_features(op_type, conditions):
    """Generate features for an operation node."""
    # One-hot encoding for operation type
    # Ensure op_type is a string and handle potential Enum values
    if hasattr(op_type, 'value'):
        op_type_str = str(op_type.value).lower()
    else:
        op_type_str = str(op_type).lower()

    # Find index or use 'unknown' index
    try:
        op_type_idx = OP_TYPES.index(op_type_str)
    except ValueError:
        op_type_idx = len(OP_TYPES) # Index for unknown

    op_onehot = np.zeros(len(OP_TYPES) + 1, dtype=np.float32)  # +1 for unknown
    op_onehot[op_type_idx] = 1.0

    # Extract temperature features
    temps = conditions.get('temperature', [])
    temp_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if temps:
        # Ensure temps are numeric before processing
        temp_values = [float(t) for t in temps if isinstance(t, (int, float, np.number)) and np.isfinite(t)]
        if temp_values:
            # Apply basic scaling and handle potential single values
            temp_features[0] = np.mean(temp_values) / 1000.0
            temp_features[1] = np.min(temp_values) / 1000.0
            temp_features[2] = np.max(temp_values) / 1000.0

    # Extract time features
    times = conditions.get('time', [])
    time_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if times:
        # Ensure times are numeric
        time_values = [float(t) for t in times if isinstance(t, (int, float, np.number)) and np.isfinite(t)]
        if time_values:
            # Apply basic scaling and handle potential single values
            time_features[0] = np.mean(time_values) / 3600.0 # Scale to hours
            time_features[1] = np.min(time_values) / 3600.0
            time_features[2] = np.max(time_values) / 3600.0

    # Concatenate operation-specific features
    base_features = np.concatenate([op_onehot, temp_features, time_features])

    # Pad base features and add indicators
    padded_features = np.zeros(MAX_FEATURE_LEN, dtype=np.float32)
    current_len = len(base_features)
    padded_features[:current_len] = base_features # Place features at the start
    padded_features[MAX_FEATURE_LEN - INDICATOR_FEATURES] = 0.0 # is_chemical = 0
    padded_features[MAX_FEATURE_LEN - INDICATOR_FEATURES + 1] = 1.0 # is_operation = 1
    return padded_features


def featurize_action_graph(ag_data):
    """Convert an action graph into a PyTorch Geometric Data object."""
    ag = ActionGraph.deserialize(ag_data)
    node_id_to_idx = {node_id: i for i, node_id in enumerate(ag.nodes())}
    num_nodes = len(node_id_to_idx)

    if num_nodes == 0: # Handle empty graphs early
        print("Warning: Empty graph encountered after deserialization.")
        # Return an empty Data object or handle as needed
        return Data(x=torch.empty((0, MAX_FEATURE_LEN), dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    y=torch.empty(0, dtype=torch.float),
                    is_output=torch.empty(0, dtype=torch.float))


    node_features_list = []
    y = torch.zeros(num_nodes, dtype=torch.float)
    is_output = torch.zeros(num_nodes, dtype=torch.float)

    # Ensure input/output nodes are valid node IDs present in the graph
    valid_input_nodes = {node_id for node_id in ag.input_nodes if node_id in node_id_to_idx}
    valid_output_nodes = {node_id for node_id in ag.output_nodes if node_id in node_id_to_idx}

    for node_id, idx in node_id_to_idx.items():
        attrs = ag.nodes[node_id]
        node_type = attrs.get('type')
        features = None

        if node_type == 'chemical':
            formula = attrs.get('formula', 'H') # Use a simple default like H if formula missing
            features = get_composition_features(formula)
            if node_id in valid_output_nodes:
                is_output[idx] = 1.0
            if node_id in valid_input_nodes:
                y[idx] = 1.0 # Mark input nodes as positive targets
        elif node_type == 'operation':
            op_type = attrs.get('op_type', 'unknown')
            conditions = attrs.get('conditions', {})
            features = get_operation_features(op_type, conditions)
        # Handle terminal node which might be marked as operation
        elif node_id == ag.terminal_node and 'op_type' in attrs:
            op_type = attrs.get('op_type', 'unknown')
            conditions = attrs.get('conditions', {})
            features = get_operation_features(op_type, conditions)
        else:
            print(f"Warning: Undefined or unexpected node attributes for node {node_id}. Attrs: {attrs}. Using zero vector.")
            features = get_zero_feature_vector()

        # Ensure features are not None before appending
        if features is None:
             print(f"Critical Warning: Features are None for node {node_id}. Using zero vector.")
             features = get_zero_feature_vector()
        # Check feature length consistency
        if features.shape[0] != MAX_FEATURE_LEN:
             print(f"Critical Warning: Feature length mismatch for node {node_id}. Expected {MAX_FEATURE_LEN}, got {features.shape[0]}. Fixing...")
             # Attempt to fix by padding/truncating
             fixed_features = np.zeros(MAX_FEATURE_LEN, dtype=np.float32)
             len_to_copy = min(features.shape[0], MAX_FEATURE_LEN)
             fixed_features[:len_to_copy] = features[:len_to_copy]
             features = fixed_features

        node_features_list.append(features)

    # Create edge index matrix
    edge_index = []
    for src, tgt in ag.edges():
        if src in node_id_to_idx and tgt in node_id_to_idx:
            src_idx = node_id_to_idx[src]
            tgt_idx = node_id_to_idx[tgt]
            edge_index.append([src_idx, tgt_idx])

    # Convert features and edge index to tensors
    x = torch.tensor(np.array(node_features_list), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.empty((2, 0), dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y, is_output=is_output)

    # Store original indices for reference (careful with batching)
    data.input_node_indices = [node_id_to_idx[node_id] for node_id in valid_input_nodes]
    data.output_node_indices = [node_id_to_idx[node_id] for node_id in valid_output_nodes]

    # Add structural info if potentially useful (optional)
    # Calculate in-degrees (can be done here or dynamically in model/eval)
    # in_degree = torch.zeros(num_nodes, dtype=torch.long)
    # if data.num_edges > 0:
    #     in_degree.scatter_add_(0, data.edge_index[1], torch.ones(data.num_edges, dtype=torch.long))
    # data.in_degree = in_degree # Example: adding in_degree

    # Check if graph has input nodes marked in y
    if y.sum() == 0 and valid_input_nodes:
        print(f"Warning: ActionGraph reported {len(valid_input_nodes)} input nodes, but none were marked in y tensor. Mapping: {node_id_to_idx}, Valid Inputs: {valid_input_nodes}")
    if is_output.sum() == 0 and valid_output_nodes:
         print(f"Warning: ActionGraph reported {len(valid_output_nodes)} output nodes, but none were marked in is_output tensor. Mapping: {node_id_to_idx}, Valid Outputs: {valid_output_nodes}")

    return data


# --- Keep `featurize_action_graphs` function mostly the same ---
# (It calls the modified `featurize_action_graph` function)
# Add a check after calling featurize_action_graph:
def featurize_action_graphs():
    # ... (setup input/output dirs, etc.) ...
    input_dir = "Data/ag-data/"
    output_dir = "Data/ag-data/featurized/"
    os.makedirs(output_dir, exist_ok=True)

    ag_files = [f for f in os.listdir(input_dir) if f.endswith('_ag.json')]
    all_data = []
    success_count = 0
    error_count = 0
    no_input_nodes_count = 0
    no_output_nodes_count = 0 # Track missing outputs too

    print(f"Found {len(ag_files)} action graph JSON files in {input_dir}")

    for i, ag_file in enumerate(ag_files):
        file_path = os.path.join(input_dir, ag_file)
        try:
            with open(file_path, 'r') as f:
                ag_data = json.load(f)

            data = featurize_action_graph(ag_data) # Call the modified function

            # --- More robust validation after featurization ---
            if not isinstance(data, Data) or data.x is None or data.edge_index is None or data.y is None or data.is_output is None:
                 print(f"Skipping {ag_file}: Featurization failed or returned invalid object.")
                 error_count += 1
                 continue
            if data.num_nodes == 0:
                # This case might have been handled inside featurize_action_graph, but double check
                print(f"Skipping {ag_file}: Featurization resulted in zero nodes.")
                error_count += 1
                continue
            if data.x.shape[1] != MAX_FEATURE_LEN:
                 print(f"Skipping {ag_file}: Incorrect feature dimension ({data.x.shape[1]} vs {MAX_FEATURE_LEN}).")
                 error_count += 1
                 continue
            if data.y.sum() == 0:
                 # Only count as "no input" if the original AG *should* have had inputs
                 ag_check = ActionGraph.deserialize(ag_data) # Re-deserialize for check
                 if ag_check.input_nodes:
                     no_input_nodes_count += 1
                     print(f"Warning for {ag_file}: No input nodes marked in 'y' tensor, though AG reported inputs.")
                 # Decide whether to skip these graphs entirely for training
                 # continue # Uncomment to skip graphs with no marked inputs
            if data.is_output.sum() == 0:
                 ag_check = ActionGraph.deserialize(ag_data) # Re-deserialize for check
                 if ag_check.output_nodes:
                     no_output_nodes_count +=1
                     print(f"Warning for {ag_file}: No output nodes marked in 'is_output' tensor, though AG reported outputs.")
                 # continue # Decide if these should be skipped
            # --- End validation ---

            all_data.append(data)
            success_count += 1
            if success_count % 100 == 0:
                print(f"Successfully processed {success_count}/{len(ag_files)} action graphs...")

        except json.JSONDecodeError:
            print(f"Error decoding JSON for {ag_file}. Skipping.")
            error_count += 1
        except FileNotFoundError:
             print(f"File not found: {file_path}. Skipping.")
             error_count += 1
        except Exception as e:
            print(f"Error processing {ag_file}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # ... (Save data as before) ...
    output_path = os.path.join(output_dir, 'all_data.pt')
    if all_data:
        print(f"\nSaving {len(all_data)} featurized graphs to {output_path}")
        # Ensure MAX_FEATURE_LEN is saved or accessible for model loading
        config = {'max_feature_len': MAX_FEATURE_LEN}
        torch.save({'config': config, 'data': all_data}, output_path)
        print("Save successful.")
    else:
        print("No action graphs were successfully featurized to save.")

    print(f"\nFeaturization Summary:")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped due to errors: {error_count}")
    print(f"  Graphs missing marked input nodes (y=0): {no_input_nodes_count}")
    print(f"  Graphs missing marked output nodes (is_output=0): {no_output_nodes_count}")


# Keep `test_featurization_detailed` and `main` (adapt if needed)
if __name__ == "__main__":
    # main() # Call featurization
    # Example: Run detailed test if needed
    # test_featurization_detailed(num_files=2)
    featurize_action_graphs()
