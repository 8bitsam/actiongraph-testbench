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

def get_element_features(element_symbol):
    """Extract element-level features."""
    try:
        element = Element(element_symbol)
        features = []
        for prop in ELEMENT_PROPS:
            try:
                value = getattr(element, prop)
                features.append(float(value) if value is not None else 0.0)
            except AttributeError: # Catch specific error
                features.append(0.0)
        return np.array(features, dtype=np.float32)
    except KeyError: # Element symbol not found
         print(f"Warning: Element {element_symbol} not found. Using zero vector.")
         return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)
    except Exception as e:
        print(f"Warning: Error getting features for element {element_symbol}: {e}. Using zero vector.")
        return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)


def get_composition_features(formula):
    """Generate features for a chemical composition."""
    try:
        comp = Composition(formula)
        if not comp.elements: # Handle empty composition if formula is invalid
             print(f"Warning: Could not parse composition from formula '{formula}'. Using zero vector.")
             return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)

        features = np.zeros(len(ELEMENT_PROPS), dtype=np.float32)
        total_amount = 0.0

        for element, amount in comp.items():
            element_features = get_element_features(element.symbol)
            features += amount * element_features
            total_amount += amount

        if total_amount > 0:
            features /= total_amount  # Normalize by total amount

        # Add a feature indicating it's a chemical node (optional, but can help)
        # chemical_indicator = np.array([1.0], dtype=np.float32)
        # return np.concatenate([features, chemical_indicator])
        return features

    except Exception as e:
        print(f"Error featurizing formula '{formula}': {e}")
        return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)

def get_operation_features(op_type, conditions):
    """Generate features for an operation node."""
    # One-hot encoding for operation type
    op_types = ['starting', 'mixing', 'shaping', 'drying', 'heating', 'quenching']
    # Ensure op_type is a string and handle potential Enum values
    if hasattr(op_type, 'value'):
        op_type = str(op_type.value).lower()
    else:
        op_type = str(op_type).lower()

    op_type_idx = op_types.index(op_type) if op_type in op_types else len(op_types)
    op_onehot = np.zeros(len(op_types) + 1, dtype=np.float32)  # +1 for unknown
    op_onehot[op_type_idx] = 1.0

    # Extract temperature features
    temps = conditions.get('temperature', [])
    temp_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if temps:
        # Ensure temps are numeric before processing
        temp_values = [float(t) for t in temps if isinstance(t, (int, float))]
        if temp_values:
            temp_features[0] = np.mean(temp_values) / 1000.0 # Basic scaling
            temp_features[1] = np.min(temp_values) / 1000.0 # Basic scaling
            temp_features[2] = np.max(temp_values) / 1000.0 # Basic scaling

    # Extract time features
    times = conditions.get('time', [])
    time_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if times:
        # Ensure times are numeric
        time_values = [float(t) for t in times if isinstance(t, (int, float))]
        if time_values:
            time_features[0] = np.mean(time_values) / 3600.0 # Basic scaling (to hours)
            time_features[1] = np.min(time_values) / 3600.0 # Basic scaling
            time_features[2] = np.max(time_values) / 3600.0 # Basic scaling

    # Add a feature indicating it's an operation node (optional)
    # operation_indicator = np.array([1.0], dtype=np.float32)
    # return np.concatenate([op_onehot, temp_features, time_features, operation_indicator])
    return np.concatenate([op_onehot, temp_features, time_features])

def featurize_action_graph(ag_data):
    """Convert an action graph into a PyTorch Geometric Data object."""
    # Deserialize the action graph
    ag = ActionGraph.deserialize(ag_data)

    # Debug: Print node information from action graph
    # print(f"Action Graph - Input nodes: {len(ag.input_nodes)}, Output nodes: {len(ag.output_nodes)}")
    # print(f"Input node IDs: {ag.input_nodes}")
    # print(f"Output node IDs: {ag.output_nodes}")

    # Create node feature matrix and mapping
    node_features = []
    node_id_to_idx = {node_id: i for i, node_id in enumerate(ag.nodes())}
    num_nodes = len(node_id_to_idx)

    # Define feature lengths
    chemical_feature_len = len(ELEMENT_PROPS)
    operation_feature_len = len(['starting', 'mixing', 'shaping', 'drying', 'heating', 'quenching']) + 1 + 6
    max_feature_len = max(chemical_feature_len, operation_feature_len)

    # Create target tensors
    y = torch.zeros(num_nodes, dtype=torch.float)
    is_output = torch.zeros(num_nodes, dtype=torch.float) # Use float for easier batching/loss

    for node_id, idx in node_id_to_idx.items():
        attrs = ag.nodes[node_id]

        # Determine node type (simplified logic)
        node_type = attrs.get('type')
        if node_type == 'chemical':
            formula = attrs.get('formula', 'H2O') # Default fallback
            features = get_composition_features(formula)
            feature_len = chemical_feature_len
            if node_id in ag.output_nodes:
                is_output[idx] = 1.0
            if node_id in ag.input_nodes:
                y[idx] = 1.0 # Mark input nodes as positive targets
        elif node_type == 'operation':
            op_type = attrs.get('op_type', 'unknown')
            conditions = attrs.get('conditions', {})
            features = get_operation_features(op_type, conditions)
            feature_len = operation_feature_len
        else:
            # Handle nodes potentially missing 'type' but identifiable (like terminal)
            if node_id == ag.terminal_node and 'op_type' in attrs:
                 op_type = attrs.get('op_type', 'unknown')
                 conditions = attrs.get('conditions', {})
                 features = get_operation_features(op_type, conditions)
                 feature_len = operation_feature_len
            else:
                print(f"Warning: Undefined or unexpected node type for {node_id}. Type: {node_type}. Using zero vector.")
                features = np.zeros(max_feature_len, dtype=np.float32)
                feature_len = max_feature_len


        # Ensure consistent feature length by padding
        if feature_len < max_feature_len:
            padded_features = np.zeros(max_feature_len, dtype=np.float32)
            padded_features[:feature_len] = features
            node_features.append(padded_features)
        elif feature_len > max_feature_len:
             node_features.append(features[:max_feature_len]) # Truncate if longer
        else:
            node_features.append(features)

    # Debug: Print mapping information
    # print(f"Node ID to index mapping: {node_id_to_idx}")

    # Create edge index matrix
    edge_index = []
    for src, tgt in ag.edges():
        if src in node_id_to_idx and tgt in node_id_to_idx:
            src_idx = node_id_to_idx[src]
            tgt_idx = node_id_to_idx[tgt]
            edge_index.append([src_idx, tgt_idx])

    # Convert features and edge index to tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.zeros((2, 0), dtype=torch.long)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y, is_output=is_output)

    # Add original node IDs for potential debugging or analysis later
    # Note: These lists won't batch nicely, use 'y' and 'is_output' for training/evaluation logic
    data.input_node_indices = [node_id_to_idx[node_id] for node_id in ag.input_nodes if node_id in node_id_to_idx]
    data.output_node_indices = [node_id_to_idx[node_id] for node_id in ag.output_nodes if node_id in node_id_to_idx]
    data.operation_node_indices = [node_id_to_idx[node_id] for node_id in ag.operation_nodes if node_id in node_id_to_idx]


    # Debug: Verify attributes were set correctly
    # print(f"PyG Data - Input nodes (y==1): {data.y.sum().item()}, Output nodes (is_output==1): {data.is_output.sum().item()}")
    # print(f"Input node indices (y==1): {torch.where(data.y == 1)[0].tolist()}")
    # print(f"Output node indices (is_output==1): {torch.where(data.is_output == 1)[0].tolist()}")

    # Check if graph has input nodes marked in y
    if y.sum() == 0 and ag.input_nodes:
        print(f"Warning: ActionGraph reported {len(ag.input_nodes)} input nodes ({ag.input_nodes}), but none were marked in y tensor. Mapping: {node_id_to_idx}")
    if is_output.sum() == 0 and ag.output_nodes:
         print(f"Warning: ActionGraph reported {len(ag.output_nodes)} output nodes ({ag.output_nodes}), but none were marked in is_output tensor. Mapping: {node_id_to_idx}")

    return data

def featurize_action_graphs():
    """Featurize all action graphs and save as PyTorch Geometric data objects."""
    input_dir = "Data/ag-data/"
    output_dir = "Data/ag-data/featurized/"
    os.makedirs(output_dir, exist_ok=True)

    ag_files = [f for f in os.listdir(input_dir) if f.endswith('_ag.json')]
    all_data = []
    success_count = 0
    error_count = 0
    no_input_nodes_count = 0

    print(f"Found {len(ag_files)} action graph JSON files in {input_dir}")

    for i, ag_file in enumerate(ag_files):
        file_path = os.path.join(input_dir, ag_file)
        # print(f"\nProcessing file {i+1}/{len(ag_files)}: {ag_file}")
        try:
            with open(file_path, 'r') as f:
                ag_data = json.load(f)

            data = featurize_action_graph(ag_data)

            # Basic validation before adding
            if data.x is None or data.edge_index is None or data.y is None or data.is_output is None:
                 print(f"Skipping {ag_file}: Featurization resulted in None tensors.")
                 error_count += 1
                 continue
            if data.num_nodes == 0:
                print(f"Skipping {ag_file}: Featurization resulted in zero nodes.")
                error_count += 1
                continue
            if data.y.sum() == 0:
                 # print(f"Skipping {ag_file}: No input nodes found/marked during featurization.")
                 no_input_nodes_count += 1
                 # Decide whether to skip these graphs entirely for training
                 # For now, let's include them but be aware of the count
                 # continue
            if data.is_output.sum() == 0:
                 print(f"Warning for {ag_file}: No output nodes found/marked during featurization.")
                 # continue # Decide if these should be skipped

            all_data.append(data)
            success_count += 1
            if success_count % 100 == 0:
                print(f"Successfully featurized {success_count} action graphs...")

        except json.JSONDecodeError:
            print(f"Error decoding JSON for {ag_file}. Skipping.")
            error_count += 1
        except FileNotFoundError:
             print(f"File not found: {file_path}. Skipping.")
             error_count += 1
        except Exception as e:
            print(f"Error featurizing {ag_file}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # Save the complete dataset using standard torch.save
    output_path = os.path.join(output_dir, 'all_data.pt')
    if all_data:
        print(f"\nSaving {len(all_data)} featurized graphs to {output_path}")
        try:
            torch.save(all_data, output_path)
            print("Save successful.")
        except Exception as e:
            print(f"Error saving dataset: {e}")
    else:
        print("No action graphs were successfully featurized to save.")

    print(f"\nFeaturization Summary:")
    print(f"  Successfully featurized: {success_count}")
    print(f"  Skipped due to errors: {error_count}")
    print(f"  Graphs with no input nodes marked: {no_input_nodes_count} (These might be included or excluded depending on checks in training)")


def test_featurization_detailed(num_files=2):
    """Test the featurization with detailed output."""
    input_dir = "Data/ag-data/"
    ag_files = [f for f in os.listdir(input_dir) if f.endswith('_ag.json')][:num_files]

    for ag_file in ag_files:
        print(f"\nDetailed test of {ag_file}")

        try:
            with open(os.path.join(input_dir, ag_file), 'r') as f:
                ag_data = json.load(f)

            # Deserialize and check node types
            ag = ActionGraph.deserialize(ag_data)
            print("\nActionGraph Node Info:")
            print(f"  Input node IDs: {ag.input_nodes}")
            print(f"  Output node IDs: {ag.output_nodes}")
            for node_id in ag.nodes():
                print(f"  Node {node_id}: type={ag.nodes[node_id].get('type')}")

            # Featurize and check resulting Data object
            data = featurize_action_graph(ag_data)
            print("\nPyG Data Object Summary:")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge index shape: {data.edge_index.shape}")
            print(f"  Target tensor (y) shape: {data.y.shape}")
            print(f"  Is_output tensor shape: {data.is_output.shape}")
            print(f"  Num nodes: {data.num_nodes}")
            print(f"  Num edges: {data.num_edges}")

            print(f"\n  Node feature sample (first node): {data.x[0][:10]}")  # First 10 values
            non_zero_features = (data.x != 0).any(dim=1).sum().item()
            print(f"  Nodes with non-zero features: {non_zero_features}/{data.x.shape[0]}")

            print(f"\n  Target tensor (y) values: {data.y}")
            print(f"  Indices where y == 1: {torch.where(data.y == 1)[0].tolist()}")
            print(f"  Is_output tensor values: {data.is_output}")
            print(f"  Indices where is_output == 1: {torch.where(data.is_output == 1)[0].tolist()}")

            print(f"\n  Original Input node indices: {data.input_node_indices}")
            print(f"  Original Output node indices: {data.output_node_indices}")

            print("\nTest successful!")
        except Exception as e:
            print(f"  Error during detailed test of {ag_file}: {e}")
            import traceback
            traceback.print_exc()

def main():
    # Run the main featurization process
    featurize_action_graphs()
    # Optionally run the detailed test on a few files
    # print("\n" + "="*20 + " Running Detailed Test " + "="*20)
    # test_featurization_detailed(num_files=2)


if __name__ == "__main__":
    main()
