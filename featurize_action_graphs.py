import os
import json
import numpy as np
import torch
from pymatgen.core import Composition, Element
from torch_geometric.data import Data
from actiongraph import ActionGraph
from serialization_utils import safe_save

# Element properties for featurization
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
            except:
                features.append(0.0)
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)

def get_composition_features(formula):
    """Generate features for a chemical composition."""
    try:
        comp = Composition(formula)
        features = np.zeros(len(ELEMENT_PROPS), dtype=np.float32)
        total_amount = 0.0
        
        for element, amount in comp.items():
            element_features = get_element_features(element.symbol)
            features += amount * element_features
            total_amount += amount
        
        if total_amount > 0:
            features /= total_amount  # Normalize by total amount
        
        return features
    except Exception as e:
        print(f"Error featurizing formula '{formula}': {e}")
        return np.zeros(len(ELEMENT_PROPS), dtype=np.float32)

def get_operation_features(op_type, conditions):
    """Generate features for an operation node."""
    # One-hot encoding for operation type
    op_types = ['starting', 'mixing', 'shaping', 'drying', 'heating', 'quenching']
    op_type = op_type.lower() if isinstance(op_type, str) else 'unknown'
    op_type_idx = op_types.index(op_type) if op_type in op_types else len(op_types)
    op_onehot = np.zeros(len(op_types) + 1, dtype=np.float32)  # +1 for unknown
    op_onehot[op_type_idx] = 1.0
    
    # Extract temperature features
    temps = conditions.get('temperature', [])
    temp_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if temps:
        temp_values = [t for t in temps if isinstance(t, (int, float))]
        if temp_values:
            temp_features[0] = np.mean(temp_values)
            temp_features[1] = np.min(temp_values)
            temp_features[2] = np.max(temp_values)
    
    # Extract time features
    times = conditions.get('time', [])
    time_features = np.zeros(3, dtype=np.float32)  # mean, min, max
    if times:
        time_values = [t for t in times if isinstance(t, (int, float))]
        if time_values:
            time_features[0] = np.mean(time_values)
            time_features[1] = np.min(time_values)
            time_features[2] = np.max(time_values)
    
    # Combine all features
    return np.concatenate([op_onehot, temp_features, time_features])

def featurize_action_graph(ag_data):
    """Convert an action graph into a PyTorch Geometric Data object."""
    # Deserialize the action graph
    ag = ActionGraph.deserialize(ag_data)
    
    # Debug: Print node information from action graph
    print(f"Action Graph - Input nodes: {len(ag.input_nodes)}, Output nodes: {len(ag.output_nodes)}")
    print(f"Input node IDs: {ag.input_nodes}")
    print(f"Output node IDs: {ag.output_nodes}")

    # Create node feature matrix and mapping
    node_features = []
    node_id_to_idx = {}
    
    # Define max feature length for consistent padding
    chemical_feature_len = len(ELEMENT_PROPS)
    operation_feature_len = len(['starting', 'mixing', 'shaping', 'drying', 'heating', 'quenching']) + 1 + 6
    max_feature_len = max(chemical_feature_len, operation_feature_len)
    
    for idx, (node_id, attrs) in enumerate(ag.nodes(data=True)):
        node_id_to_idx[node_id] = idx
        
        # Determine node type using multiple strategies
        node_type = None
        
        # Strategy 1: Check node category memberships
        if node_id in ag.input_nodes or node_id in ag.output_nodes:
            node_type = 'chemical'
        elif node_id in ag.operation_nodes or node_id == ag.terminal_node:
            node_type = 'operation'
        
        # Strategy 2: Check node attribute dictionary for 'type'
        elif attrs.get('type') == 'chemical' or attrs.get('type') == 'operation':
            node_type = attrs.get('type')
        
        # Strategy 3: Infer from characteristic attributes
        elif any(k in attrs for k in ['formula', 'composition', 'elements']):
            node_type = 'chemical'
        elif any(k in attrs for k in ['op_type', 'conditions', 'token']):
            node_type = 'operation'
        
        # Generate features based on determined node type
        if node_type == 'chemical':
            formula = attrs.get('formula')
            if not formula and 'composition' in attrs:
                # Try to extract formula from composition if available
                try:
                    formula = str(attrs['composition'])
                except:
                    formula = "H2O"  # Default fallback
            
            if not formula:
                formula = "H2O"  # Default fallback
                
            features = get_composition_features(formula)
        elif node_type == 'operation':
            op_type = attrs.get('op_type', 'unknown')
            # Handle enum values
            if hasattr(op_type, 'value'):
                op_type = op_type.value
            op_type = str(op_type)
            
            conditions = attrs.get('conditions', {})
            features = get_operation_features(op_type, conditions)
        else:
            # Use zero vector as fallback
            print(f"Warning: Undefined node type for {node_id}. Using zero vector.")
            features = np.zeros(max_feature_len, dtype=np.float32)
        
        # Ensure consistent feature length
        if len(features) < max_feature_len:
            # Pad with zeros
            padded = np.zeros(max_feature_len, dtype=np.float32)
            padded[:len(features)] = features
            features = padded
        elif len(features) > max_feature_len:
            # Truncate
            features = features[:max_feature_len]
        
        node_features.append(features)
    
    # Debug: Print mapping information
    print(f"Node ID to index mapping: {node_id_to_idx}")
    
    # Create edge index matrix
    edge_index = []
    for src, tgt in ag.edges():
        if src in node_id_to_idx and tgt in node_id_to_idx:
            src_idx = node_id_to_idx[src]
            tgt_idx = node_id_to_idx[tgt]
            edge_index.append([src_idx, tgt_idx])
    
    # Convert to tensors
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.zeros((2, 0), dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    
    # IMPORTANT: Set metadata attributes for node types using the mapping
    data.input_nodes = [node_id_to_idx[node_id] for node_id in ag.input_nodes if node_id in node_id_to_idx]
    data.output_nodes = [node_id_to_idx[node_id] for node_id in ag.output_nodes if node_id in node_id_to_idx]
    data.operation_nodes = [node_id_to_idx[node_id] for node_id in ag.operation_nodes if node_id in node_id_to_idx]
    
    # Debug: Verify attributes were set correctly
    print(f"PyG Data - Input nodes: {len(data.input_nodes)}, Output nodes: {len(data.output_nodes)}")
    print(f"Input node indices: {data.input_nodes}")
    print(f"Output node indices: {data.output_nodes}")
    
    return data

def featurize_action_graphs():
    """Featurize all action graphs and save as PyTorch Geometric data objects."""
    input_dir = "Data/ag-data/"
    output_dir = "Data/ag-data/featurized/"
    os.makedirs(output_dir, exist_ok=True)
    
    ag_files = [f for f in os.listdir(input_dir) if f.endswith('_ag.json')]
    all_data = []
    success_count = 0
    
    for ag_file in ag_files:
        try:
            with open(os.path.join(input_dir, ag_file), 'r') as f:
                ag_data = json.load(f)
            
            data = featurize_action_graph(ag_data)
            all_data.append(data)
            
            # Save individual featurized graph using safe save
            output_file = os.path.join(output_dir, f"{os.path.splitext(ag_file)[0]}_feat.pt")
            safe_save(data, output_file)
            
            success_count += 1
            if success_count % 20 == 0:
                print(f"Successfully featurized {success_count} action graphs")
        except Exception as e:
            print(f"Error featurizing {ag_file}: {e}")
    
    # Save the complete dataset
    if all_data:
        safe_save(all_data, os.path.join(output_dir, 'all_data.pt'))
        print(f"Featurized {len(all_data)} action graphs")
    else:
        print("No action graphs were successfully featurized")

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
            print("\nNode types after deserialization:")
            for node_id in ag.nodes():
                print(f"  Node {node_id}: type={ag.nodes[node_id].get('type')}")
            
            # Featurize and check feature shapes
            data = featurize_action_graph(ag_data)
            print("\nFeature summary:")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Node feature sample (first node): {data.x[0][:5]}")  # First 5 values
            
            # Check if any features are non-zero
            non_zero = (data.x != 0).any(dim=1).sum().item()
            print(f"  Nodes with non-zero features: {non_zero}/{data.x.shape[0]}")
            
            print("\nTest successful!")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    test_featurization_detailed()

if __name__ == "__main__":
    main()
