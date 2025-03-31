import os
import torch
from torch_geometric.data import Data, Dataset
import torch.serialization

def safe_save(obj, path):
    """Save PyTorch Geometric data safely with explicit attribute preservation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if isinstance(obj, (list, tuple)) and all(isinstance(x, Data) for x in obj):
        # For PyG data lists, convert to dictionaries that preserve metadata
        processed_data = []
        for data in obj:
            # Convert to dict representation with explicit attributes
            data_dict = {
                'x': data.x,
                'edge_index': data.edge_index,
                'input_nodes': getattr(data, 'input_nodes', []),
                'output_nodes': getattr(data, 'output_nodes', []),
                'operation_nodes': getattr(data, 'operation_nodes', [])
            }
            processed_data.append(data_dict)
        torch.save(processed_data, path)
    else:
        torch.save(obj, path)

def safe_load(path):
    """Load PyTorch Geometric data safely, ensuring metadata is restored."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")
    
    try:
        raw_data = torch.load(path, weights_only=False)
        
        # Check if this is a list of dictionaries (our saved format)
        if isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict) and 'x' in raw_data[0]:
            result = []
            for item_dict in raw_data:
                # Reconstruct Data object with all attributes
                data = Data(
                    x=item_dict['x'],
                    edge_index=item_dict['edge_index']
                )
                # Restore metadata attributes
                data.input_nodes = item_dict.get('input_nodes', [])
                data.output_nodes = item_dict.get('output_nodes', [])
                data.operation_nodes = item_dict.get('operation_nodes', [])
                result.append(data)
            return result
        
        return raw_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
