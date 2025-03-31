import os
import torch
# from torch_geometric.data import Data # No longer needed here if using standard save/load for Data

# Note: This file might become optional if standard torch.save/load works well
# for your model state dict and list of Data objects.
# Keeping it for now for the model state dict saving/loading consistency.

def safe_save(obj, path):
    """Safely save PyTorch objects (like model state dicts)."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(obj, path)
    except Exception as e:
        print(f"Error during safe_save to {path}: {e}")
        raise

def safe_load(path):
    """Safely load PyTorch objects (like model state dicts)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found at {path}")

    try:
        # For state dicts, allow loading even if untrusted (common practice)
        # Set weights_only=False if loading more complex objects, but be careful.
        # For model state dicts, weights_only=True is safer if possible,
        # but might fail if the saved object isn't strictly weights.
        # Let's stick with weights_only=False for flexibility, assuming user trusts the source.
        obj = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        return obj
    except Exception as e:
        print(f"Error during safe_load from {path}: {e}")
        raise
