# featurize.py

import json
import numpy as np
from pymatgen.core import Composition, Element
import warnings
import os
import time
import sys
# Removed hashlib

# --- Configuration ---
# Assume script is run from the parent directory of Data/
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
# Default output dir for standalone run or final output
DEFAULT_FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-weighted/")

ELEMENT_PROPS = [
    'atomic_mass', 'atomic_radius', 'atomic_volume',
    'electronegativity', 'electron_affinity', 'ionization_energy'
]
NUM_ELEMENT_PROPS = len(ELEMENT_PROPS)

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Helper Functions ---
def load_data_from_directory(dir_path):
    """Loads synthesis data from all .json files in a directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
        return None
    all_data = []
    print(f"Scanning directory: {dir_path}")
    start_time = time.time()
    loaded_count = 0
    error_count = 0
    all_files = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
             if filename.lower().endswith('.json'):
                  all_files.append(os.path.join(root, filename))

    print(f"Found {len(all_files)} potential JSON files. Loading...")
    for filepath in all_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data_entry = json.load(f)
                if isinstance(data_entry, dict):
                    data_entry['_source_file'] = os.path.basename(filepath)
                    all_data.append(data_entry)
                    loaded_count += 1
                else: error_count += 1
        except Exception: error_count += 1
    end_time = time.time()
    print(f"Finished loading. Loaded {loaded_count}, skipped {error_count} files. Took {end_time - start_time:.2f}s.")
    if not all_data:
        print(f"Error: No valid JSON data loaded from {dir_path}", file=sys.stderr)
        return None
    return all_data

def get_element_map(data):
    """Creates a mapping from element symbol to feature vector index."""
    all_elements = set()
    parse_error_count = 0
    for entry in data:
        if not isinstance(entry, dict) or 'target' not in entry or 'material_formula' not in entry['target']: continue
        target_formula = entry['target'].get('material_formula')
        if not target_formula or not isinstance(target_formula, str): continue
        try:
            comp = Composition(target_formula)
            if comp.num_atoms >= 1:
                for el in comp.elements: all_elements.add(el.symbol)
        except Exception: parse_error_count +=1
    # print(f"Parsed {len(data) - parse_error_count} formulas, {parse_error_count} errors.")
    try: # Add common elements
        common_elements = {el.symbol for el in Element}
        all_elements.update(common_elements)
    except Exception as e: print(f"Warning: Could not get all Pymatgen elements: {e}", file=sys.stderr)
    sorted_elements = sorted(list(all_elements))
    if not sorted_elements:
         print("Error: No valid elements found.", file=sys.stderr)
         return None
    return {element: i for i, element in enumerate(sorted_elements)}

def featurize_target(formula, element_map, element_props_list, z_weights):
    """Generates features: element fraction vector + weighted averaged properties."""
    if not formula or not isinstance(formula, str): return None
    if z_weights is None or len(z_weights) != len(element_props_list): return None

    try:
        comp = Composition(formula)
        if comp.num_atoms < 1: return None
        el_amt_dict = comp.get_el_amt_dict()
        total_atoms = comp.num_atoms

        comp_vector = np.zeros(len(element_map))
        for el_sym, amount in el_amt_dict.items():
            if el_sym in element_map: comp_vector[element_map[el_sym]] = amount / total_atoms

        prop_vector = np.zeros(len(element_props_list))
        elements_in_comp_symbols = list(el_amt_dict.keys())
        for i, prop_name in enumerate(element_props_list):
            weighted_sum, total_amount_for_prop, prop_available = 0.0, 0.0, False
            for el_symbol in elements_in_comp_symbols:
                 amount = el_amt_dict[el_symbol]
                 try:
                     el_obj = Element(el_symbol)
                     prop_value = getattr(el_obj, prop_name, None)
                     if prop_value is not None and isinstance(prop_value, (int, float)):
                         weighted_sum += prop_value * amount
                         total_amount_for_prop += amount
                         prop_available = True
                 except Exception: pass
            if prop_available and total_amount_for_prop > 0:
                prop_vector[i] = weighted_sum / total_amount_for_prop

        weighted_prop_vector = prop_vector * z_weights
        feature_vector = np.concatenate((comp_vector, weighted_prop_vector))
        return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None

def get_canonical_recipe_string(entry):
    """Creates a canonical, sortable string representation for exact deduplication."""
    try:
        target_formula = entry.get('target', {}).get('material_formula', '')
        precursors = entry.get('precursors', [])
        operations = entry.get('operations', [])
        prec_formulas = sorted([p.get('material_formula', '') for p in precursors if isinstance(p, dict)])
        op_strings = sorted([json.dumps(op, sort_keys=True) for op in operations if isinstance(op, dict)])
        canonical_string = json.dumps({
            "target": target_formula,
            "precursors": prec_formulas,
            "operations": op_strings
        }, sort_keys=True)
        return canonical_string
    except Exception:
        return f"SOURCE_FILE:{entry.get('_source_file', os.urandom(16))}"

def run_featurize(z_weights=None, output_dir=DEFAULT_FEATURIZED_DATA_DIR):
    """Loads raw data, deduplicates, featurizes (with weighting), and saves."""
    print("--- Running Featurization Step ---")

    if z_weights is None:
        print("Using default weights (all 1.0) for properties.")
        z_weights = np.ones(NUM_ELEMENT_PROPS)
    elif len(z_weights) != NUM_ELEMENT_PROPS:
        print(f"Error: z_weights length mismatch.", file=sys.stderr)
        return False
    z_weights_array = np.array(z_weights)
    print(f"Using z_weights: {z_weights_array.tolist()}")

    raw_data = load_data_from_directory(RAW_DATA_DIR)
    if not raw_data: return False

    print("\nDeduplicating entries...")
    start_dedup_time = time.time()
    unique_recipes = {}
    duplicate_count = 0
    for entry in raw_data:
        canonical_string = get_canonical_recipe_string(entry)
        if canonical_string not in unique_recipes:
            unique_recipes[canonical_string] = entry
        else: duplicate_count += 1
    deduplicated_data = list(unique_recipes.values())
    end_dedup_time = time.time()
    print(f"Deduplication took {end_dedup_time - start_dedup_time:.2f}s. Removed {duplicate_count} duplicates.")
    print(f"Proceeding with {len(deduplicated_data)} unique entries.")
    if not deduplicated_data: return False

    print("\nCreating element map...")
    element_map = get_element_map(deduplicated_data)
    if not element_map: return False
    print(f"Generated element map with {len(element_map)} elements.")

    print("\nFeaturizing unique entries...")
    features, recipes, targets, original_indices = [], [], [], []
    processed_count, skipped_count = 0, 0
    start_featurize_time = time.time()
    for i, entry in enumerate(deduplicated_data):
        target_formula = entry.get('target', {}).get('material_formula', None)
        precursors = entry.get('precursors', None)
        operations = entry.get('operations', None)
        source_file = entry.get('_source_file', i)
        valid_entry = (target_formula and isinstance(precursors, list) and isinstance(operations, list))

        if valid_entry:
            feature_vector = featurize_target(target_formula, element_map, ELEMENT_PROPS, z_weights_array)
            if feature_vector is not None:
                features.append(feature_vector)
                recipes.append({"precursors": precursors, "operations": operations})
                targets.append(target_formula)
                original_indices.append(source_file)
                processed_count += 1
            else: skipped_count += 1
        else: skipped_count += 1

    end_featurize_time = time.time()
    print(f"Featurization took {end_featurize_time - start_featurize_time:.2f}s.")
    if not features:
        print("\nError: No entries successfully featurized.", file=sys.stderr)
        return False

    print(f"Processed {processed_count} entries into vectors. Skipped {skipped_count}.")
    feature_array = np.array(features)
    print(f"Final feature array shape: {feature_array.shape}")

    print(f"\nSaving featurized data to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'features.npy'), feature_array)
        with open(os.path.join(output_dir, 'recipes.json'), 'w') as f: json.dump(recipes, f)
        with open(os.path.join(output_dir, 'targets.json'), 'w') as f: json.dump(targets, f)
        with open(os.path.join(output_dir, 'original_indices.json'), 'w') as f: json.dump(original_indices, f)
        with open(os.path.join(output_dir, 'element_map.json'), 'w') as f: json.dump(element_map, f)
    except Exception as e:
        print(f"Error saving featurized data: {e}", file=sys.stderr)
        return False

    print(f"Featurization step completed successfully for z={z_weights_array.tolist()}.")
    return True

if __name__ == "__main__":
    print("Running featurize.py standalone with default paths and weights...")
    if not run_featurize(): # Will use default output dir and z=1.0 weights
        sys.exit(1)
