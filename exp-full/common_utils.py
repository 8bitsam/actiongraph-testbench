# common_utils.py
import json
import os
import numpy as np
from pymatgen.core import Composition, Element
from tqdm import tqdm
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)

ELEMENT_PROPS = [
    'atomic_mass', 'atomic_radius', 'atomic_volume',
    'electronegativity', 'electron_affinity', 'ionization_energy'
]
NUM_ELEMENT_PROPS = len(ELEMENT_PROPS)

def load_json_data_from_directory(dir_path, desc="Loading JSON files"):
    """Loads all .json files from a directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
        return None
    all_data = []
    all_files = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.lower().endswith('.json'):
                all_files.append(os.path.join(root, filename))

    if not all_files:
        print(f"No JSON files found in {dir_path}", file=sys.stderr)
        return None

    print(f"Found {len(all_files)} potential JSON files in {dir_path}. Loading...")
    for filepath in tqdm(all_files, desc=desc):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data_entry = json.load(f)
                if isinstance(data_entry, dict):
                    data_entry['_source_file'] = os.path.basename(filepath)
                    all_data.append(data_entry)
        except Exception as e:
            # print(f"Warning: Error loading {filepath}: {e}", file=sys.stderr)
            pass # Optionally log
    print(f"Loaded {len(all_data)} entries from {dir_path}.")
    return all_data

def get_global_element_map(list_of_data_entries, data_key='target', formula_key='material_formula'):
    """Creates a global element map from a list of data entries (MP JSONs or AG nodes)."""
    print("Creating global element map...")
    all_elements = set()
    for entry in tqdm(list_of_data_entries, desc="Scanning elements for map"):
        formula_str = None
        if data_key == 'target': # For MP JSON
            formula_str = entry.get('target', {}).get(formula_key)
        elif data_key == 'node_attrs': # For AG chemical node attributes
            formula_str = entry.get(formula_key) # 'entry' here is node_attrs

        if formula_str and isinstance(formula_str, str):
            try:
                comp = Composition(formula_str)
                if comp.num_atoms >= 1:
                    all_elements.update(el.symbol for el in comp.elements)
            except Exception:
                pass # Ignore parsing errors for element map creation
    try:
        common_elements = {el.symbol for el in Element}
        all_elements.update(common_elements)
    except Exception: pass
    sorted_elements = sorted(list(all_elements))
    if not sorted_elements:
         print("Error: No valid elements found to create map.", file=sys.stderr)
         return None
    print(f"Global element map created with {len(sorted_elements)} elements.")
    return {element: i for i, element in enumerate(sorted_elements)}

def featurize_chemical_formula(formula_str, element_map):
    """Generates features (element fractions + avg props) for a single formula string."""
    if not formula_str or not isinstance(formula_str, str) or not element_map:
        return None
    try:
        comp = Composition(formula_str)
        if comp.num_atoms < 1: return None
        el_amt_dict = comp.get_el_amt_dict()
        total_atoms = comp.num_atoms

        comp_vector = np.zeros(len(element_map))
        for el_sym, amount in el_amt_dict.items():
            if el_sym in element_map:
                comp_vector[element_map[el_sym]] = amount / total_atoms

        prop_vector = np.zeros(NUM_ELEMENT_PROPS)
        elements_in_comp_symbols = list(el_amt_dict.keys())
        for i, prop_name in enumerate(ELEMENT_PROPS):
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

        feature_vector = np.concatenate((comp_vector, prop_vector))
        return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None

def get_canonical_recipe_string_mp(entry):
    """Creates a canonical string for an MP JSON entry for deduplication."""
    try:
        target_formula = entry.get('target', {}).get('material_formula', '')
        precursors = entry.get('precursors', [])
        operations = entry.get('operations', [])
        prec_formulas = sorted([p.get('material_formula', '') for p in precursors if isinstance(p, dict)])
        op_strings = sorted([json.dumps(op, sort_keys=True) for op in operations if isinstance(op, dict)])
        canonical_string = json.dumps({
            "target": target_formula, "precursors": prec_formulas, "operations": op_strings
        }, sort_keys=True)
        return canonical_string
    except Exception:
        return f"SOURCE_FILE_MP:{entry.get('_source_file', os.urandom(16))}"


def save_featurized_data(output_dir, features_arr, recipes_list, targets_list, original_ids_list, element_map_dict):
    """Saves all featurized data components."""
    print(f"Saving featurized data to: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'features.npy'), features_arr)
        with open(os.path.join(output_dir, 'recipes.json'), 'w') as f: json.dump(recipes_list, f)
        with open(os.path.join(output_dir, 'targets.json'), 'w') as f: json.dump(targets_list, f)
        with open(os.path.join(output_dir, 'original_ids.json'), 'w') as f: json.dump(original_ids_list, f)
        with open(os.path.join(output_dir, 'element_map.json'), 'w') as f: json.dump(element_map_dict, f)
        print("Featurized data saved successfully.")
    except Exception as e:
        print(f"Error saving featurized data: {e}", file=sys.stderr)
        return False
    return True

def calculate_set_prf1(pred_set, true_set):
    """Calculates Precision, Recall, and F1 score for two sets."""
    if not isinstance(pred_set, set): pred_set = set(pred_set)
    if not isinstance(true_set, set): true_set = set(true_set)
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set.difference(true_set))
    fn = len(true_set.difference(pred_set))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not true_set else 0.0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1
