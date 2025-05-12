# featurize.py

import hashlib  # Added for creating content hash
import json
import os
import sys
import time
import warnings

import numpy as np
from pymatgen.core import Composition, Element

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
RAW_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
FEATURIZED_DATA_DIR = os.path.join(DATA_DIR, "featurized-data-baseline/")

ELEMENT_PROPS = [
    "atomic_mass",
    "atomic_radius",
    "melting_point",
    "X",
    "electron_affinity",
    "ionization_energy",
]

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- Helper Functions (load_data_from_directory, get_element_map, featurize_target) ---
def load_data_from_directory(dir_path):
    """Loads synthesis data from all .json files in a directory."""
    if not os.path.isdir(dir_path):
        print(f"Error: Path is not a valid directory: {dir_path}", file=sys.stderr)
        return None
    all_data = []
    print(f"Scanning directory: {dir_path}")
    start_time = time.time()
    loaded_count = 0
    error_count = 0
    for root, _, files in os.walk(dir_path):
        json_files = [f for f in files if f.lower().endswith(".json")]
        if not json_files:
            continue
        for filename in json_files:
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data_entry = json.load(f)
                    if isinstance(data_entry, dict):
                        data_entry["_source_file"] = filename  # Add identifier
                        all_data.append(data_entry)
                        loaded_count += 1
                    else:
                        error_count += 1
            except Exception:
                error_count += 1  # Simplified error handling
    end_time = time.time()
    print(
        f"Finished scanning. Loaded {loaded_count} entries, skipped {error_count} files."
    )
    print(f"Data loading took {end_time - start_time:.2f} seconds.")
    if not all_data:
        print(
            f"Error: No valid JSON data loaded from directory: {dir_path}",
            file=sys.stderr,
        )
        return None
    return all_data


def get_element_map(data):
    """Creates a mapping from element symbol to feature vector index."""
    all_elements = set()
    parsed_count = 0
    parse_error_count = 0
    for i, entry in enumerate(data):
        if (
            not isinstance(entry, dict)
            or "target" not in entry
            or "material_formula" not in entry["target"]
        ):
            continue
        target_formula = entry["target"].get("material_formula")
        if not target_formula or not isinstance(target_formula, str):
            continue
        try:
            comp = Composition(target_formula)
            if comp.num_atoms < 1:
                continue
            for el in comp.elements:
                all_elements.add(el.symbol)
            parsed_count += 1
        except Exception:
            parse_error_count += 1
    if parse_error_count > 0:
        print(
            f"Warning: Could not parse target formula for {parse_error_count} entries during element map creation."
        )
    try:
        common_elements = {el.symbol for el in Element}
        all_elements.update(common_elements)
    except Exception as e:
        print(
            f"Warning: Could not retrieve all Pymatgen elements: {e}", file=sys.stderr
        )
    sorted_elements = sorted(list(all_elements))
    if not sorted_elements:
        print("Error: No valid elements found in the dataset.", file=sys.stderr)
        return None
    return {element: i for i, element in enumerate(sorted_elements)}


def featurize_target(formula, element_map, element_props_list):
    """Generates features: element fraction vector + averaged properties."""
    if not formula or not isinstance(formula, str):
        return None
    try:
        comp = Composition(formula)
        if comp.num_atoms < 1:
            return None
        el_amt_dict = comp.get_el_amt_dict()
        total_atoms = comp.num_atoms
        comp_vector = np.zeros(len(element_map))
        for element_symbol, amount in el_amt_dict.items():
            if element_symbol in element_map:
                comp_vector[element_map[element_symbol]] = amount / total_atoms
        prop_vector = np.zeros(len(element_props_list))
        elements_in_comp_symbols = list(el_amt_dict.keys())
        for i, prop_name in enumerate(element_props_list):
            weighted_sum = 0.0
            total_amount_for_prop = 0.0
            prop_available = False
            for el_symbol in elements_in_comp_symbols:
                amount = el_amt_dict[el_symbol]
                try:
                    el_obj = Element(el_symbol)
                    prop_value = getattr(el_obj, prop_name, None)
                    if prop_value is not None and isinstance(prop_value, (int, float)):
                        weighted_sum += prop_value * amount
                        total_amount_for_prop += amount
                        prop_available = True
                except Exception:
                    pass  # Catch all errors during property lookup
            if prop_available and total_amount_for_prop > 0:
                prop_vector[i] = weighted_sum / total_amount_for_prop
        feature_vector = np.concatenate((comp_vector, prop_vector))
        return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def get_recipe_hash(entry):
    """Creates a unique hash for a recipe based on target, precursors, and
    operations."""
    try:
        target_formula = entry.get("target", {}).get("material_formula", "")
        precursors = entry.get("precursors", [])
        operations = entry.get("operations", [])

        # Create canonical representations
        # Sort precursors by formula
        prec_formulas = sorted(
            [p.get("material_formula", "") for p in precursors if isinstance(p, dict)]
        )
        # Sort operations based on their JSON string representation
        op_strings = sorted(
            [
                json.dumps(op, sort_keys=True)
                for op in operations
                if isinstance(op, dict)
            ]
        )

        # Combine into a single string
        hash_content = f"TARGET:{target_formula}|PRECURSORS:{','.join(prec_formulas)}|OPERATIONS:{'|'.join(op_strings)}"

        # Return MD5 hash (or other hash like SHA256)
        return hashlib.md5(
            hash_content.encode("utf-8")
        ).hexdigest()  # DONT USE md5 hash
    except Exception as e:
        # Handle cases where serialization fails (e.g., non-standard data in ops)
        # print(f"Warning: Could not generate hash for entry {_entry.get('_source_file', 'unknown')}: {e}", file=sys.stderr)
        # Return a unique hash based on the source file as a fallback, though less robust for content duplicates
        return hashlib.md5(
            str(entry.get("_source_file", os.urandom(16))).encode("utf-8")
        ).hexdigest()


def run_featurize():
    """Loads raw data, **deduplicates**, featurizes it, and saves to disk."""
    print("--- Running Featurization Step ---")
    raw_data = load_data_from_directory(RAW_DATA_DIR)
    if not raw_data:
        return False
    print(f"\nSuccessfully loaded {len(raw_data)} potential entries overall.")

    # --- Deduplication Step ---
    print("\nDeduplicating entries based on recipe content...")
    start_dedup_time = time.time()
    unique_recipes = {}  # Use dict to store first encountered entry for each hash
    duplicate_count = 0
    processed_for_dedup = 0
    for entry in raw_data:
        recipe_hash = get_recipe_hash(entry)
        if recipe_hash not in unique_recipes:
            unique_recipes[recipe_hash] = entry
        else:
            duplicate_count += 1
        processed_for_dedup += 1
        if processed_for_dedup % 10000 == 0:
            print(f"  Checked {processed_for_dedup}/{len(raw_data)} for duplicates...")

    deduplicated_data = list(unique_recipes.values())
    end_dedup_time = time.time()
    print(f"Deduplication took {end_dedup_time - start_dedup_time:.2f} seconds.")
    print(f"Removed {duplicate_count} duplicate entries based on recipe hash.")
    print(f"Proceeding with {len(deduplicated_data)} unique entries.")

    if not deduplicated_data:
        print(
            "Error: No unique data entries remaining after deduplication.",
            file=sys.stderr,
        )
        return False

    # --- Proceed with Featurization using deduplicated_data ---
    print("\nCreating element map from unique entries...")
    element_map = get_element_map(deduplicated_data)  # Use unique data
    if not element_map:
        print("Failed to create element map. Exiting.", file=sys.stderr)
        return False
    print(f"Generated element map with {len(element_map)} elements.")

    print("\nFeaturizing unique entries...")
    features, recipes, targets, original_indices = [], [], [], []
    processed_count, skipped_count = 0, 0
    start_featurize_time = time.time()

    # Iterate over the unique data now
    for i, entry in enumerate(deduplicated_data):
        target_data = entry.get("target", {})
        target_formula = target_data.get("material_formula", None)
        precursors = entry.get("precursors", None)
        operations = entry.get("operations", None)
        source_file = entry.get("_source_file", i)  # Get original identifier

        valid_entry = (
            isinstance(target_data, dict)
            and target_formula
            and isinstance(target_formula, str)
            and precursors is not None
            and isinstance(precursors, list)
            and operations is not None
            and isinstance(operations, list)
        )

        if valid_entry:
            feature_vector = featurize_target(
                target_formula, element_map, ELEMENT_PROPS
            )
            if feature_vector is not None:
                features.append(feature_vector)
                recipes.append({"precursors": precursors, "operations": operations})
                targets.append(target_formula)
                original_indices.append(source_file)  # Store identifier of unique entry
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
        # Reduce progress update frequency
        if (i + 1) % 20000 == 0:
            print(f"  Featurized {i+1}/{len(deduplicated_data)} unique entries...")

    end_featurize_time = time.time()
    print(
        f"\nFeaturization of unique entries took {end_featurize_time - start_featurize_time:.2f} seconds."
    )

    if not features:
        print(
            "\nError: No valid data points found after featurization.", file=sys.stderr
        )
        return False

    print(
        f"\nSuccessfully processed {processed_count} unique entries into feature vectors."
    )
    print(f"Skipped {skipped_count} unique entries during featurization.")
    feature_array = np.array(features)
    print(f"Final feature array shape: {feature_array.shape}")

    print(f"\nSaving featurized data to: {FEATURIZED_DATA_DIR}")
    try:
        os.makedirs(FEATURIZED_DATA_DIR, exist_ok=True)
        np.save(os.path.join(FEATURIZED_DATA_DIR, "features.npy"), feature_array)
        with open(os.path.join(FEATURIZED_DATA_DIR, "recipes.json"), "w") as f:
            json.dump(recipes, f)
        with open(os.path.join(FEATURIZED_DATA_DIR, "targets.json"), "w") as f:
            json.dump(targets, f)
        with open(os.path.join(FEATURIZED_DATA_DIR, "original_indices.json"), "w") as f:
            json.dump(original_indices, f)  # These are identifiers from unique entries
        with open(os.path.join(FEATURIZED_DATA_DIR, "element_map.json"), "w") as f:
            json.dump(element_map, f)
    except Exception as e:
        print(f"Error saving featurized data: {e}", file=sys.stderr)
        return False

    print("Featurization step completed successfully (with deduplication).")
    return True


if __name__ == "__main__":
    if not run_featurize():
        sys.exit(1)
