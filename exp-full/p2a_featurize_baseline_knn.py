# 2a_featurize_baseline_knn.py
import json
import os
import sys
import time
import numpy as np
from common_utils import (
    load_json_data_from_directory, get_global_element_map,
    featurize_chemical_formula, get_canonical_recipe_string_mp,
    save_featurized_data
)
from tqdm import tqdm

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
RAW_MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/") # Input
FEATURIZED_BASELINE_KNN_DIR = os.path.join(DATA_DIR, "featurized-baseline-knn/") # Output

def run_featurize_baseline():
    print("--- Running Baseline k-NN Featurization (Target-Only from MP JSONs) ---")
    raw_mp_entries = load_json_data_from_directory(RAW_MP_DATA_DIR, "Loading MP JSONs for Baseline")
    if not raw_mp_entries: return False

    print("\nDeduplicating MP JSON entries...")
    unique_mp_entries_map = {}
    duplicate_count = 0
    for entry in raw_mp_entries:
        canonical_string = get_canonical_recipe_string_mp(entry)
        if canonical_string not in unique_mp_entries_map:
            unique_mp_entries_map[canonical_string] = entry
        else:
            duplicate_count += 1
    deduplicated_mp_data = list(unique_mp_entries_map.values())
    print(f"Removed {duplicate_count} duplicate MP entries. Unique entries: {len(deduplicated_mp_data)}")
    if not deduplicated_mp_data: return False

    element_map = get_global_element_map(deduplicated_mp_data, data_key='target', formula_key='material_formula')
    if not element_map: return False

    print("\nFeaturizing target products...")
    features_list, recipes_list, targets_list, original_ids_list = [], [], [], []
    processed_count, skipped_count = 0, 0

    for entry in tqdm(deduplicated_mp_data, desc="Featurizing baseline targets"):
        target_formula = entry.get('target', {}).get('material_formula')
        precursors = entry.get('precursors')
        operations = entry.get('operations')
        source_id = entry.get('_source_file', f"entry_{processed_count}")

        if target_formula and precursors is not None and operations is not None:
            feature_vector = featurize_chemical_formula(target_formula, element_map)
            if feature_vector is not None:
                features_list.append(feature_vector)
                recipes_list.append({"precursors": precursors, "operations": operations})
                targets_list.append(target_formula)
                original_ids_list.append(source_id)
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    if not features_list:
        print("Error: No entries successfully featurized for baseline.", file=sys.stderr)
        return False

    print(f"Processed {processed_count} entries. Skipped {skipped_count}.")
    features_arr = np.array(features_list)
    print(f"Final feature array shape: {features_arr.shape}")

    return save_featurized_data(
        FEATURIZED_BASELINE_KNN_DIR, features_arr, recipes_list,
        targets_list, original_ids_list, element_map
    )

if __name__ == "__main__":
    if not run_featurize_baseline():
        sys.exit(1)
