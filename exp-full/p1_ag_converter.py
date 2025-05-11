# p1_ag_converter.py
import json
import os
import sys
import time
# import hashlib # No longer needed for this specific hashing
from tqdm import tqdm

try:
    from actiongraph import ActionGraph
    from common_utils import load_json_data_from_directory, get_canonical_recipe_string_mp
except ImportError as e:
    print(f"Error importing required modules: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))

RAW_MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
FILTERED_ACTION_GRAPH_DIR = os.path.join(DATA_DIR, "filtered-ag-data/")

def get_canonical_ag_string_for_hashing(ag_serial_data: dict) -> str:
    """
    Creates a canonical, sorted JSON string representation of the AG data.
    This string can then be used with Python's built-in hash() or other
    non-cryptographic hashing if needed.
    """
    try:
        # Create a deep copy to sort without modifying the original dict if it's reused
        data_to_hash = json.loads(json.dumps(ag_serial_data))

        # Ensure consistent ordering for hashing internal lists
        if 'nodes' in data_to_hash and isinstance(data_to_hash['nodes'], list):
            # Sort nodes by their 'id' attribute
            data_to_hash['nodes'] = sorted(data_to_hash['nodes'], key=lambda x: str(x.get('id', '')))
        if 'links' in data_to_hash and isinstance(data_to_hash['links'], list):
            # Sort links based on a tuple of source and target IDs (as strings)
            data_to_hash['links'] = sorted(data_to_hash['links'], key=lambda x: (str(x.get('source','')), str(x.get('target',''))))

        # Sort custom node lists if they are part of the serialization
        # These list names come from your ActionGraph.serialize() method
        for key in ['custom_input_nodes', 'custom_operation_nodes', 'custom_output_nodes']:
            if key in data_to_hash and isinstance(data_to_hash[key], list):
                data_to_hash[key] = sorted([str(item) for item in data_to_hash[key]]) # Ensure items are strings for sorting

        # Convert the consistently sorted dict to a JSON string with sorted keys
        # Using separators=(',', ':') makes the string more compact and consistent
        content_string = json.dumps(data_to_hash, sort_keys=True, separators=(',', ':'))
        return content_string
    except Exception as e:
        # Fallback if complex hashing fails (e.g., unexpected data types)
        # print(f"Warning: Could not generate canonical AG string due to: {e}", file=sys.stderr)
        # Using a combination of potentially unique fields as a string
        fallback_str = str(ag_serial_data.get('_source_file', '')) + \
                       str(ag_serial_data.get('doi', '')) + \
                       str(len(ag_serial_data.get('nodes', []))) + \
                       str(len(ag_serial_data.get('links', [])))
        return fallback_str


def run_ag_conversion_and_dedup():
    print("--- Running Filtered MP JSON to ActionGraph Conversion & Deduplication ---")

    raw_mp_entries = load_json_data_from_directory(RAW_MP_DATA_DIR, desc="Loading original MP JSONs")
    if not raw_mp_entries:
        print("No raw MP entries loaded. Exiting.", file=sys.stderr)
        return False
    print(f"Loaded {len(raw_mp_entries)} raw MP entries.")

    print("\nDeduplicating original MP JSON entries (Recipe Content)...")
    unique_mp_entries_map = {}
    mp_duplicate_count = 0
    for entry in tqdm(raw_mp_entries, desc="Deduplicating MP JSONs"):
        canonical_mp_string = get_canonical_recipe_string_mp(entry) # From common_utils
        # Use Python's built-in hash for the canonical MP string
        mp_entry_hash = hash(canonical_mp_string)
        if mp_entry_hash not in unique_mp_entries_map:
            unique_mp_entries_map[mp_entry_hash] = entry
        else:
            mp_duplicate_count += 1
    deduplicated_mp_entries = list(unique_mp_entries_map.values())
    print(f"Removed {mp_duplicate_count} duplicate MP entries. Proceeding with {len(deduplicated_mp_entries)} unique MP entries.")
    if not deduplicated_mp_entries: return False

    print("\nConverting unique MP entries to ActionGraphs and serializing...")
    serialized_ags_with_source = []
    conversion_failures = 0
    construction_value_errors = 0

    for mp_entry in tqdm(deduplicated_mp_entries, desc="Converting unique MP to AGs"):
        source_filename = mp_entry.get('_source_file', 'unknown_source.json')
        ag_instance = ActionGraph.from_mp_synthesis(mp_entry)

        if ag_instance:
            try:
                ag_data_serialized = ag_instance.serialize()
                ag_data_serialized['_internal_source_file'] = source_filename
                serialized_ags_with_source.append(ag_data_serialized)
            except Exception as e:
                conversion_failures += 1
        else:
            construction_value_errors += 1

    print(f"Successfully attempted conversion for {len(serialized_ags_with_source)} unique MP entries.")
    if construction_value_errors > 0:
        print(f"{construction_value_errors} unique MP entries skipped by AG.from_mp_synthesis.")
    if conversion_failures > 0:
        print(f"{conversion_failures} ActionGraphs failed during serialization.")
    if not serialized_ags_with_source:
        print("Error: No ActionGraphs created from unique MP entries.", file=sys.stderr)
        return False

    print("\nDeduplicating serialized ActionGraphs (AG Content)...")
    start_dedup_time = time.time()
    unique_ag_data_map = {} # Stores hash -> ag_data_serialized
    ag_duplicate_count = 0

    for ag_data_s in tqdm(serialized_ags_with_source, desc="Deduplicating AGs"):
        data_for_hashing = ag_data_s.copy()
        data_for_hashing.pop('_internal_source_file', None) # Remove temporary key

        # Create a canonical string representation for the AG data
        canonical_ag_string = get_canonical_ag_string_for_hashing(data_for_hashing)
        # Use Python's built-in hash() on this canonical string
        content_hash = hash(canonical_ag_string)

        if content_hash not in unique_ag_data_map:
            unique_ag_data_map[content_hash] = ag_data_s # Store the version with _internal_source_file
        else:
            ag_duplicate_count += 1

    final_unique_ags_to_save = list(unique_ag_data_map.values())
    end_dedup_time = time.time()
    print(f"AG Deduplication took {end_dedup_time - start_dedup_time:.2f} seconds.")
    print(f"Removed {ag_duplicate_count} duplicate ActionGraphs based on content.")
    print(f"Proceeding with {len(final_unique_ags_to_save)} final unique ActionGraphs.")
    if not final_unique_ags_to_save: return False

    print(f"\nSaving final unique ActionGraph JSONs to: {FILTERED_ACTION_GRAPH_DIR}")
    try:
        os.makedirs(FILTERED_ACTION_GRAPH_DIR, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory {FILTERED_ACTION_GRAPH_DIR}: {e}", file=sys.stderr)
        return False

    saved_count = 0; save_errors = 0
    for ag_data_to_save in tqdm(final_unique_ags_to_save, desc="Saving final AGs"):
        original_source_filename = ag_data_to_save.pop('_internal_source_file', f"filtered_ag_{saved_count}.json")
        clean_filename = "".join(c for c in original_source_filename if c.isalnum() or c in ('.', '_', '-')).rstrip()
        if not clean_filename.lower().endswith(".json"):
            base, ext = os.path.splitext(clean_filename)
            clean_filename = base + ".json" if base else f"filtered_ag_{saved_count}.json" # Ensure some name

        output_path = os.path.join(FILTERED_ACTION_GRAPH_DIR, clean_filename)
        counter = 0; temp_output_path = output_path
        while os.path.exists(temp_output_path):
            counter += 1; base, ext = os.path.splitext(output_path) # Use original output_path for base
            temp_output_path = f"{base}_{counter}{ext}"
        output_path = temp_output_path
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(ag_data_to_save, f, indent=2)
            saved_count += 1
        except Exception as e: save_errors += 1

    print(f"Successfully saved {saved_count} final unique ActionGraph JSON files.")
    if save_errors > 0: print(f"Encountered {save_errors} errors during saving final AGs.")
    print("Filtered ActionGraph conversion step completed successfully.")
    return True

if __name__ == "__main__":
    if not run_ag_conversion_and_dedup():
        sys.exit(1)
