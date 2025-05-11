# 1_ag_converter.py
import json
import os
import sys
import time
import hashlib
from tqdm import tqdm
from actiongraph import ActionGraph # Your class
from common_utils import load_json_data_from_directory # Use common loader

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
RAW_MP_DATA_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
ACTION_GRAPH_RAW_DIR = os.path.join(DATA_DIR, "ag-data-raw/") # Output

def get_ag_content_hash(ag_serial_data: dict):
    """Creates a hash from sorted serialized ActionGraph data."""
    try:
        # Ensure consistent ordering for hashing
        ag_serial_data['nodes'] = sorted(ag_serial_data.get('nodes', []), key=lambda x: x['id'])
        ag_serial_data['links'] = sorted(ag_serial_data.get('links', []), key=lambda x: (x['source'], x['target']))
        for key in ['custom_input_nodes', 'custom_operation_nodes', 'custom_output_nodes']:
            if key in ag_serial_data:
                ag_serial_data[key] = sorted(ag_serial_data[key])
        content_string = json.dumps(ag_serial_data, sort_keys=True)
        return hashlib.md5(content_string.encode('utf-8')).hexdigest()
    except Exception as e:
        # print(f"Warning: Could not generate AG hash: {e}", file=sys.stderr)
        return str(os.urandom(16)) # Fallback

def run_ag_conversion_and_dedup():
    print("--- Running MP JSON to ActionGraph Conversion & Deduplication ---")
    raw_mp_entries = load_json_data_from_directory(RAW_MP_DATA_DIR, desc="Loading MP JSONs")
    if not raw_mp_entries: return False

    print("\nConverting to ActionGraphs and serializing...")
    serialized_ags = []
    conversion_failures = 0
    for mp_entry in tqdm(raw_mp_entries, desc="Converting entries"):
        ag_instance = ActionGraph.from_mp_synthesis(mp_entry) # Returns AG or None
        if ag_instance:
            try:
                ag_data = ag_instance.serialize()
                ag_data['_source_file'] = mp_entry.get('_source_file', 'unknown_source.json')
                serialized_ags.append(ag_data)
            except Exception as e:
                # print(f"Warning: Error serializing AG from {mp_entry.get('_source_file')}: {e}")
                conversion_failures +=1
        else:
            conversion_failures += 1
    print(f"Converted {len(serialized_ags)} entries. Failures/skips in conversion: {conversion_failures}")
    if not serialized_ags: return False

    print("\nDeduplicating serialized ActionGraphs...")
    unique_ag_data_map = {}
    duplicate_count = 0
    for ag_data in tqdm(serialized_ags, desc="Deduplicating AGs"):
        content_hash = get_ag_content_hash(ag_data.copy()) # Hash a copy
        if content_hash not in unique_ag_data_map:
            unique_ag_data_map[content_hash] = ag_data
        else:
            duplicate_count += 1
    final_unique_ags_data = list(unique_ag_data_map.values())
    print(f"Removed {duplicate_count} duplicate AGs. Unique AGs: {len(final_unique_ags_data)}")
    if not final_unique_ags_data: return False

    print(f"\nSaving unique ActionGraph JSONs to: {ACTION_GRAPH_RAW_DIR}")
    os.makedirs(ACTION_GRAPH_RAW_DIR, exist_ok=True)
    saved_count = 0
    for ag_data in tqdm(final_unique_ags_data, desc="Saving AGs"):
        source_filename = ag_data.pop('_source_file', f"actiongraph_{saved_count}.json")
        # Basic filename sanitization
        clean_filename = "".join(c for c in source_filename if c.isalnum() or c in ('.', '_', '-')).rstrip()
        if not clean_filename.lower().endswith(".json"): clean_filename += ".json"
        
        output_path = os.path.join(ACTION_GRAPH_RAW_DIR, clean_filename)
        # Handle potential filename collisions if sanitization leads to same name
        if os.path.exists(output_path):
             output_path = os.path.join(ACTION_GRAPH_RAW_DIR, f"dup_{saved_count}_{clean_filename}")

        try:
            with open(output_path, 'w') as f:
                json.dump(ag_data, f, indent=2)
            saved_count += 1
        except Exception as e:
            print(f"Warning: Error saving {output_path}: {e}")
    print(f"Saved {saved_count} unique ActionGraph JSON files.")
    return True

if __name__ == "__main__":
    if not run_ag_conversion_and_dedup():
        sys.exit(1)
