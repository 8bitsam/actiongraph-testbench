# make_ags.py (or ag_converter.py)

from actiongraph import ActionGraph # Your updated actiongraph.py
import json
import os
import sys
from tqdm import tqdm # For progress

# --- Configuration (adjust paths as needed) ---
# Assuming this script is in a 'gat' subdirectory and Data is one level up
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))

MP_DATA_DIR = os.path.join(BASE_DATA_DIR, 'filtered-mp-data') # Or your specific raw data dir
AG_DATA_DIR = os.path.join(BASE_DATA_DIR, 'filtered-ag-data')    # Or your specific output dir

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2) # Use indent for readability

def main():
    print(f"Processing files from: {MP_DATA_DIR}")
    print(f"Saving ActionGraphs to: {AG_DATA_DIR}")

    if not os.path.isdir(MP_DATA_DIR):
        print(f"Error: Input directory not found: {MP_DATA_DIR}")
        sys.exit(1)

    if not os.path.isdir(AG_DATA_DIR):
        print(f"Output directory {AG_DATA_DIR} not found. Creating it.")
        try:
            os.makedirs(AG_DATA_DIR, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {AG_DATA_DIR}: {e}")
            sys.exit(1)

    all_files = []
    for root, _, files in os.walk(MP_DATA_DIR):
        for filename in files:
            if filename.lower().endswith('.json'):
                all_files.append(os.path.join(root, filename))

    if not all_files:
        print(f"No JSON files found in {MP_DATA_DIR}")
        return

    converted_count = 0
    skipped_count = 0
    error_count = 0

    for file_path in tqdm(all_files, desc="Converting MP data to ActionGraphs"):
        filename = os.path.basename(file_path)
        try:
            mp_data = load_json(file_path)

            # *** THIS IS THE CRITICAL FIX IN YOUR CALLING SCRIPT ***
            action_graph_instance = ActionGraph.from_mp_synthesis(mp_data)

            if action_graph_instance is not None: # Check if conversion was successful
                ag_data_serialized = action_graph_instance.serialize()
                output_file_path = os.path.join(AG_DATA_DIR, filename) # Save with original filename
                save_json(ag_data_serialized, output_file_path)
                converted_count += 1
            else:
                # ActionGraph.from_mp_synthesis returned None, so skip this file
                # Optional: print(f"  Skipping {filename}: Invalid data for ActionGraph.")
                skipped_count += 1
            # *** END OF CRITICAL FIX ***

        except json.JSONDecodeError:
            # print(f"  Skipping {filename}: Invalid JSON format.")
            error_count += 1
        except Exception as e:
            # print(f"  An unexpected error occurred with file {filename}: {e}")
            error_count += 1

    print(f"\n--- Conversion Summary ---")
    print(f"Total files processed: {len(all_files)}")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped (invalid/empty for AG): {skipped_count}")
    print(f"Skipped (JSON/other errors): {error_count}")

if __name__ == "__main__":
    main()
