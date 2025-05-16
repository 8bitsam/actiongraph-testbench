import json
import os
import sys

from tqdm import tqdm

from actiongraph import ActionGraph

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
RAW_MP_DATA_INPUT_DIR = os.path.join(DATA_DIR, "filtered-mp-data/")
FILTERED_AG_OUTPUT_DIR = os.path.join(DATA_DIR, "filtered-ag-data/")


def load_mp_json_files(dir_path, desc="Loading MP JSONs"):
    """Loads all .json files from a directory, adding a source file
    identifier."""
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found: {dir_path}", file=sys.stderr)
        return []
    all_data = []
    all_filenames = []
    for root, _, files in os.walk(dir_path):
        for filename in files:
            if filename.lower().endswith(".json"):
                all_filenames.append(os.path.join(root, filename))

    if not all_filenames:
        print(f"No JSON files found in {dir_path}", file=sys.stderr)
        return []

    print(f"Found {len(all_filenames)} potential MP JSON files. Loading...")
    for filepath in tqdm(all_filenames, desc=desc):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data_entry = json.load(f)
                if isinstance(data_entry, dict):
                    data_entry["_source_file"] = os.path.basename(filepath)
                    all_data.append(data_entry)
        except Exception:
            pass
    print(f"Successfully loaded {len(all_data)} MP JSON entries.")
    return all_data


def get_canonical_ag_json_string_for_hash(ag_serial_data: dict) -> str:
    """Creates a canonical, sorted JSON string for hashing ActionGraph data."""
    try:
        data_to_hash = json.loads(json.dumps(ag_serial_data))
        if "nodes" in data_to_hash and isinstance(data_to_hash["nodes"], list):
            data_to_hash["nodes"] = sorted(
                data_to_hash["nodes"], key=lambda x: str(x.get("id", ""))
            )
        edge_key = (
            "links"
            if "links" in data_to_hash
            else ("edges" if "edges" in data_to_hash else None)
        )
        if (
            edge_key
            and edge_key in data_to_hash
            and isinstance(data_to_hash[edge_key], list)
        ):
            data_to_hash[edge_key] = sorted(
                data_to_hash[edge_key],
                key=lambda x: (str(x.get("source", "")), str(x.get("target", ""))),
            )
        for key in [
            "custom_input_nodes",
            "custom_operation_nodes",
            "custom_output_nodes",
        ]:
            if key in data_to_hash and isinstance(data_to_hash[key], list):
                data_to_hash[key] = sorted([str(item) for item in data_to_hash[key]])
        return json.dumps(data_to_hash, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(ag_serial_data.get("_source_file", os.urandom(16)))


def run_conversion_to_ag():
    print("--- Converting MP JSONs to Unique Serialized ActionGraphs ---")
    raw_mp_entries = load_mp_json_files(RAW_MP_DATA_INPUT_DIR)
    if not raw_mp_entries:
        return False

    print("\nConverting MP entries to ActionGraph objects and serializing...")
    serialized_ags_with_source_info = []
    skipped_mp_entries_count = 0
    serialization_errors_count = 0

    for mp_entry in tqdm(raw_mp_entries, desc="MP to AG Conversion"):
        source_filename_for_ag = mp_entry.get("_source_file", "unknown_mp_source.json")
        action_graph_instance = ActionGraph.from_mp_synthesis(mp_entry)

        if action_graph_instance:
            try:
                ag_data_serialized = action_graph_instance.serialize()
                ag_data_serialized["_original_mp_filename"] = source_filename_for_ag
                serialized_ags_with_source_info.append(ag_data_serialized)
            except Exception:
                serialization_errors_count += 1
        else:
            skipped_mp_entries_count += 1

    print(f"Attempted conversion for {len(raw_mp_entries)} MP entries.")
    print(
        f"Successfully serialized {len(serialized_ags_with_source_info)} \
            ActionGraphs."
    )
    print(
        f"MP entries skipped during AG construction: \
          {skipped_mp_entries_count}"
    )
    print(f"AGs failed during serialization: {serialization_errors_count}")
    if not serialized_ags_with_source_info:
        return False

    print("\nDeduplicating serialized ActionGraphs by content...")
    unique_ag_map_for_dedup = {}
    ag_duplicate_removed_count = 0
    for ag_data_item in tqdm(serialized_ags_with_source_info, desc="Deduplicating AGs"):
        data_for_hashing_step = ag_data_item.copy()
        data_for_hashing_step.pop("_original_mp_filename", None)
        canonical_ag_str = get_canonical_ag_json_string_for_hash(data_for_hashing_step)
        content_hash_val = hash(canonical_ag_str)
        if content_hash_val not in unique_ag_map_for_dedup:
            unique_ag_map_for_dedup[content_hash_val] = ag_data_item
        else:
            ag_duplicate_removed_count += 1
    final_unique_serialized_ags = list(unique_ag_map_for_dedup.values())
    print(
        f"Removed {ag_duplicate_removed_count} duplicate AGs. Unique AGs: \
            {len(final_unique_serialized_ags)}"
    )
    if not final_unique_serialized_ags:
        return False
    print(f"\nSaving unique ActionGraph JSONs to: {FILTERED_AG_OUTPUT_DIR}")
    os.makedirs(FILTERED_AG_OUTPUT_DIR, exist_ok=True)
    final_saved_count = 0
    for ag_data_to_save_final in tqdm(final_unique_serialized_ags, desc="Saving AGs"):
        original_mp_filename = ag_data_to_save_final.pop(
            "_original_mp_filename", f"filtered_ag_{final_saved_count}.json"
        )
        base_mp_name, _ = os.path.splitext(original_mp_filename)
        output_ag_filename = f"{base_mp_name}_ag.json"
        output_ag_filepath = os.path.join(FILTERED_AG_OUTPUT_DIR, output_ag_filename)
        counter = 0
        temp_final_path = output_ag_filepath
        while os.path.exists(temp_final_path):
            counter += 1
            temp_final_path = os.path.join(
                FILTERED_AG_OUTPUT_DIR, f"{base_mp_name}_{counter}_ag.json"
            )
        output_ag_filepath = temp_final_path
        try:
            with open(output_ag_filepath, "w", encoding="utf-8") as f:
                json.dump(ag_data_to_save_final, f, indent=2)
            final_saved_count += 1
        except Exception:
            pass
    print(
        f"Saved {final_saved_count} unique ActionGraph JSON files to \
            {FILTERED_AG_OUTPUT_DIR}."
    )
    print("Conversion and AG deduplication step completed.")
    return True


if __name__ == "__main__":
    if not run_conversion_to_ag():
        sys.exit(1)
