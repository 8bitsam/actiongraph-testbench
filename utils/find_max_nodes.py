import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Data"))
FILTERED_AG_DATA_DIR = os.path.join(DATA_DIR, "filtered-ag-data/")


def find_max_nodes(ag_dir=FILTERED_AG_DATA_DIR):
    max_total = 0
    max_data = None
    for root, _, files in os.walk(ag_dir):
        for file in files:
            if not file.endswith("_ag.json"):
                continue
            with open(os.path.join(root, file), "r") as f:
                try:
                    data = json.load(f)
                    total_nodes = len(data.get("nodes", []))
                    if total_nodes > max_total:
                        max_total = total_nodes
                        max_data = data
                except json.JSONDecodeError:
                    continue
    return max_total, max_data


if __name__ == "__main__":
    max_nodes, max_data = find_max_nodes()
    print(f"Maximum nodes in any ActionGraph: {max_nodes}")
    print(max_data)
