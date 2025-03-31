from actiongraph import ActionGraph
import json
import os

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def main():
    mp_data_dir = 'Data/mp-data'
    ag_data_dir = 'Data/ag-data'
    for filename in os.listdir(mp_data_dir):
        if filename.endswith('.json'):
            input_file_path = os.path.join(mp_data_dir, filename)
            mp_data = load_json(input_file_path)
            ag_data = ActionGraph.from_mp_synthesis(mp_data).serialize()
            output_file_path = os.path.join(ag_data_dir, filename)
            save_json(ag_data, output_file_path)

if __name__ == "__main__":
    main()
