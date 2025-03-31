import os
import json
from actiongraph import ActionGraph

def convert_to_action_graphs():
    """Convert filtered synthesis data to action graphs."""
    input_dir = "Data/filtered-mp-data/"
    output_dir = "Data/ag-data/"
    os.makedirs(output_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    converted_count = 0
    
    for json_file in json_files:
        with open(os.path.join(input_dir, json_file), 'r') as f:
            data = json.load(f)
        
        try:
            # Use the ActionGraph class to create a graph representation
            ag = ActionGraph.from_mp_synthesis(data)
            
            # Save the serialized action graph
            output_file = os.path.join(output_dir, f"{os.path.splitext(json_file)[0]}_ag.json")
            with open(output_file, 'w') as f:
                json.dump(ag.serialize(), f, indent=2)
            converted_count += 1
        except Exception as e:
            print(f"Error converting {json_file}: {e}")
    
    print(f"Successfully converted {converted_count} reactions to action graphs.")

def test_conversion(num_files=5):
    """Test the conversion process with detailed logging"""
    input_dir = "Data/mp-data/"
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')][:num_files]
    
    for json_file in json_files:
        print(f"\nTesting conversion of {json_file}")
        try:
            with open(os.path.join(input_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # Try to create an action graph
            ag = ActionGraph.from_mp_synthesis(data)
            
            # Log key statistics about the created graph
            print(f"  Success: Created graph with {len(ag.nodes())} nodes and {len(ag.edges())} edges")
            print(f"  Input nodes: {len(ag.input_nodes)}")
            print(f"  Operation nodes: {len(ag.operation_nodes)}")
            print(f"  Output nodes: {len(ag.output_nodes)}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_conversion()
