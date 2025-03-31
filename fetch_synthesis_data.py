import os
import json
from mp_api.client.routes.materials.synthesis import SynthesisRester

# Read API key from file and set the environment variable.
with open("api-key.txt", "r") as key_file:
    API_KEY = key_file.readline().strip()
os.environ["MP_API_KEY"] = API_KEY

# Initialize the SynthesisRester.
rester = SynthesisRester(api_key=API_KEY)

# Create local directory if it doesn't exist.
data_dir = "Data/mp-data/"
os.makedirs(data_dir, exist_ok=True)

# Fetch all synthesis records (adjust chunk_size as needed).
try:
    synthesis_docs = rester.search(chunk_size=100)
except Exception as e:
    print(f"Error fetching synthesis data: {e}")
    synthesis_docs = []

count = 0
# Iterate over the returned synthesis documents.
for doc in synthesis_docs:
    # Convert the model to a dict.
    doc_dict = doc.model_dump()
    
    # Filter: Only keep solid-state synthesis reactions.
    # (Assumes that each document contains a key 'synthesis_type' that indicates the reaction type.)
    if doc_dict.get("synthesis_type", "").lower() == "solid-state":
        # Determine a unique file name.
        # Here we try to use a provided identifier (e.g., 'task_id') if available;
        # otherwise, we create a unique filename using a hash.
        file_name = doc_dict.get("task_id")
        if file_name is None:
            file_name = f"reaction_{abs(hash(json.dumps(doc_dict)))}"
        
        file_path = os.path.join(data_dir, file_name + ".json")
        
        # Write the record to a JSON file.
        try:
            with open(file_path, "w") as f:
                json.dump(doc_dict, f, indent=2)
            count += 1
        except Exception as write_err:
            print(f"Error writing file {file_path}: {write_err}")

print(f"Saved {count} solid-state synthesis reaction files to {data_dir}")
