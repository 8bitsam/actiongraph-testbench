import json
import os

from mp_api.client.routes.materials.synthesis import SynthesisRester

with open("api-key.txt", "r") as key_file:
    API_KEY = key_file.readline().strip()
os.environ["MP_API_KEY"] = API_KEY
rester = SynthesisRester(api_key=API_KEY)
data_dir = "Data/mp-data/"
os.makedirs(data_dir, exist_ok=True)
try:
    synthesis_docs = rester.search(chunk_size=100)
except Exception as e:
    print(f"Error fetching synthesis data: {e}")
    synthesis_docs = []
count = 0
for doc in synthesis_docs:
    doc_dict = doc.model_dump()
    if doc_dict.get("synthesis_type", "").lower() == "solid-state":
        file_name = doc_dict.get("task_id")
        if file_name is None:
            file_name = f"reaction_{abs(hash(json.dumps(doc_dict)))}"
        file_path = os.path.join(data_dir, file_name + ".json")
        try:
            with open(file_path, "w") as f:
                json.dump(doc_dict, f, indent=2)
            count += 1
        except Exception as write_err:
            print(f"Error writing file {file_path}: {write_err}")
print(f"Saved {count} solid-state synthesis reaction files to {data_dir}")
