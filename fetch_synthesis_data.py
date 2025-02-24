import os
import json
from mp_api.client.routes.materials.synthesis import SynthesisRester

with open("api-key.txt", "r") as key_file:
    API_KEY = key_file.readline().strip()
os.environ["MP_API_KEY"] = API_KEY
rester = SynthesisRester(api_key=API_KEY)

# Example: Search for synthesis records that mention "silicon".
try:
    synthesis_docs = rester.search(keywords=["silicon"], chunk_size=5)
except Exception as e:
    print(f"Error fetching synthesis data: {e}")
    synthesis_docs = []
for doc in synthesis_docs:
    print(json.dumps(doc.model_dump(), indent=2))
    print("-" * 50)
