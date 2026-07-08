import requests
import json

pid = "d9a6e423-9824-4035-9665-81213e35dec9"
print("Setting active provider...")
res = requests.patch("http://localhost:8000/settings", json={"active_provider_id": pid})
print("PATCH Response:", res.status_code, res.text)

print("\nTesting Query Endpoint...")
res2 = requests.post("http://localhost:8000/query", json={"query": "test"})
print("Query POST Status:", res2.status_code)
