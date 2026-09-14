# SMOKE SCRIPT (not a pytest test): manual live-server check; excluded from pytest via pytest.ini addopts.
import requests
import json

print("Testing Settings Endpoint...")
try:
    res = requests.get("http://localhost:8000/settings")
    print("Settings GET:", json.dumps(res.json(), indent=2))
except Exception as e:
    print("Settings GET Failed:", e)

print("\nTesting Query Endpoint...")
try:
    res = requests.post("http://localhost:8000/query", json={"query": "test"})
    print("Query POST Status:", res.status_code)
    print("Query POST Response:", res.text[:200])
except Exception as e:
    print("Query POST Failed:", e)
