import requests
import json
import os

# Delete the config file to start fresh
try:
    os.remove("data/llm_config.json")
    print("Deleted config file")
except FileNotFoundError:
    pass

# 1. Switch to online mode
print("1. Switching to online mode")
res = requests.patch("http://localhost:8000/settings", json={"inference_mode": "online"})
print(res.status_code, res.text)

# 2. Add provider
print("2. Adding provider")
res = requests.post("http://localhost:8000/settings/providers", json={
    "provider": "groq",
    "model": "llama3",
    "api_key": "test_key"
})
print(res.status_code, res.text)

# 3. Trigger query
print("3. Querying")
res = requests.post("http://localhost:8000/query", json={"query": "hello"})
print(res.status_code, res.text)
