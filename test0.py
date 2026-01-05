# test_negative.py
import requests

url = "http://localhost:9696/predict"

payload = {
    "review": "This movie was terrible. The plot was boring and the acting was very bad."
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
