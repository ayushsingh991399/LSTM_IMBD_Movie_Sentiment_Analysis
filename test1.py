# test_positive.py
import requests

url = "http://localhost:9696/predict"

payload = {
    "review": "This movie was absolutely amazing. The acting, story, and direction were excellent."
}

response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())
