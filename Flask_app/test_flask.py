import requests
import json

# URL of your Flask endpoint
url = 'http://127.0.0.1:5000/predict'

# JSON payload to send to the Flask endpoint
payload = {
    "year": 2015,
    "kilometers_driven": 50000,
    "mileage": 18.5,
    "engine": 1197,
    "power": 83.1,
    "seats": 5,
    "location": "bangalore",
    "fuel_type": "petrol",
    "transmission": "manual",
    "owner_type": "first",
    "brand": "hyundai"
}

# Send POST request
try:
    response = requests.post(url, json=payload)
    # Print the response from the server
    print("Response Status Code:", response.status_code)
    print("Response JSON:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
