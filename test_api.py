import requests

url = "http://127.0.0.1:5000/predict"
sample_input = {
    "connected_handling_time": 45,
    "issue_responded": "Yes",
    "channel_name": "Chat",
    "category": "Electronics",
    "Customer_Remarks": "delivery delayed but support was good",
    "Item_price": 1200,
    "Agent_name": "Amit"
}

res = requests.post(url, json=sample_input)
print(res.status_code)
print(res.json())