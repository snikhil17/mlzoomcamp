import requests

url = "https://localhost:9696"

customer = {"contract": "two_year", 
		 "tenure": 12, 
		 "monthlycharges": 10}

requests.post(url, json=customer).json()