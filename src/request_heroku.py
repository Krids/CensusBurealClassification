"""
This script is responsible for the request on heroku server.

Author: Felipe Lana Machado
Date: 01/08/2022
"""

import requests


data = {
    "age": 39,
    "workclass": "state-gov",
    "fnlgt": 77516,
    "education": "bachelors",
    "education_num": 13,
    "marital_status": "never-married",
    "occupation": "adm-clerical",
    "relationship": "not-in-family",
    "race": "white",
    "sex": "male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "united-states"
}

# GET request
response = requests.get('https://census-bureal-classification.herokuapp.com/feature_info/age')
print(response.status_code)
print(response.json())

# POST request
response = requests.post('https://census-bureal-classification.herokuapp.com/predict/', auth=('user', 'pass'), json=data)
print(response.status_code)
print(response.json())
