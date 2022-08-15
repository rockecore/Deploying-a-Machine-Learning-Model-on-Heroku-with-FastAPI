"""
Created on Mon Aug 15 12:38:09 2022

@author: rbarker
"""

import requests

inputdata = {'age': 28,
             'workclass': 'Private',
             'fnlgt': 338409,
             'education': 'Bachelors',
             'education_num': 13,
             'marital_status': 'Married_civ_spouse',
             'occupation': 'Prof_specialty',
             'relationship': 'Wife',
             'race': 'Black',
             'sex': 'Female',
             'capital_gain': 0,
             'capital_loss': 0,
             'hours_per_week': 40,
             'native_country': 'Cuba'}

response = requests.post(
    url='https://census-prediction-app.herokuapp.com/inference',
    json=inputdata)

print(response.status_code)
print(response.json())