"""
Created on Fri Aug 12 13:37:35 2022

@author: rbarker
"""

from fastapi.testclient import TestClient

from fastapi_deploy import app

client = TestClient(app)


def test_welcome():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Bienvenidos!"}


def test_predict_income_gt():
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
    
    r = client.post("/inference", json=inputdata)
    assert r.status_code == 200
    assert r.json() == {"income": ">50k"}


def test_predict_income_lt():
    test_data = {'age': 40,
                  'workclass': 'Private',
                  'fnlgt': 154374,
                  'education': 'HS_grad',
                  'education_num': 9,
                  'marital_status': 'Married_civ_spouse',
                  'occupation': 'Machine_op_inspct',
                  'relationship': 'Husband',
                  'race': 'Black',
                  'sex': 'Male',
                  'capital_gain': 0,
                  'capital_loss': 0,
                  'hours_per_week': 30,
                  'native_country': 'United_States'}
    
    r = client.post("/inference", json=test_data)
    assert r.status_code == 200
    assert r.json() == {"income": "<=50k"}
