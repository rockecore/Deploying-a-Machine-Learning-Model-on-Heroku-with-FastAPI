"""
Created on Sun Aug  7 11:30:50 2022

@author: rbarker

Put the code for your API here.

To run: uvicorn fastapi_deploy:app --reload

Available at: http://127.0.0.1:8000/docs
"""

import os
import numpy as np
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi_predict_model import predict_model_api
from typing import Literal
from pydantic import BaseModel, Field

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"]


class FeatureData(BaseModel):
    
    age: int #= Field(..., example=40)
    workclass: Literal['State_gov', 'Self_emp_not_inc', 'Private', 'Federal_gov',
        'Local_gov', '?', 'Self_emp_inc', 'Without_pay', 'Never_worked']
    fnlgt: int
    education: Literal['Bachelors', 'HS_grad', '11th', 'Masters', '9th', 'Some_college',
        'Assoc_acdm', 'Assoc_voc', '7th_8th', 'Doctorate', 'Prof_school',
        '5th_6th', '10th', '1st_4th', 'Preschool', '12th']
    education_num: int
    marital_status: Literal['Never_married', 'Married_civ_spouse', 'Divorced',
        'Married_spouse_absent', 'Separated', 'Married_AF_spouse',
        'Widowed']
    occupation: Literal['Adm_clerical', 'Exec_managerial', 'Handlers_cleaners',
        'Prof_specialty', 'Other_service', 'Sales', 'Craft_repair',
        'Transport_moving', 'Farming_fishing', 'Machine_op_inspct',
        'Tech_support', '?', 'Protective_serv', 'Armed_Forces',
        'Priv_house_serv']
    relationship: Literal['Not_in_family', 'Husband', 'Wife', 'Own_child', 'Unmarried',
        'Other_relative']
    race: Literal['White', 'Black', 'Asian_Pac_Islander', 'Amer_Indian_Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal['United_States', 'Cuba', 'Jamaica', 'India', '?', 'Mexico',
        'South', 'Puerto_Rico', 'Honduras', 'England', 'Canada', 'Germany',
        'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia',
        'Thailand', 'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican_Republic', 'El_Salvador', 'France', 'Guatemala',
        'China', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying_US(Guam_USVI_etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand_Netherlands']


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -q") != 0:
        exit("dvc pull failed")
    os.system("rm -rf .dvc .apt/usr/lib/dvc")
    
app = FastAPI()

@app.get("/")
def welcome():
    return {"message": "Bienvenidos!"}


@app.post('/inference')
async def predict_income(inputdata: FeatureData):
    
    model_dir = 'model/'
    
    # encoded_data = jsonable_encoder(inputdata)
    
    output = predict_model_api(inputdata, model_dir)

    return {"income": output}

