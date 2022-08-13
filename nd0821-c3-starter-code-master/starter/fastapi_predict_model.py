# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 11:25:43 2022

@author: rbarker
"""

import joblib
import os
import numpy as np
import sys
import pandas as pd
sys.path.append("./starter")
from ml.model import inference
from ml.data import process_data

def predict_model_api(data, model_out, cat_features=None):
    
    model = joblib.load(os.path.join(model_out, "salary_predictor_model.joblib"))
    encoder = joblib.load(os.path.join(model_out, "salary_predictor_encoder.joblib"))
    lb = joblib.load(os.path.join(model_out, "salary_predictor_lb.joblib"))
    
    cat_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country"]
    
    # get the data and convert to array
    target_data = np.array([[data.age,data.workclass,data.fnlgt,data.education,
                             data.education_num, data.marital_status, data.occupation,
                             data.relationship, data.race, data.sex, 
                             data.capital_gain, data.capital_loss, 
                             data.hours_per_week, data.native_country]])
    
    # need df for process_data
    fixed_data = pd.DataFrame(target_data, columns=[
        "age", "workclass", "fnlgt", "education", "education_num", "marital_status", 
        "occupation", "relationship", "race", "sex", "capital_gain",
        "capital_loss", "hours_per_week", "native_country"])
    
    # process the data for the model prediction
    X_test, _, _, _ = process_data(fixed_data, 
                                   categorical_features=cat_features, 
                                   training=False, encoder=encoder, lb=lb)
    
    income_pred = inference(model, X_test)
    
    if income_pred[0]:
        pred = ">50k"
    else:
        pred = "<=50k"
        
    return pred
