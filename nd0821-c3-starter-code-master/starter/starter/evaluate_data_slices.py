"""
Created on Sun Aug  7 07:17:51 2022

@author: Rocky Barker

Function that outputs the performance of the model on slices of data
"""

import os
import pandas as pd
import joblib
import sys
sys.path.append(".")
from ml.data import process_data
from train_model import split_data
from sklearn.metrics import (accuracy_score, 
                             f1_score, 
                             precision_score, 
                             recall_score)


class EvaluateDataSlices:
    
    
    def __init__(self):
    
        my_dir = os.getcwd()
        self.par_dir = os.path.abspath(os.path.join(my_dir, os.pardir))
        self.data_dir = os.path.join(self.par_dir, "data/clean_census.csv")
        
        self.cat_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country"]
        
        self.label = "salary"
        
        # run methods
        self.get_data()
        self.get_model()
        self.evaluate_slices()
        self.save_slice_metrics()
        
    def get_data(self):
        
        self.train_data, self.test_data = split_data(
            data_dir=self.data_dir)
        
    def get_model(self):
        
        self.model = joblib.load(
            os.path.join(self.par_dir, 
                         "model/salary_predictor_model.joblib"))
        self.encoder = joblib.load(
            os.path.join(self.par_dir, 
                         "model/salary_predictor_encoder.joblib"))
        self.lb = joblib.load(
            os.path.join(self.par_dir, 
                         "model/salary_predictor_lb.joblib"))
        
        
    def produce_metrics(self, pred, y):
        
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        accuracy = accuracy_score(y, pred)
        f1 = f1_score(y, pred, zero_division=0)
        
        return [precision, recall, accuracy, f1]
    
    def evaluate_slices(self):
        
        self.slice_metrics_list = []
        self.feature_cat_name_list = []
        
        for split in self.cat_features:
            
            for feature in self.test_data[split].unique():
                
                split_df = self.test_data[self.test_data[split] == feature]
                
                self.X_test, self.y_test, encoder, lb = process_data(
                    split_df, 
                    categorical_features=self.cat_features, 
                    label=self.label, training=False, 
                    encoder=self.encoder, lb=self.lb)
                
                scored_pred = self.model.predict(self.X_test)
                
                metrics = self.produce_metrics(
                    scored_pred, self.y_test)
                
                self.slice_metrics_list.append(metrics)
                
                self.feature_cat_name_list.append(f"{split}_{feature}")
                
    def save_slice_metrics(self):
        
        metrics_df = pd.DataFrame(self.slice_metrics_list)
        metrics_df.index = self.feature_cat_name_list
        metrics_df.columns = ['precision', 'recall', 'accuracy', 'f1_score']
        
        metrics_df.to_csv(
            os.path.join(self.par_dir, "model/model_slice_eval.csv"))
                
if __name__ == '__main__':
    EvaluateDataSlices()
