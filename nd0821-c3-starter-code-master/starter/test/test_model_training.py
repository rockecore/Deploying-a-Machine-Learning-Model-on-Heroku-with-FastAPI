"""
Created on Mon May 18 10:30:00 2022

@author: rbarker

Unit testing for all data prep and model training functions
"""

import sys
import os
sys.path.append("../starter")
import pytest
import unittest
import pandas as pd
import numpy as np
from train_model import split_data, fit_model
from ml.model import compute_model_metrics
import warnings
warnings.filterwarnings("ignore")

class model_training_test(unittest.TestCase):

    """
    
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, base_fixtures):
        
        ## make test dataset
        self.df = pd.DataFrame(
            {"numeric_feat": [3.14, 2.72, 5.62, 5.34, 4.34, 2.32, 2.45, 6.43, 1.65, 7.89],
             "status": ["good", "good", "evil", "evil", "evil", "good", "good", "evil", "good", "evil"],
             "animal": ["dog", "dog", "cat", "cat", "dog", "dog", "dog", "cat", "dog", "cat"]})
        
        ## split test dataset
        self.train, self.test = split_data(self.df, run=False)
        
        ## create test eval data
        self.y = np.array([1,0,1,0,1,0,1,0,1,0,1,0])
        self.preds = np.array([0,1,0,1,0,1,1,0,1,0,1,0])

        ## expected values
        self.sd_train_num_feat = [4.34, 7.89, 2.72, 2.45]
        self.sd_test_num_feat = [5.62, 1.65]
        
        self.cmm_precision = 0.5
        self.cmm_recall = 0.5

    def test_split_data(self):
        
        ## split_data called in fixtures to use for later test methods
        self.assertEqual(self.sd_train_num_feat, list(self.train.iloc[0:4,0]))
        self.assertEqual(self.sd_test_num_feat, list(self.test.iloc[0:3,0]))

    def test_fit_model(self):
        
        cat_features = ["status"]
        model_out = "./test_cache/test_model.pkl"
        label = "animal"
        
        fit_model(self.train, model_out, cat_features=cat_features, label=label)
        
        ## check to see that the model was created
        self.assertTrue(os.path.exists(model_out))

    def test_compute_model_metrics(self):
        
        precision, recall, fbeta = compute_model_metrics(self.y, self.preds)

        self.assertEqual(self.cmm_precision, precision)
        self.assertEqual(self.cmm_recall, recall)


if __name__ == "__main__":
    unittest.main()
