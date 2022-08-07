# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append(".")
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
import joblib

# Add the necessary imports for the starter code.

def split_data(data_dir, run=True):
    # Add code to load in the data.
    
    if run:
        data = pd.read_csv(data_dir)
    else:
        data = data_dir
    
    train, test = train_test_split(data, test_size=0.20, random_state=0)
    
    return train, test


def fit_model(model_out, train=None, data_dir=None, cat_features=None, label="salary"):

    if cat_features is None:
        cat_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country"]
        
    if train is None:
        train, test = split_data(data_dir)
    
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True)

    # Train and save a model.
    model = train_model(X_train, y_train)
    
    # Dump it
    joblib.dump(model, model_out+"salary_predictor_model.joblib")
    joblib.dump(encoder, model_out+"salary_predictor_encoder.joblib")
    joblib.dump(lb, model_out+"salary_predictor_lb.joblib")
    
#     # test it
#     X_test, y_test, _, _ = process_data(
#         test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb)

#     preds = inference(model, X_test)
#     precision, recall, fbeta = compute_model_metrics(y_test, preds)
#     print(f"precision is {precision}, recall is {recall}, fbeta is {fbeta}")

# model_out = r"C:\Users\rbarker\OneDrive - Imdex Limited\Documents\Python Scripts\udacity_training\Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI\nd0821-c3-starter-code-master\starter\model/"
# data_dir = r"C:\Users\rbarker\OneDrive - Imdex Limited\Documents\Python Scripts\udacity_training\Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI\nd0821-c3-starter-code-master\starter\data/clean_census.csv"

# fit_model(model_out, data_dir=data_dir)


# loaded_model = joblib.load(open(model_out, 'rb'))

# from ml.model import inference, compute_model_metrics

# preds = inference()
