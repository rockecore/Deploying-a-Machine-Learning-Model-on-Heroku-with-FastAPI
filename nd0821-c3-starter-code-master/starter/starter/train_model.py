# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd

from .ml.data import process_data
from .ml.model import train_model
from joblib import dump

# Add the necessary imports for the starter code.

def split_data(data_dir):
    # Add code to load in the data.
    data = pd.read_csv(data_dir)
    
    train, test = train_test_split(data, test_size=0.20)
    
    return train, test


def fit_model(train, model_out):


    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"]
    
    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True)

    # Train and save a model.
    model = train_model(X_train, y_train)
    
    # Dump it
    dump(model, model_out)



