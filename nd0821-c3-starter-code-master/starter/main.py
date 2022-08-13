"""
To run pipeline locally
"""

from starter.train_model import split_data, fit_model, test_model


cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country"]


if __name__ == '__main__':
    
    data_dir = "data/clean_census.csv"
    model_out = "model/"
    
    train, test = split_data(data_dir)
    fit_model(model_out, train=train)
    precision, recall, fbeta = test_model(model_out, 
                                          test, 
                                          cat_features=cat_features)
    print(f"Fbeta: {fbeta}, Precision: {precision}, Recall: {recall}")