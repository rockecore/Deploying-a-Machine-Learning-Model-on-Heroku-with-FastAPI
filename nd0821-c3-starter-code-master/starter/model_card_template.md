# Model Card
This ML model is used to predict salaries from Census Bureau data.

## Model Details
This is a random forest classification model with default hyperparameters.

## Intended Use
Use this model to predict whether or not an employee will make more or less than 50k/year from the Census Bureau data.

## Training Data
Data was obtained from https://archive.ics.uci.edu/ml/datasets/census+income. 

The full dataset contains 32,561 samples with employee statistics that include age, workclass, education, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, and native_country. Data was cleaned by removing spaces and dashes, in addition to the column labels.

## Evaluation Data
The evaluation data was 20% of the full dataset.

## Metrics
For this model the precision is 0.7341211225997046, recall is 0.6231974921630095, and fbeta is 0.6741268226517464

## Ethical Considerations
Starter code provided by Udacity and data provided by the UCI ML Repository. Give credit where credit is due.

## Caveats and Recommendations
Data needs cleaning before implementation. There were a lot of spaces and the dashes were replaced with underscores. 