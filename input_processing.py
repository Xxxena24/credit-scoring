import joblib
import numpy as np
import pandas as pd


def preprocess(features):

    # Load OneHotEncoder
    ohe = joblib.load('one_hot_encoder.joblib')

    # Load OrdinalEncoder
    ordinal_encoder = joblib.load('ordinal_encoder.joblib')

    column_names = pd.read_csv('german_credit_data.csv', nrows=0).columns
    column_names = column_names.drop('Credit amount')
    input_series = pd.DataFrame([pd.Series(features, index=column_names)])

    input_series.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
    input_series = input_series.replace('nan', 'unknown')
    categorical_columns = ['Sex', 'Housing', 'Checking account', 'Purpose']

    # Load OneHotEncoder
    ohe = joblib.load('one_hot_encoder.joblib')

    # Load OrdinalEncoder
    oe = joblib.load('ordinal_encoder.joblib')

    input_series['Saving accounts'] = oe.transform(input_series[['Saving accounts']])

    # Fit and transform the categorical data
    encoded_data = ohe.transform(input_series[categorical_columns]).toarray()

    # Convert encoded data to DataFrame
    encoded_input_series = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out(categorical_columns))

    # Concatenate with the original DataFrame
    input_series = pd.concat([input_series, encoded_input_series], axis=1).drop(categorical_columns, axis=1)

    # Ensure that any infinite values are converted to NaN
    input_series = input_series.replace([np.inf, -np.inf], np.nan)

    scaler = joblib.load('scaler.joblib')


    input_series = scaler.transform(input_series)

    return input_series





