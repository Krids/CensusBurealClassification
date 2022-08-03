"""
This script holds the conftest data used with pytest module.

Author: Felipe Lana Machado
Date: 01/05/2022
"""
import os
import pytest
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split

from src.utils.project_paths import DATA_RAW
from src.etl.etl import Etl
from src.utils import config

etl = Etl()

@pytest.fixture(scope='session')
def data():
    """
    csv file data tested
    """
    if not os.path.exists(DATA_RAW):
        pytest.fail(f"Data not found at path: {DATA_RAW}")

    X_df, y_df = etl.get_clean_data(os.path.join(DATA_RAW, 'census.csv'))
    X_df['salary'] = y_df
    X_df['salary'] = X_df['salary'].map({1: '>50k', 0: '<=50k'})

    df = ge.from_pandas(X_df)

    return df


@pytest.fixture(scope='session')
def sample_data():
    """
    csv sample data tseted
    Returns:
        X_train: Features train data
        X_test: Features test data
        y_train: Labels train data
        y_test: Labels test data
    """
    if not os.path.exists(DATA_RAW):
        pytest.fail(f"Data not found at path: {DATA_RAW}")

    # X_df, y_df = etl.get_clean_data(os.path.join(DATA_RAW, 'census.csv'))
    # df = pd.concat([X_df, y_df], axis=1)
    # y_df = df['salary']
    # X_df = df.drop(columns=['salary'])

    X, y = etl.get_clean_data(os.path.join(DATA_RAW, 'census.csv'))

    cat_features = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native_country",
        ]
    
    data = pd.concat([X, y], axis=1)

    train, test = train_test_split(data, test_size=0.3, random_state=12)
    
    X_train, y_train, encoder, lb = etl.process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )

    X_test, y_test, encoder, lb = etl.process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
    
    X_train, y_train, X_test, y_test = X_train[:20], y_train[:20], X_test[:20], y_test[:20]

    return X_train, X_test, y_train, y_test
