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

@pytest.fixture(scope='session')
def data():
    """
    csv file data tested
    """
    if not os.path.exists(DATA_RAW):
        pytest.fail(f"Data not found at path: {DATA_RAW}")
    
    etl = Etl()
    X_df, y_df = etl.get_clean_data(os.path.join(DATA_RAW, 'census.csv'))
    X_df['salary'] = y_df
    X_df['salary'] = X_df['salary'].map({1: ' >50k', 0: ' <=50k'})

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

    data_df = pd.read_csv(os.path.join(DATA_RAW, 'census.csv'), nrows=10)


    columns = data_df.columns
    columns = [col.replace('-', '_') for col in columns]
    data_df.columns = columns


    data_df = data_df.applymap(
        lambda s: s.lower() if isinstance(s, str) else s)

    data_df['salary'] = data_df['salary'].map({' >50k': 1, ' <=50k': 0,'>50k': 1, '<=50k': 0})

    y_df = data_df.pop('salary')
    X_df = data_df

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=12, stratify=y_df)

    return X_train, X_test, y_train, y_test