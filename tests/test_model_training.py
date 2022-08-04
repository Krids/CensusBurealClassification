"""
This script holds the functions to test the training file.

Author: Felipe Lana Machado
Date: 01/08/2022
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.etl.etl import Etl
from src.utils.project_paths import MODELS_PATH, DATA_RAW
from src.model.model_training import ModelTraining

etl = Etl()
model_training = ModelTraining()


def test_model_output_shape(sample_data: pd.DataFrame):
    """
    Test model predictions are of correct shape
    Args:
        sample_data (pd.DataFrame): Sample data to be tested
    """
    X_train, X_test, y_train, _ = sample_data

    model = joblib.load(os.path.join(MODELS_PATH,
                                     'gbclassifier.pkl'))

    y_train_pred = model_training.inference(model, X_train)
    y_test_pred = model_training.inference(model, X_test)

    assert X_train.shape[
        1] == 108, f"Train data number of columns should be\
             108 not {X_train.shape[1]}"
    assert X_test.shape[
        1] == 108, f"Test data number of columns should be\
             108 not {X_test.shape[1]}"
    assert y_train_pred.shape[0] == X_train.shape[
        0], f"Predictions output shape {y_train_pred.shape[0]} is \
            incorrect does not match input shape {X_train.shape[0]}"
    assert y_test_pred.shape[0] == X_test.shape[
        0], f"Predictions output shape {y_test_pred.shape[0]} is \
            incorrect does not match input shape {X_test.shape[0]}"


def test_model_output_range(sample_data: pd.DataFrame):
    """
    Test model predictions are within range 0-1
    Args:
        sample_data (pd.DataFrame): [description]
    """
    X_train, X_test, y_train, _ = sample_data

    model = joblib.load(os.path.join(MODELS_PATH,
                                     'gbclassifier.pkl'))

    y_train_pred = model_training.inference(model, X_train)
    y_test_pred = model_training.inference(model, X_test)

    assert (y_train_pred >= 0).all() & (y_train_pred <=
                                        1).all(), "Predictions \
                                            output range is not from 0-1"
    assert (y_test_pred >= 0).all() & (y_test_pred <= 1).all(
    ), "Predictions output range is not from 0-1"


def test_model_evaluation():
    """
    Test evaluated model metrics are above certain thresholds
    """
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
        test, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder,
        lb=lb
    )

    model = joblib.load(os.path.join(MODELS_PATH,
                                     'gbclassifier.pkl'))

    y_train_pred = model_training.inference(model, X_train)
    y_test_pred = model_training.inference(model, X_test)

    pre_train, rec_train, f1_train = model_training.compute_model_metrics(
        y_train_pred, y_train)
    pre_test, rec_test, f1_test = model_training.compute_model_metrics(
        y_test_pred, y_test)

    assert pre_train > 0.44, "Train precision should be above 0.45"
    assert rec_train > 0.81, "Train recall should be above 0.82"
    assert f1_train > 0.52, "Train f1 should be above 0.53"

    assert pre_test > 0.45, "Test precision should be above 0.46"
    assert rec_test > 0.82, "Test recall should be above 0.83"
    assert f1_test > 0.53, "Test f1 should be above 0.54"
