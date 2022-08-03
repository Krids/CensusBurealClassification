"""
This file is responsible for the ETL on the original data,
cleaning and preparing the data for training.

Name: Felipe Lana Machado
Date: 01/05/2022
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


class Etl:

    def __init__(self) -> None:
        pass

    def get_clean_data(self, path):
        """
        Loads and cleans the data from a given path
        """
        data_df = pd.read_csv(path)

        columns = data_df.columns
        columns = [col.replace('-', '_') for col in columns]
        data_df.columns = columns

        data_df = data_df[~data_df.duplicated()]
        data_df.columns = data_df.columns.str.strip()
        data_df = data_df.applymap(
            lambda s: s.lower().strip() if isinstance(s, str) else s)

        data_df['salary'] = data_df['salary'].map({'>50k': 1, '<=50k': 0})

        y_df = data_df.pop('salary')
        x_df = data_df

        return x_df, y_df

    def process_data(
        self, X,
        categorical_features=[], label=None,
        training=True, encoder=None, lb=None
    ):
        """ Process the data used in the machine
        learning pipeline.

        Processes the data using one hot encoding for
        the categorical features and a
        label binarizer for the labels. This can be
        used in either training or
        inference/validation.

        Note: depending on the type of model used, you may
        want to add in functionality that
        scales the continuous data.

        Inputs
        ------
        X : pd.DataFrame
            Dataframe containing the features and label.
            Columns in `categorical_features`
        categorical_features: list[str]
            List containing the names of the categorical
            features (default=[])
        label : str
            Name of the label column in `X`. If None, then
            an empty array will be returned
            for y (default=None)
        training : bool
            Indicator if training mode or
            inference/validation mode.
        encoder:
            sklearn.preprocessing._encoders.OneHotEncoder
            Trained sklearn OneHotEncoder,
            only used if training=False.
        lb : sklearn.preprocessing._label.LabelBinarizer
            Trained sklearn LabelBinarizer,
            only used if training=False.

        Returns
        -------
        X : np.array
            Processed data.
        y : np.array
            Processed labels if labeled=True,
            otherwise empty np.array.
        encoder :
            sklearn.preprocessing._encoders.OneHotEncoder
            Trained OneHotEncoder if training is True,
            otherwise returns the encoder passed in.
        lb :
            sklearn.preprocessing._label.LabelBinarizer
            Trained LabelBinarizer if training is True,
            otherwise returns the binarizer passed in.
        """

        if label is not None:
            y = X[label]
            X = X.drop([label], axis=1)
        else:
            y = np.array([])

        X_categorical = X[categorical_features].values
        X_continuous = X.drop(*[categorical_features], axis=1)

        if training is True:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            lb = LabelBinarizer()
            X_categorical = encoder.fit_transform(X_categorical)
            y = lb.fit_transform(y.values).ravel()
        else:
            X_categorical = encoder.transform(X_categorical)
            try:
                y = lb.transform(y.values).ravel()
            # Catch the case where y is None because we're doing inference.
            except AttributeError:
                pass

        X = np.concatenate([X_continuous, X_categorical], axis=1)
        return X, y, encoder, lb
