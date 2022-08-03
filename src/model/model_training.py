"""
This file is responsible for the training for the model on the Census Bureal Data.

Name: Felipe Lana Machado
Date: 01/05/2022
"""

import os
import logging as log
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from src.etl.etl import Etl

from src.utils import config
from src.utils.project_paths import DATA_PROCESSED, MODELS_PATH, LOGS_PATH

class ModelTraining:

    def __init__(self) -> None:
        self.etl = Etl()
        log.basicConfig(
            filename= os.path.join(LOGS_PATH, 'training.log'),
            level=log.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')

    def get_trained_model(self, filepath=os.path.join(MODELS_PATH, "gbclassifier.pkl")):
        """This method reads the model in the filepath.

        Args:
            filepath (str, optional): Defaults to os.path.join(MODELS_PATH, "gbclassifier.pkl").

        Returns:
            GradientBoostingClassifier: Returns a trained model.
        """        
        model = joblib.load(os.path.join(MODELS_PATH, 'gbclassifier.pkl'))
        return model

    def train_model(self, X_train, y_train, filepath=os.path.join(MODELS_PATH, "gbclassifier.pkl")):
        """
        Trains a machine learning model and returns it.
        Inputs
        ------
        X_train : np.array
            Training data.
        y_train : np.array
            Labels.
        Returns
        -------
        model
            Trained machine learning model.
        """
        gbc = GradientBoostingClassifier(random_state=config.RANDOM_STATE)
        parameters = {"n_estimators": (5, 10),
                    "learning_rate": (0.1, 0.01, 0.001),
                    "max_depth": [2, 3, 4],
                    "max_features": ("auto", "log2")}
        clf = GridSearchCV(gbc, parameters)
        clf.fit(X_train, y_train)
        with open(filepath, 'wb') as file:
            pickle.dump(clf.best_estimator_, file)
        model = clf.best_estimator_
        return model


    def compute_model_metrics(self, y, preds):
        """
        Validates the trained machine learning model using
        precision, recall, and F1.
        Inputs
        ------
        y : np.array
            Known labels, binarized.
        preds : np.array
            Predicted labels, binarized.
        Returns
        -------
        precision : float
        recall : float
        fbeta : float
        """
        fbeta = fbeta_score(y, preds, beta=0.7, zero_division=1)
        precision = precision_score(y, preds, zero_division=1)
        recall = recall_score(y, preds, zero_division=1)
        log.info(f"fbeta : {fbeta}\nprecision : {precision}\nrecall : {recall}")
        return precision, recall, fbeta


    def inference(self, model, X):
        """ Run model inferences and return the predictions.
        Inputs
        ------
        model :
            Trained gradient boosted classifier
        X : np.array
            Data used for prediction.
        Returns
        -------
        predictions : np.array
            Predictions from the model.
        """
        preds = model.predict(X)

        return preds

    def execute(self):
        # Script to train machine learning model.

        # Add code to load in the data.
        data = pd.read_csv(os.path.join(DATA_PROCESSED, "census_cleaned.csv"))

        # Optional enhancement, use K-fold cross validation instead of a train-test split.
        train, test = train_test_split(data, test_size=0.20)

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
        X_train, y_train, encoder, lb = self.etl.process_data(
            train, categorical_features=cat_features, label="salary", training=True
        )
        pickle.dump(lb, open(os.path.join(MODELS_PATH, 'lb.pkl'), "wb"))
        pickle.dump(encoder, open(os.path.join(MODELS_PATH, 'encoder.pkl'), 'wb'))

        # Proces the test data with the process_data function.
        X_test, y_test, encoder, lb = self.etl.process_data(
            test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )
        # Train and save a model.
        log.info("___Train Data___")
        classifier = self.train_model(X_train, y_train)
        y_train_pred = self.inference(classifier, X_train)
        train_precision, train_recall, train_fbeta = self.compute_model_metrics(y_train, y_train_pred)
        log.info("train_precision: {train_precision}, train_recall: {train_recall}, train_fbeta: {train_fbeta}".format(
            train_precision=train_precision, train_recall=train_recall, train_fbeta=train_fbeta))

        log.info("___Test Data___")
        y_test_pred = self.inference(classifier, X_test)
        test_precision, test_recall, test_fbeta = self.compute_model_metrics(y_test, y_test_pred)
        log.info("test_precision: {test_precision}, test_recall: {test_recall}, test_fbeta: {test_fbeta}".format(
            test_precision=test_precision, test_recall=test_recall, test_fbeta=test_fbeta))