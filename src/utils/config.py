"""
This script holds the config data for training model pipeline and running tests
related to the pipeline.

Author: Felipe Lana Machado
Date: 01/08/2022
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TEST_SIZE = 0.3
RANDOM_STATE = 17

MODEL = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE)

PARAM_GRID = None
if isinstance(MODEL, RandomForestClassifier):
    PARAM_GRID = {
        'model__n_estimators': list(range(50, 151, 25)),
        'model__max_depth': list(range(2, 11, 2)),
        'model__min_samples_leaf': list(range(1, 51, 5)),
    }
elif isinstance(MODEL, LogisticRegression):
    PARAM_GRID = {
        'model__C': np.linspace(0.1, 10, 3)
    }

FEATURES = {
    'categorical': [
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'workclass',
        'native_country'
    ],
    'numeric': [
        'age',
        'fnlgt',
        'capital_gain',
        'capital_loss',
        'hours_per_week'
    ],
    'drop': ['education']
}

SLICE_COLUMNS = ['sex', 'race']
