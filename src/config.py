"""
This script is responsible for the config data for training model pipeline and running tests
related to the pipeline

Author: Felipe Lana Machado
Date: 01/08/2022
"""
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

TEST_SIZE = 0.3
RANDOM_STATE = 17
__MAIN_DIR = Path(__file__).parent.parent.absolute()

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

__DATA_FILE = 'census.csv'
__MODEL_FILE = 'pipe_' + MODEL.__class__.__name__
__EVAL_FILE = f'model_evaluation_{MODEL.__class__.__name__}.txt'
__SLICE_FILE = f'slice_output_{MODEL.__class__.__name__}.txt'


DATA_DIR = os.path.join(__MAIN_DIR, 'data', 'raw', __DATA_FILE)
MODEL_DIR = os.path.join(__MAIN_DIR, 'docs', 'models', __MODEL_FILE)
EVAL_DIR = os.path.join(__MAIN_DIR, 'docs', 'metrics', __EVAL_FILE)
SLICE_DIR = os.path.join(__MAIN_DIR, 'docs', 'metrics', __SLICE_FILE)
PLOT_DIR = os.path.join(__MAIN_DIR, 'docs', 'plots')
EXAMPLES_DIR = os.path.join(__MAIN_DIR, 'src', 'app', 'examples_api.yaml')
