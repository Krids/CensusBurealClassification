"""
This script holds the conftest data used with pytest module.

Author: Felipe Lana Machado
Date: 01/08/2022
"""

import os
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body

from src.utils.project_paths import MODELS_PATH, EXAMPLES_API_PATH
from src.api.model import Person, FeatureInfo
from src.etl.etl import Etl

if "DYNO" in os.environ and os.path.isdir("../.dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r ../.dvc ../.apt/usr/lib/dvc")

app = FastAPI(
    title="Census Bureal Classification",
    description="Deploying a ML Model on Heroku with FastAPI",
    version="1.0.0",
)

model = joblib.load(os.path.join(MODELS_PATH, 'gbclassifier.pkl'))
with open(EXAMPLES_API_PATH) as fp:
    examples = yaml.safe_load(fp)
etl = Etl()


@app.get("/")
async def greetings():
    return "Greetings everybody!!"


@app.get("/feature_info/{feature_name}")
async def feature_info(feature_name: FeatureInfo):

    info = examples['features_info'][feature_name]
    return info


@app.post("/predict/")
async def predict(person: Person = Body(..., examples=examples['post_examples'])):

    person = person.dict()
    features = np.array([person[f]
                        for f in examples['features_info'].keys()]).reshape(1, -1)
    df = pd.DataFrame(features, columns=examples['features_info'].keys())

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

    encoder = joblib.load(os.path.join(MODELS_PATH, 'encoder.pkl'))
    lb = joblib.load(os.path.join(MODELS_PATH, 'lb.pkl'))

    X, _, _, _ = etl.process_data(
            df, categorical_features=cat_features, training=False, encoder=encoder , lb=lb
        )

    pred_label = int(model.predict(X))
    pred_probs = float(model.predict_proba(X)[:, 1])
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'label': pred_label, 'prob': pred_probs, 'salary': pred}