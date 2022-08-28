"""
Author: Felipe Lana Machado
Date: 01/08/2022
This script is responsible for the fastapi and routing functions
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body

from config import MODEL_DIR, EXAMPLES_DIR
from app.schemas import Person, FeatureInfo


if "DYNO" in os.environ and os.path.isdir("../.dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r ../.dvc ../.apt/usr/lib/dvc")

app = FastAPI(
    title="Census Bureal Classification",
    description="Deploying a ML Model on Heroku with FastAPI.",
    version="1.0.0",
)

model = joblib.load(MODEL_DIR)
with open(EXAMPLES_DIR) as fp:
    examples = yaml.safe_load(fp)


@app.get("/")
async def greetings():
    return "Census Bureal Classification"


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

    pred_label = int(model.predict(df))
    pred_probs = float(model.predict_proba(df)[:, 1])
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'label': pred_label, 'prob': pred_probs, 'salary': pred}
