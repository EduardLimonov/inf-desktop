import json

import pandas as pd
from fastapi import FastAPI
import sys
sys.path.append("src")

from core.core import Core
from core.http_utils import HTTPBodyDf, RESULT_MARK

core = Core()
app = FastAPI()


@app.get("/get_columns")
async def get_columns():
    return {RESULT_MARK: core.get_columns()}


@app.get("/check_correct_feature")
async def check_correct_feature(feature_name: str, value: str):
    return {RESULT_MARK: core.check_correct_feature(feature_name, value)}


@app.post("/predict/")
async def predict(body: HTTPBodyDf):
    return {RESULT_MARK: core.predict(pd.DataFrame.from_dict(json.loads(body.dataframe_json)))}


@app.post("/get_recommendations/")
async def get_recommendations(body: HTTPBodyDf):
    return {RESULT_MARK: core.get_recommendations(
        json.loads(body.dataframe_json),
        body.method,
        body.target_p
    ).to_json()}


@app.get("/get_test_data")
async def get_test_data():
    return {RESULT_MARK: core.get_test_data().to_dict()}


@app.get("/get_data_info")
async def get_data_info():
    return {RESULT_MARK: core.get_data_info()}


@app.get("/get_decoded_limits")
async def get_decoded_limits(feature_name: str):
    return {RESULT_MARK: (core.get_decoded_limits(feature_name))}


@app.post("/fill_empty/")
async def fill_empty(body: HTTPBodyDf, encode_only: bool):
    return {RESULT_MARK: core.fill_empty(
        pd.DataFrame.from_dict(json.loads(body.dataframe_json)),
        encode_only
    ).to_json()}


@app.get("/set_define")
async def set_define(param_name: str, value: str):
    core.set_define(param_name, value)
    return {RESULT_MARK: "OK"}


@app.get("/get_cat_feature_values")
async def get_cat_feature_values(feature_name: str):
    return {RESULT_MARK: core.get_cat_feature_values(feature_name)}
