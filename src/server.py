import json
import pickle

from fastapi import FastAPI

from core.core import Core
from core.server_utils import HTTPBodyDf

core = Core()
app = FastAPI()


@app.get("/get_columns")
async def get_columns():
    return {"result": core.get_columns()}


@app.get("/check_correct_feature")
async def check_correct_feature(feature_name: str, value: str):
    return {"result": core.check_correct_feature(feature_name, value)}


@app.post("/predict/")
async def predict(body: HTTPBodyDf):
    return {"result": core.predict(json.loads(body.dataframe_json))}


@app.post("/get_recommendations/")
async def get_recommendations(body: HTTPBodyDf):
    return {"result": core.get_recommendations(
        json.loads(body.dataframe_json),
        body.method,
        body.target_p
    ).to_json()}


@app.get("/get_test_data")
async def get_test_data():
    return {"result": pickle.dumps(core.get_test_data())}


@app.get("/get_data_info")
async def get_data_info():
    return {"result": pickle.dumps(core.get_data_info())}


@app.get("/get_decoded_limits")
async def get_decoded_limits(feature_name: str):
    return {"result": (core.get_decoded_limits(feature_name))}


@app.post("/fill_empty/")
async def fill_empty(body: HTTPBodyDf, encode_only: bool):
    return {"result": core.fill_empty(
        json.loads(body.dataframe_json),
        encode_only
    ).to_json()}


@app.get("/set_define")
async def set_define(param_name: str, value: str):
    core.set_define(param_name, value)
    return {"result": "OK"}
