import json
import pickle
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
import pydantic
import requests

from core.http_utils import RESULT_MARK, HTTPBodyDf
from data.initializers import XYTables
from data.utils import CommonDataInfo


class CoreManager:

    url: str = "http://127.0.0.1:8000/"

    def __init__(self):
        pass

    @staticmethod
    def __get_result(result: Dict) -> Any:
        return result[RESULT_MARK]

    def get_columns(self) -> List[str]:
        ans = requests.get(self.url + "get_columns")
        return self.__get_result(
            json.loads(ans.text)
        )

    def check_correct_feature(self, feature_name: str, value: str) -> bool:
        ans = requests.get(
            self.url + "check_correct_feature",
            params=dict(feature_name=feature_name, value=value)
        )
        return bool(self.__get_result(
            json.loads(ans.text)
        ))

    def predict(self, df: pd.DataFrame) -> List[float]:
        ans = requests.post(
            self.url + "predict/",
            data=HTTPBodyDf(dataframe_json=df.to_json()).json()
        )
        return self.__get_result(
            json.loads(ans.text)
        )

    def get_recommendations(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9,
                            _pb: Optional = None) -> pd.DataFrame:
        if _pb is None:
            _pb = lambda x: x

        res = []
        for _, row in _pb(df.iterrows()):
            ans = requests.post(
                self.url + "get_recommendations/",
                data=HTTPBodyDf(dataframe_json=pd.DataFrame([row]).to_json(), method=method, target_p=target_p).json()
            )
            res.append(
                pd.DataFrame.from_dict(json.loads(self.__get_result(
                    json.loads(ans.text)
                )))
            )

        return pd.concat(res)

    def get_test_data(self) -> XYTables:
        ans = requests.get(
            self.url + "get_test_data",
        )
        return XYTables.from_dict(self.__get_result(
            json.loads(ans.text)
        ))

    def get_data_info(self) -> CommonDataInfo:
        ans = requests.get(
            self.url + "get_data_info",
        )
        return CommonDataInfo(**self.__get_result(
            json.loads(ans.text)
        ))

    def get_decoded_limits(self, feature_name: str) -> Tuple[float, float]:
        ans = requests.get(
            self.url + "get_decoded_limits",
            params=dict(feature_name=feature_name)
        )
        return self.__get_result(
            json.loads(ans.text)
        )

    def fill_empty(self, rows: pd.DataFrame, encode_only: bool = False) -> pd.DataFrame:
        ans = requests.post(
            self.url + "fill_empty/",
            params=dict(encode_only=encode_only),
            data=HTTPBodyDf(dataframe_json=rows.to_json()).json()
        )
        return pd.DataFrame.from_dict(json.loads(self.__get_result(
            json.loads(ans.text)
        )))

    @staticmethod
    def set_define(param_name: str, value: str):
        pass

    def get_cat_feature_values(self, feature_name: str) -> List[str]:
        ans = requests.get(
            self.url + "get_cat_feature_values",
            params=dict(feature_name=feature_name)
        )
        return self.__get_result(
            json.loads(ans.text)
        )
