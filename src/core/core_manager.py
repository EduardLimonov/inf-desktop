import json
from typing import List, Tuple, Optional

import pandas as pd
import requests

from data.initializers import XYTables
from data.utils import CommonDataInfo


class CoreManager:

    url: str = "https://127.0.0.1"

    def __init__(self):
        pass

    def get_columns(self) -> List[str]:
        print(json.loads(str(requests.get(
                self.url
            ).content)))
        return json.loads(str(requests.get(
                self.url
            ).content))

    def check_correct_feature(self, feature_name: str, value: str) -> bool:
        pass

    def predict(self, df: pd.DataFrame) -> List[float]:
        pass

    def get_recommendations(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9,
                            _pb: Optional = None) -> pd.DataFrame:
        if _pb is None:
            _pb = lambda x: x

        pass

    def get_test_data(self) -> XYTables:
        pass

    def get_data_info(self) -> CommonDataInfo:
        pass

    def get_decoded_limits(self, feature_name) -> Tuple[float, float]:
        pass

    def fill_empty(self, rows: pd.DataFrame, encode_only: bool = False) -> pd.DataFrame:
        pass

    @staticmethod
    def set_define(param_name: str, value: str):
        pass
