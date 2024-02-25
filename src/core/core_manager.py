from __future__ import annotations
import json
import os.path
import subprocess
import time
from typing import List, Tuple, Optional, Dict, Any, Callable

import killport

import pandas as pd
import requests
from pydantic import BaseModel, Field
from pydantic.typing import Annotated

from core.http_utils import RESULT_MARK, HTTPBodyDf
from data.initializers import XYTables
from data.utils import CommonDataInfo
from settings import settings
from settings.network import network_settings


class CoreManager:

    url: Optional[str]
    url_name: Optional[str]
    url_known: Optional[Dict[str, str]]

    _status: Tuple[bool, Optional[Exception]]
    _process_created: Optional[subprocess.Popen]

    _connection_success_fn: Optional[Callable[[str], None]]
    _connection_error_fn: Optional[Callable[[str], None]]

    _errors_to_signal: Tuple[Exception, ...] = \
        requests.ConnectionError, requests.ConnectTimeout, \
        requests.exceptions.MissingSchema, requests.exceptions.InvalidURL

    def __init__(
            self,
            url: Optional[str] = None,
            url_name: Optional[str] = network_settings.DEFAULT_LOCAL_NAME,
            url_known: Optional[Dict[str, str]] = None,
    ):
        self._process_created = None
        self.__start_local_server()

        if url is None:
            url = network_settings.LOCAL_URL

        self.url = url
        self.url_name = url_name
        if url_known is None:
            self.url_known = {url_name: url}
        else:
            self.url_known = url_known

        self._connection_error_fn = None
        self._connection_success_fn = None

        self._status = self.__check_connection(url)

    @staticmethod
    def init_from_factory() -> CoreManager:
        if os.path.exists(settings.core_manager_path):
            return CoreManager.load(settings.core_manager_path)
        else:
            cm = CoreManager()
            cm.save(settings.core_manager_path)
            return cm

    def save(self, path: str = settings.core_manager_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(settings.core_manager_path, "w") as f:
            json.dump(self.__dict(), f, ensure_ascii=False, sort_keys=False, indent=4)

    def __dict(self) -> Dict[str, str]:
        return {v: getattr(self, v) for v in ("url", "url_known", "url_name")}

    @staticmethod
    def load(path: str = settings.core_manager_path) -> CoreManager:
        with open(path, "r") as f:
            return CoreManager(**json.load(f))

    def __start_local_server(self):
        self._process_created = None

        if not CoreManager.check_connection(network_settings.LOCAL_URL, 10):
            self._process_created = subprocess.Popen(network_settings.LOCAL_SERVER_START, shell=True)

            _cnt = 0
            while not CoreManager.check_connection(network_settings.LOCAL_URL):
                _cnt += 0.1
                time.sleep(0.1)
                if _cnt >= network_settings.MAX_SERVER_AWAIT_SEC:
                    break

    def remove(self):
        # may be in __dell__()?
        if self._process_created is not None:
            killport.kill_ports(ports=[network_settings.PORT])

    def add_url(self, url: str, url_name: str) -> bool:
        assert url_name not in self.url_known
        self.url_known[url_name] = url
        self.save()
        return self.check_connection(url)

    def set_url(self, url: Optional[str] = None, url_name: Optional[str] = "") -> bool:
        if url is None and url_name:
            url = network_settings.LOCAL_URL

        status, e = self.__check_connection(url)
        if status:
            if url_name not in self.url_known:
                self.url_known[url_name] = url
            self.url = url
            self.url_name = url_name
            self._status = self.__check_connection(url)
            self.save()
            return True
        else:
            self._connection_error_fn(f": не удалось подключиться к ядру {url}")
            return False

    def remove_url(self, url_name: str) -> bool:
        if url_name == network_settings.DEFAULT_LOCAL_NAME and self.url_known[url_name] == network_settings.LOCAL_URL:
            return False
        else:
            self.url_known.pop(url_name)
            self.save()
            return True

    def get_urls_known(self) -> Dict[str, str]:
        return self.url_known.copy()

    def set_connection_error_fn(self, connection_error_fn: Callable[[str], None]):
        self._connection_error_fn = connection_error_fn

    def set_connection_success_fn(self, connection_success_fn: Callable[[str], None]):
        self._connection_success_fn = connection_success_fn

    @staticmethod
    def __check_connection(url: str) -> Tuple[bool, Optional[Exception]]:
        try:
            return CoreManager.check_connection(url), None
        except CoreManager._errors_to_signal as e:
            return False, e

    @staticmethod
    def __get_result(result: Dict) -> Any:
        return result[RESULT_MARK]

    @staticmethod
    def check_connection(url: str, timeout: float = network_settings.CONN_CHECK_TIMEOUT) -> bool:
        try:
            requests.get(url, timeout=timeout)
            ans = requests.get(url + "get_columns", timeout=timeout)
            return ans.status_code == network_settings.STATUS_OK
        except CoreManager._errors_to_signal:
            return False

    def get_columns(self) -> List[str]:
        try:
            ans = requests.get(self.url + "get_columns")
            self._connection_success_fn("")
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))
            return []

        return self.__get_result(
            json.loads(ans.text)
        )

    def check_correct_feature(self, feature_name: str, value: str) -> bool:
        try:
            ans = requests.get(
                self.url + "check_correct_feature",
                params=dict(feature_name=feature_name, value=value)
            )
            self._connection_success_fn("")
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))
            return False

        return bool(self.__get_result(
            json.loads(ans.text)
        ))

    def predict(self, df: pd.DataFrame) -> List[float]:
        try:
            ans = requests.post(
                self.url + "predict/",
                data=HTTPBodyDf(dataframe_json=df.to_json()).json()
            )
            self._connection_success_fn("")
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))
            return []

        return self.__get_result(
            json.loads(ans.text)
        )

    def get_recommendations(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9,
                            _pb: Optional = None) -> pd.DataFrame:
        if _pb is None:
            _pb = lambda x: x

        res = []
        try:
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
            self._connection_success_fn("")
            return pd.concat(res)
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def get_test_data(self) -> XYTables:
        try:
            ans = requests.get(
                self.url + "get_test_data",
            )
            self._connection_success_fn("")

            return XYTables.from_dict(self.__get_result(
                json.loads(ans.text)
            ))
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def get_data_info(self) -> CommonDataInfo:
        try:
            ans = requests.get(
                self.url + "get_data_info",
            )
            self._connection_success_fn("")

            return CommonDataInfo(**self.__get_result(
                json.loads(ans.text)
            ))
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def get_decoded_limits(self, feature_name: str) -> Tuple[float, float]:
        try:
            ans = requests.get(
                self.url + "get_decoded_limits",
                params=dict(feature_name=feature_name)
            )
            self._connection_success_fn("")

            return self.__get_result(
                json.loads(ans.text)
            )
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def fill_empty(self, rows: pd.DataFrame, encode_only: bool = False) -> pd.DataFrame:
        try:
            ans = requests.post(
                self.url + "fill_empty/",
                params=dict(encode_only=encode_only),
                data=HTTPBodyDf(dataframe_json=rows.to_json()).json()
            )
            self._connection_success_fn("")

            return pd.DataFrame.from_dict(json.loads(self.__get_result(
                json.loads(ans.text)
            )))
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def set_define(self, param_name: str, value: str):
        try:
            requests.get(
                self.url + "set_define",
                params=dict(param_name=param_name, value=value)
            )
            self._connection_success_fn("")

        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))

    def get_cat_feature_values(self, feature_name: str) -> List[str]:
        try:
            ans = requests.get(
                self.url + "get_cat_feature_values",
                params=dict(feature_name=feature_name)
            )
            self._connection_success_fn("")

            return self.__get_result(
                json.loads(ans.text)
            )
        except CoreManager._errors_to_signal as e:
            self._connection_error_fn(str(e))
