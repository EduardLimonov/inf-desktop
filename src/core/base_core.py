from abc import ABC, abstractmethod
from typing import Dict, Callable, Optional, List, Tuple

import pandas as pd

from data.initializers import XYTables
from data.utils import CommonDataInfo
from settings.network import network_settings


class CoreInterface(ABC):

    @abstractmethod
    def url_switch_is_enable(self) -> bool:
        pass

    @abstractmethod
    def shutdown_server_if_started(self):
        pass

    @staticmethod
    @abstractmethod
    def force_remove_server():
        pass

    @abstractmethod
    def get_status(self) -> Tuple[bool, Optional[Exception]]:
        pass

    @abstractmethod
    def add_url(self, url: str, url_name: str) -> bool:
        pass

    @abstractmethod
    def set_url(self, url: Optional[str] = None, url_name: Optional[str] = "") -> bool:
        pass

    @abstractmethod
    def remove_url(self, url_name: str) -> bool:
        pass

    @abstractmethod
    def get_url(self) -> str:
        pass

    @abstractmethod
    def get_url_name(self) -> str:
        pass

    @abstractmethod
    def get_urls_known(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def set_connection_error_fn(self, connection_error_fn: Callable[[str], None]):
        pass

    @abstractmethod
    def set_connection_success_fn(self, connection_success_fn: Callable[[str], None]):
        pass

    @staticmethod
    @abstractmethod
    def check_connection(url: str, timeout: float = network_settings.CONN_CHECK_TIMEOUT) -> bool:
        pass

    @abstractmethod
    def get_columns(self) -> List[str]:
        pass

    @abstractmethod
    def check_correct_feature(self, feature_name: str, value: str) -> bool:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> List[float]:
        pass

    @abstractmethod
    def get_recommendations(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9,
                            _pb: Optional = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_test_data(self) -> XYTables:
        pass

    @abstractmethod
    def get_data_info(self) -> CommonDataInfo:
        pass

    @abstractmethod
    def get_decoded_limits(self, feature_name: str) -> Tuple[float, float]:
        pass

    @abstractmethod
    def fill_empty(self, rows: pd.DataFrame, encode_only: bool = False) -> pd.DataFrame:
        pass

    @abstractmethod
    def set_define(self, param_name: str, value: str):
        pass

    @abstractmethod
    def get_cat_feature_values(self, feature_name: str) -> List[str]:
        pass
