from typing import List, Tuple, Optional, Callable, Dict

import pandas as pd

from core.base_core import CoreInterface
from data.data_handler import ProcessHandler
from data.initializers import init_all, System, XYTables
from data.utils import CommonDataInfo
from settings import settings
from settings.defines import defines
from settings.network import network_settings


class Core(CoreInterface):
    system: System

    _core: ProcessHandler

    def __init__(self, path: str = settings.dataset_path):
        self.system = init_all(path)
        self._core = ProcessHandler(
            self.system.encoder, self.system.processor, self.system.data_info, self.system.model
        )

    def url_switch_is_enable(self) -> bool:
        return False

    def get_status(self) -> Tuple[bool, Optional[Exception]]:
        return True, None

    def shutdown_server_if_started(self):
        pass

    @staticmethod
    def force_remove_server():
        pass

    def add_url(self, url: str, url_name: str) -> bool:
        return False

    def set_url(self, url: Optional[str] = None, url_name: Optional[str] = "") -> bool:
        return True

    def remove_url(self, url_name: str) -> bool:
        return False

    def get_url(self) -> str:
        return network_settings.NON_SERVER_CORE_URL

    def get_url_name(self) -> str:
        return network_settings.NON_SERVER_CORE_NAME

    def get_urls_known(self) -> Dict[str, str]:
        return {network_settings.NON_SERVER_CORE_NAME: network_settings.NON_SERVER_CORE_URL}

    def set_connection_error_fn(self, connection_error_fn: Callable[[str], None]):
        pass

    def set_connection_success_fn(self, connection_success_fn: Callable[[str], None]):
        pass

    @staticmethod
    def check_connection(url: str, timeout: float = network_settings.CONN_CHECK_TIMEOUT) -> bool:
        return True

    def get_columns(self) -> List[str]:
        return self.system.data_info.num_to_column_mapper

    def check_correct_feature(self, feature_name: str, value: str) -> bool:
        if feature_name in self.system.data_info.cat_features:
            return value in self.system.encoder.encoders[feature_name].classes_
        else:
            if not value.replace('.', '', 1).isdigit():
                return False
            value = float(value)
            l, h = self.get_decoded_limits(feature_name)
            return l <= value <= h

    def predict(self, df: pd.DataFrame) -> List[float]:
        return self._core.predict_alive(df)

    def get_recommendations(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9,
                            _pb: Optional = None) -> pd.DataFrame:
        res = self._core.optimize(df, method, target_p, _pb)
        return res.result_df

    def get_test_data(self) -> XYTables:
        X = self._core.decode_data(self.system.test_data.X, rescale=False)
        return XYTables(X, self.system.test_data.y)

    def get_data_info(self) -> CommonDataInfo:
        return self.system.data_info

    def get_decoded_limits(self, feature_name: str) -> Tuple[float, float]:
        min_l, max_l = self.system.data_info.feature_limits[feature_name]
        min_l, max_l = self.system.processor.inverse_process(
            [
                [min_l] * len(self.system.data_info.feature_limits),
                [max_l] * len(self.system.data_info.feature_limits),
            ]
        )[:, self.system.data_info.num_to_column_mapper.index(feature_name)]
        return min_l, max_l

    def fill_empty(self, rows: pd.DataFrame, encode_only: bool = False) -> pd.DataFrame:
        if encode_only:
            return self._core.encode_data(rows[self.system.data_info.num_to_column_mapper], rescale=False)

        df_enc_resc_fill = self._core.encode_data(rows[self.system.data_info.num_to_column_mapper], rescale=True)
        df_enc_resc_fill = pd.DataFrame(
            df_enc_resc_fill, columns=self.system.data_info.num_to_column_mapper, index=rows.index
        )

        df_dec = self._core.decode_data(df_enc_resc_fill, rescale=True)
        return df_dec

    def set_define(self, param_name: str, value: str):
        assert hasattr(defines, param_name), f"Defines settings has not attribute {param_name}"

        defines.set_attr(param_name, value)

    def get_cat_feature_values(self, feature_name: str) -> List[str]:
        return list(self.system.encoder.encoders[feature_name].classes_)
