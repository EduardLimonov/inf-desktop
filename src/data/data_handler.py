from dataclasses import dataclass
from typing import Literal, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from data.utils import CommonDataInfo, LabelEncoderPool, ModelProcessor, ModelHandler, _optim_fn


@dataclass
class OptimResult:
    result_series: pd.Series
    result_node: OptimizeResult


@dataclass
class TableOptimResult:
    result_df: pd.DataFrame
    result_nodes: List[OptimizeResult]


@dataclass
class ProcessHandler:
    _encoder: LabelEncoderPool
    _model_processor: ModelProcessor
    _data_info: CommonDataInfo
    _model_handler: ModelHandler

    def optimization_fn(self, x: np.ndarray, x0: np.ndarray, x_unchangeable: np.ndarray, target_p: float,
                        reg_type: Literal['l1', 'l2'] = 'l1') -> float:
        res = _optim_fn(x, x0, x_unchangeable, target_p, model=self._model_handler, common_di=self._data_info,
                        _type=reg_type)

        if res >= 0.97:
            return 1
        return res

    def decode_data(self, df: pd.DataFrame, rescale: bool = True) -> pd.DataFrame:
        if rescale:
            df = pd.DataFrame(self._model_processor.inverse_process(df), columns=df.columns, index=df.index)

        return self._encoder.decode_df(
            df
        )

    def encode_data(self, df: pd.DataFrame, rescale: bool = True) -> pd.DataFrame:
        df = self._encoder.encode_df(df)
        if rescale:
            return self._model_processor.process(df)
        else:
            return df

    def predict_alive(self, data: pd.DataFrame) -> List[float]:
        data = self.encode_data(data)
        return [self._model_handler.predict_proba_alive(row.reshape(1, -1)) for row in data]

    def __optimize_object(self, obj: pd.Series, method: str = 'Powell', target_p: float = 0.9) -> OptimResult:
        changeable_feat_mask = obj.index.isin(self._data_info.changeable_features)

        x, x_add = obj[changeable_feat_mask].values, obj[~changeable_feat_mask].values

        res = minimize(
            lambda t: 1 - self.optimization_fn(t, x, x_add, target_p),
            x,
            method=method,
            options=dict(ftol=5e-2, xtol=5e-2, return_all=True),
            bounds=[self._data_info.feature_limits[f] for f in self._data_info.changeable_features],
            # tol=1e-2
        )
        x_result = max(res.allvecs, key=lambda v: self.optimization_fn(v, x, x_add, target_p))
        result_v = np.concatenate((x_result, x_add))
        return OptimResult(pd.Series(result_v, index=obj.index, name=obj.name), res)

    def optimize(self, df: pd.DataFrame, method: str = 'Powell', target_p: float = 0.9, pb: Optional = None) \
            -> TableOptimResult:
        if pb is None:
            pb = lambda x: x

        opts = []

        df = pd.DataFrame(self.encode_data(df), columns=df.columns, index=df.index)

        for _, row in pb(df.iterrows()):
            opts.append(self.__optimize_object(row, method=method, target_p=target_p))

        results_df = pd.DataFrame([r.result_series for r in opts])

        return TableOptimResult(self.decode_data(results_df), [r.result_node for r in opts])


class DataHandler(ProcessHandler):
    __objects: pd.DataFrame
    _encoder: LabelEncoderPool
    _model_processor: ModelProcessor
    _data_info: CommonDataInfo
    _model_handler: ModelHandler

    def __init__(self, df: pd.DataFrame, base: ProcessHandler):
        self.__objects = df
        self._encoder, self._model_processor, self._data_info, self._model_handler = \
            base._encoder, base._model_processor, base._data_info, base._model_handler

    def get_decoded_data(self) -> pd.DataFrame:
        return self.__objects.copy()

    def get_encoded_data(self) -> pd.DataFrame:
        return self.encode_data(self.__objects)

    def predict_objects(self) -> List[float]:
        return self.predict_alive(self.__objects)

    def optimize_df(self, method: str = 'Powell', target_p: float = 0.9):
        return self.optimize(self.__objects, method, target_p)
