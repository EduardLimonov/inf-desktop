import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from settings import settings
from settings.defines import defines
from data.utils import LabelEncoderPool, get_train_test_final, get_xy, fit_predictor, ModelProcessor, ModelHandler, \
    CommonDataInfo


NAN_MARK = "NAN"


@dataclass
class XYTables:
    X: pd.DataFrame
    y: pd.Series

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "X": self.X.fillna(NAN_MARK).to_dict(),
            "y": self.y.fillna(NAN_MARK).to_dict()
        }

    @staticmethod
    def from_dict(dct: Dict):
        X = pd.DataFrame.from_dict(dct["X"]).replace(NAN_MARK, np.nan)
        y = pd.Series(dct["y"]).replace(NAN_MARK, np.nan)
        return XYTables(X, y)


@dataclass
class _FitModelResults:
    scaler: StandardScaler
    imputer: KNNImputer
    classifier: RandomForestClassifier

    Xy: XYTables
    Xy_train: XYTables
    Xy_test: XYTables


@dataclass
class System:
    encoder: LabelEncoderPool
    processor: ModelProcessor
    model: ModelHandler
    data_info: CommonDataInfo
    test_data: Optional[XYTables]


def init_all(data_path: str) -> System:
    def replace_diap(inp):
        inp = str(inp).replace(",", ".")
        if "-" in inp:
            diap = inp.split("-")
            diap = [float(d) for d in diap]
            return np.mean(diap)
        else:
            return float(inp)

    def load_data(fname):
        data = pd.read_excel(fname)
        data = data.rename({"Госпитальная летальность": "alive"}, axis=1)
        data["alive"] = data["alive"].apply(lambda x: int(x == "выжившие"))
        data["тропонин"] = data["тропонин"].apply(replace_diap)  # чтобы заменить численный диапазон на число

        cols = sorted(list(data.columns), key=lambda x: 1 if x in defines.changeable_features else 0, reverse=True)
        data = data[cols]

        return data

    data = load_data(data_path)
    fr = _init_results(data)
    encoder = _init_encoder(data[defines.features])

    processor = _init_processor(fr)
    model = _init_model(fr)

    data_info = _init_common_data_info(pd.DataFrame(processor.process(fr.Xy.X), columns=fr.Xy.X.columns),
                                       cat_columns=list(encoder.encoders.keys()))

    return System(encoder, processor, model, data_info, fr.Xy_test)


def _init_common_data_info(x: pd.DataFrame, cat_columns: List[str]) -> CommonDataInfo:
    return CommonDataInfo(
        _get_limits(x, defines.feature_limits),
        defines.recommended_limits,
        defines.changeable_features,
        defines.feature_change_coef,
        list(x.columns),
        cat_columns
    )


def _get_limits(x: pd.DataFrame, config: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    res = dict()
    for f in x.columns:
        if f in config:
            res[f] = config[f]
        else:
            res[f] = (x[f].min(), x[f].max())

    return res


def _init_encoder(df: pd.DataFrame) -> LabelEncoderPool:
    return LabelEncoderPool(df)


def _init_results(df: pd.DataFrame) -> _FitModelResults:
    if settings.fit_model:
        fr = _fit_model(df, train_size=settings.train_size)
        _save_model_results(fr, settings.model_path)
        return fr
    else:
        return _load_model_results(settings.model_path)


def _init_processor(f_model: _FitModelResults) -> ModelProcessor:
    return ModelProcessor(f_model.scaler, f_model.imputer)


def _init_model(f_model: _FitModelResults) -> ModelHandler:
    return ModelHandler(f_model.classifier, columns=list(f_model.Xy.X.columns))


def _fit_model(df_raw: pd.DataFrame, target_column: str = "alive", train_size=0.7) -> _FitModelResults:
    df_raw = df_raw[[*defines.features, target_column]]
    le = _init_encoder(df_raw.drop(target_column, axis=1))
    df = le.encode_df(df_raw)

    data_final_train = df.sample(frac=train_size, random_state=42)
    data_final_test = df.drop(data_final_train.index)

    X, y, X_train, X_test_, y_train, y_test_ = get_train_test_final(data_final_train, num_samples_train=600,
                                                                    train_size=1)

    X_test, y_test = get_xy(data_final_test)

    model = fit_predictor(X_train, y_train, model=RandomForestClassifier(n_estimators=20,
                                                                         class_weight="balanced_subsample"))

    return _FitModelResults(
        model[0],
        model[1],
        model[2],
        XYTables(X, y),
        XYTables(X_train, y_train),
        XYTables(X_test, y_test)
    )


def _save_model_results(results: _FitModelResults, path: str = '../../resource/model_dumps/model.PICKLE'):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(results, f)


def _load_model_results(path: str = '../../resource/model_dumps/model.PICKLE') -> _FitModelResults:
    with open(path, 'rb') as f:
        res = pickle.load(f)

    return res
