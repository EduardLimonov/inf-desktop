from dataclasses import dataclass
from itertools import product
from math import floor, ceil
from typing import Dict, Tuple, List, Literal, Union, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

CLF_MODEL = RandomForestClassifier
MAX_NOISE_K = 4
INFINITY = 1e5


@dataclass
class CommonDataInfo:
    feature_limits: Dict[str, Tuple[float, float]]
    recommended_limits: Dict[str, Tuple[float, float]]
    changeable_features: List[str]
    feature_change_coef: Dict[str, float]
    num_to_column_mapper: List[str]
    cat_features: List[str]

    @staticmethod
    def get_normalized_distance(distance: float) -> float:
        return distance / 1e3

    def distance_metric(self, x1: np.ndarray, x2: np.ndarray, _type: Literal['l1', 'l2'] = 'l1') -> float:
        delta = abs(x1 - x2)
        change_coef_vector = np.array(
            [self.feature_change_coef[col] for col in self.changeable_features]
        )
        delta *= change_coef_vector

        if _type == 'l1':
            res = delta
        else:  # elif _type == 'l2':
            res = delta * delta

        return res.sum()

    def check_feature_limits(self, obj: Union[np.ndarray, pd.Series]) -> bool:
        if type(obj) == np.ndarray:
            obj = pd.Series(obj, index=self.num_to_column_mapper)

        for c in obj.index:
            if not self.feature_limits[c][0] <= obj[c] <= self.feature_limits[c][1]:
                return False

        return True


class BaseHandler:
    _columns: List[str]

    def _fit(self, df: pd.DataFrame):
        self._columns = list(df.columns)


class LabelEncoderPool(BaseHandler):
    encoders: Dict[str, LabelEncoder]
    is_fitted: bool
    _columns: List[str]

    def __init__(self, df_to_fit: pd.DataFrame):
        self.encoders = dict()
        self.is_fitted = False

        self._fit(df_to_fit)

    def _fit(self, df: pd.DataFrame) -> None:
        super()._fit(df)

        cat_columns = list(df.columns[(df.dtypes != np.float_) & (df.dtypes != np.int_)])

        for c in cat_columns:
            self.encoders[c] = LabelEncoder().fit(df[c])

        self.is_fitted = True

    def encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_fitted, "Error: LabelEncoderPool is not fitted yet, but tries to encode df..."

        df = df.copy()
        for cat_col in self.encoders:
            df[cat_col] = self.encoders[cat_col].transform(df[cat_col])

        return df

    def decode_df(self, df: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        assert self.is_fitted, "Error: LabelEncoderPool is not fitted yet, but tries to encode df..."

        if type(df) == np.ndarray:
            df = pd.DataFrame(df, columns=self._columns)

        df = df.copy()
        for cat_col in self.encoders:
            df[cat_col] = self.encoders[cat_col].inverse_transform(np.round(df[cat_col]).astype(int))

        return df


class ModelProcessor:
    processor: Optional[StandardScaler]  # have .transform, .inverse_transform methods
    imputer: Optional[KNNImputer]

    def __init__(self, processor: Optional[StandardScaler] = None, imputer: Optional[KNNImputer] = None):
        self.set_model(processor, imputer)

    def set_model(self, processor: Optional[StandardScaler] = None, imputer: Optional[KNNImputer] = None):
        self.processor = processor
        self.imputer = imputer

    @staticmethod
    def _process(x, model):
        if model is None:
            return x
        else:
            return model.transform(x)

    def process(self, x):
        return self._process(
            self._process(x, self.processor),
            self.imputer
        )

    def inverse_process(self, x):
        if self.processor is not None:
            return self.processor.inverse_transform(x)
        else:
            return x


class ModelHandler(BaseHandler):
    model: CLF_MODEL
    columns: List[str]

    def __init__(self, model: CLF_MODEL, columns: Iterable):
        self.model = model
        self.columns = list(columns)

    def _predict_proba_alive(self, x: np.ndarray) -> float:
        assert len(x.shape) == 2 and x.shape[0] == 1, f"x shape should be (1, N_feats), got: {x.shape}"

        return self.model.predict_proba(x)[0, 1]

    def __calc_p_for_variant(self, x: np.ndarray, coord: Tuple[float], cols_to_vary: List[int]) -> float:
        x = x.copy()
        for coord_i, i in enumerate(cols_to_vary):
            x[i] = coord[coord_i]

        return self._predict_proba_alive(x.reshape(1, -1))

    def _vary(self, x: np.ndarray, cat_columns_to_vary: List[str]) -> float:
        assert cat_columns_to_vary is not None, "Error: cat_columns_to_vary is None"

        points_vary = []
        cols_vary = []

        for cat_col in cat_columns_to_vary:
            ci = self.columns.index(cat_col)  # cringe way, but ok(

            x1 = floor(x[ci])
            x2 = ceil(x[ci])

            if x1 == x2:
                continue

            points_vary.append((x1, x2))
            cols_vary.append(ci)

        # print(cat_columns_to_vary, cols_vary, points_vary, list(product(*points_vary)), sep="\n\n")
        p_alive_vary = {
            coord: self.__calc_p_for_variant(x, coord, cols_vary)
            for coord in product(*points_vary)
        }

        return rec(x, tuple(), p_alive_vary, coords_left=cols_vary, points_vary=points_vary)

    def predict_proba_alive(self, x: np.ndarray, cat_columns_to_vary: Optional[List[str]] = None,
                            smooth: bool = False) -> float:
        if cat_columns_to_vary is None or not smooth:
            return self._predict_proba_alive(x)

        # else: we should use linear smoothing for categorical features: vary each feature in (floor(f), ceil(f)) and
        # get estimation

        return self._vary(x, cat_columns_to_vary)


def at(p_lower, p_upper, t: float) -> float:
    t = t % 1
    return t * p_lower + (1 - t) * p_upper


def rec(
        x: np.ndarray,
        fixed_coords: Tuple[int, ...],
        p_alive_vary: Dict[Tuple[int, ...], float],
        coords_left: List[int],
        points_vary: List[Tuple[int, int]]
) -> float:
    l_left = len(coords_left)

    if l_left == 0:
        return p_alive_vary[fixed_coords]

    coords_lower = (*fixed_coords, points_vary[l_left - 1][0])
    coords_upper = (*fixed_coords, points_vary[l_left - 1][1])
    t = x[coords_left[l_left - 1]]

    return at(
        rec(
            x,
            coords_lower,
            p_alive_vary,
            coords_left[: l_left - 1],
            points_vary
        ),
        rec(
            x,
            coords_upper,
            p_alive_vary,
            coords_left[: l_left - 1],
            points_vary
        ),
        t
    )


def __alive_p(x: np.ndarray, model: ModelHandler, target_p: float, cat_columns_to_vary: List[str]) -> float:
    p = model.predict_proba_alive(x, cat_columns_to_vary=cat_columns_to_vary, smooth=True)
    if p >= target_p:
        p = min(p, model.predict_proba_alive(x.reshape(1, -1)))

    if p >= target_p:
        return 1
    else:
        return p


def _optim_fn(x: np.ndarray, x0: np.ndarray, x_unchangeable: np.ndarray, target_p: float, model: ModelHandler,
              common_di: CommonDataInfo, _type: Literal['l1', 'l2'] = 'l1') -> float:
    if not __check_borders(x, common_di.feature_limits, model.columns):
        return -INFINITY

    alive_p = __alive_p(np.concatenate((x, x_unchangeable)), model, target_p,
                        cat_columns_to_vary=[f for f in common_di.cat_features if f in common_di.changeable_features])
    distance_penalty = CommonDataInfo.get_normalized_distance(common_di.distance_metric(x, x0, _type))

    return alive_p - distance_penalty


def __check_borders(x: np.ndarray, borders: Dict[str, Tuple[float, float]], num_to_col_mapper) -> bool:
    for i in range(len(x)):
        col = num_to_col_mapper[i]
        if not borders[col][0] <= x[i] <= borders[col][1]:
            return False

    return True


def resample(data_to_sample: pd.DataFrame, n_samples, noises=None):
    return data_to_sample.sample(n_samples, weights=noises)


def get_xy(df: pd.DataFrame):
    return df.drop("alive", axis=1), df["alive"]


def get_noise_coef(point: Union[pd.Series, np.ndarray], data: pd.DataFrame, exp_multiplier=2, scaler=None):
    # коэффициент шума: чем больше значение, там больше можно добавить шума
    cols = [c for c in data.columns if c != "alive"]

    p_alive = point["alive"]
    X, y = get_xy(data)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X.values)

    data = scaler.transform(X.values)

    point = scaler.transform([point.drop("alive")])[0]

    point, data = pd.Series(point, index=cols), pd.DataFrame(data, columns=cols)
    data["alive"] = y.values

    data = data[data["alive"] != p_alive]
    data = data.drop("alive", axis=1)

    def distance2(x, y):
        # один из аргументов -- таблица (len(shape) == 2)
        delta = x - y
        return (delta * delta).sum(axis=1)

    d = distance2(point, data) + 1e-8
    restrict_noise = (1 / d).sum()
    res = 1 / (restrict_noise + 1e-4)

    return np.exp(res * exp_multiplier)  # np.power(res, 10)


def noise(arr: pd.DataFrame, noise_alpha: np.array, cat_columns_nums=None):
    std = arr.std()
    if cat_columns_nums is not None:
        cat_columns_nums = [num for num in cat_columns_nums if num < len(std)]
        std[cat_columns_nums] = 0

    noises = (np.random.normal(0, std, size=(len(arr), len(std))).transpose() * noise_alpha).transpose()

    return arr + noises


def noise_postprocess(before_df: pd.DataFrame, after_df: pd.DataFrame, cat_columns) -> pd.DataFrame:
    for c in after_df.columns:
        if c in cat_columns or c == "alive":
            continue
        after_df[c] = np.max((after_df[c], np.full_like(after_df[c], before_df[c].min())), axis=0)
        after_df[c] = np.min((after_df[c], np.full_like(after_df[c], before_df[c].max())), axis=0)

    return after_df


def upsample(df: pd.DataFrame, num_samples, label=None, cat_columns_nums=None, noise_multiplicator=0.8,
             exp_multiplier=4, cat_columns=None):
    if cat_columns is None:
        cat_columns = list(df.columns[(df.dtypes != np.float_) & (df.dtypes != np.int_)])

    df.reset_index(drop=True, inplace=True)
    if num_samples <= 0:
        return df

    if cat_columns_nums is None:
        tX = df  # .drop("alive", axis=1)
        cat_columns_nums = np.where(tX.columns.isin(list(cat_columns) + ["alive"]))[0]

    data_to_sample = df if label is None else df[df["alive"] == label]
    weights = data_to_sample.apply(
        lambda x: get_noise_coef(
            x, df, exp_multiplier=exp_multiplier,
            scaler=StandardScaler().fit(df.drop("alive", axis=1).values)
        ),
        axis=1)

    # res = data_to_sample.sample(n_samples, weights=noises)
    p = weights * weights
    p /= p.sum()
    res = data_to_sample.loc[np.random.choice(data_to_sample.index, size=num_samples, p=p)]

    noises = weights.loc[res.index]  # res.apply(lambda x: get_noise_coef(x, df), axis=1)

    noise_alpha = np.min(
        [noises, np.full_like(noises, MAX_NOISE_K)], 0
    )
    res = noise(res, noise_alpha * noise_multiplicator, cat_columns_nums=cat_columns_nums)
    res = noise_postprocess(df, res, cat_columns)

    res = pd.concat((df, res))

    return res


def get_split(data: pd.DataFrame, num_samples_train=500, num_samples_test=0, train_size=0.8):
    data_train = data.sample(frac=train_size, random_state=42)
    data_test = data.drop(data_train.index)

    if len(data_train):
        data_train = upsample(data_train, num_samples=num_samples_train)
        class_quants_train = np.bincount(data_train["alive"])
        delta_train = class_quants_train[1] - class_quants_train[0]
        data_train = upsample(data_train, num_samples=delta_train, label=0)

    if len(data_test):
        data_test = upsample(data_test, num_samples=num_samples_test)
        class_quants_test = np.bincount(data_test["alive"])
        delta_test = class_quants_test[1] - class_quants_test[0]
        # data_test = upsample(data_test, num_samples=delta_test, label=0)

    X, y = get_xy(data)
    X_train, y_train = get_xy(data_train)
    X_test, y_test = get_xy(data_test)

    return X, y, X_train, X_test, y_train, y_test


def get_train_test_final(data, num_samples_train=500, num_samples_test=0, train_size=0.8):
    X, y, X_train, X_test, y_train, y_test = get_split(data, num_samples_train, num_samples_test, train_size)
    # X_test, y_test = get_xy(
    #     upsample(pd.concat((X_test, y_test), axis=1), num_samples=150)
    # )
    return X, y, X_train, X_test, y_train, y_test


def fit_predictor(X_train, y_train, model=None):
    y_train = y_train.astype(int)

    if model is None:
        model = RandomForestClassifier(n_estimators=25, class_weight="balanced_subsample")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("imputer", KNNImputer()),
            ("classifier", model)
        ]
    )

    sample_weights = pd.Series(y_train).apply(lambda x: 2.5 if x == 0 else 1)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)

    return pipeline
