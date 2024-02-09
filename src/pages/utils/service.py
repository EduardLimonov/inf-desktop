from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from core.core import Core
from gui.custom_pb import CustomPB
from settings import settings


PATIENT_INPUT_KEY = "PATIENT_INPUT_KEY"
PROGRESS_BAR = lambda x: CustomPB(x, "Поиск лечения")


@st.cache_resource
def init_core():
    return Core()


@st.cache_data(show_spinner=False)
def get_recommendation(_core: Core, df: pd.DataFrame) -> pd.DataFrame:
    if not settings.recalc_test:
        ans = pd.read_csv(settings.test_recom_path)

    else:
        pd.options.mode.chained_assignment = None

        rec = _core.get_recommendations(df, method="Powell", _pb=PROGRESS_BAR)

        ans = unite_recommendations(df, rec, _core)
        ans.to_csv(settings.test_recom_path, index=False)

    return ans


def unite_recommendations(df: pd.DataFrame, res: pd.DataFrame, _core: Core) -> pd.DataFrame:
    delta_df = _core.calc_delta(df, res)

    result_df = []
    for i in range(len(df)):
        pat = df.iloc[i]
        pat["Оценка выживания"] = _core.predict(pd.DataFrame([pat]))[0]
        pat["Метка"] = "Исходное состояние"

        optimized = res.iloc[i]
        optimized["Оценка выживания"] = _core.predict(pd.DataFrame([optimized]))[0]
        optimized["Метка"] = "Желаемое состояние"

        delta = delta_df.iloc[i]
        delta["Метка"] = "Дельта"

        pat_result = pd.DataFrame([pat, optimized, delta])
        pat_result["Идентификатор"] = pat.name
        pat_result = pat_result[["Идентификатор", "Метка", "Оценка выживания", *list(df.columns)]]

        result_df.append(pat_result)

    result_df = pd.concat(result_df)
    result_df = result_df.applymap(lambda x: x if type(x) != float else round(x, 3), na_action="ignore")

    return result_df


def get_patient_input_table(core: Core):
    di = core.get_data_info()

    all_c = ["Идентификатор",  # "Оценка выживания",
             *list(di.feature_limits.keys())]

    # if PATIENT_INPUT_KEY in st.session_state:
    #     df = st.session_state[PATIENT_INPUT_KEY]
    #     # st.session_state.pop(PATIENT_INPUT_KEY)
    # else:
    #     df = st.session_state[PATIENT_INPUT_KEY] = pd.DataFrame(columns=all_c)

    res = st.data_editor(
        pd.DataFrame(columns=all_c),
        column_config={
            f: __get_column_config(f, core)
            for f in all_c
        },
        num_rows="dynamic",
        hide_index=True,
        # key=PATIENT_INPUT_KEY
        # on_change=lambda: None if PATIENT_INPUT_KEY not in st.session_state else st.session_state.pop(PATIENT_INPUT_KEY)
    )
    # st.session_state[PATIENT_INPUT_KEY] = res
    # if PATIENT_INPUT_KEY in st.session_state:
    #     st.session_state.pop(PATIENT_INPUT_KEY)
    return res


def __get_column_config(col_name: str, core: Core):
    if col_name == "Оценка выживания":
        return st.column_config.NumberColumn(disabled=True)
    elif col_name == "Идентификатор":
        return st.column_config.TextColumn(required=True, max_chars=50)
    elif col_name in core.system.encoder.encoders:
        # categorical feature
        options = list(core.system.encoder.encoders[col_name].classes_)
        return st.column_config.SelectboxColumn(default=options[0], options=options, required=True)

    else:
        min_l, max_l = core.get_decoded_limits(col_name)
        return st.column_config.NumberColumn(min_value=min_l, max_value=max_l)


def __extract_patients(df: pd.DataFrame, core: Core) -> pd.DataFrame:
    df = df.set_index("Идентификатор", drop=True)
    return df[core.system.data_info.num_to_column_mapper]


def predict_patients(df: pd.DataFrame, core: Core):
    if len(df) == 0:
        st.warning("Записи о пациентах отсутствуют")
        return
    patients = __extract_patients(df, core)
    predictions = core.predict(patients)
    df["Оценка выживания"] = predictions
    df = df[["Идентификатор", "Оценка выживания", *patients.columns]]
    return df


def optimize_treatment(df: pd.DataFrame, core: Core):
    if len(df) == 0:
        st.warning("Записи о пациентах отсутствуют")
        return

    patients = __extract_patients(df, core)
    rec = core.get_recommendations(patients, _pb=PROGRESS_BAR)
    if "index" in rec:
        rec = rec.drop("index", axis=1)

    return unite_recommendations(patients, rec, core)


def fill_highlight(df: Optional[pd.DataFrame], core: Core, color="#d1f4ff") -> Optional:
    if df is None or len(df) == 0:
        return df

    if "index" in df.columns:
        df = df.drop("index", axis=1)

    filled_df = core.fill_empty_values(df).reset_index(drop=False).style
    filled_df = filled_df.apply(
        lambda s: np.where(df.loc[s.name].isna() & s.notna(), f"color:black;background-color:{color};", ";"),
        axis=1,
    )
    return filled_df.format(precision=3, thousands=" ", decimal=",", na_rep="")
