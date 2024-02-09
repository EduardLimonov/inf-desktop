import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode

from pages.utils.service import init_core, get_patient_input_table, predict_patients, optimize_treatment, fill_highlight

st.set_page_config(page_title="AD рекомендации", layout="wide", page_icon="❤️")


core = init_core()


st.header("Рекомендации по лечению")
# with st.form('editor'):
st.subheader("Ввод данных")
pat_table = get_patient_input_table(core)
# st.form_submit_button("Подтвердить ввод")

c1, c2 = st.columns([1, 6])

with c1:
    clicked = st.button("Прогноз выживания")  # on_click=lambda: predict_patients(pat_table, core))

if clicked:
    st.subheader("Результаты")

    with st.expander("Прогноз выживания", expanded=True):
        res = predict_patients(pat_table, core)
        res = fill_highlight(res, core)
        st.dataframe(res, hide_index=True, use_container_width=True)

with c2:
    clicked = st.button("Поиск лечения")

if clicked:
    st.subheader("Результаты")

    with st.expander("Рекомендации по лечению", expanded=True):
        res = optimize_treatment(pat_table, core).reset_index(drop=False)
        res = fill_highlight(res, core)
        st.dataframe(res, hide_index=True, use_container_width=True)



