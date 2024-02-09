import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode, GridUpdateMode

from gui.plots import plot_accuracy
from pages.utils.service import init_core, get_recommendation, fill_highlight

st.set_page_config(page_title="AD аналитика", layout="wide", page_icon="📊")

st.title("Анализ работы системы")
core = init_core()

test_xy = core.get_test_data()
test_df = test_xy.X.copy()
cols = list(test_df.columns)
test_df["Оценка выживания"] = core.predict(test_df)
test_df["Выжил"] = test_xy.y.values
test_df["Идентификатор"] = test_df.index.values
test_df = test_df[["Идентификатор", "Выжил", "Оценка выживания", *cols]]

with st.expander("Точность прогноза выживания"):
    st.subheader("Порог выживания")
    st.plotly_chart(plot_accuracy(test_df["Оценка выживания"], test_df["Выжил"]), use_container_width=True)

    st.subheader("Тестовые данные")
    AgGrid(test_df, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, update_mode=GridUpdateMode.NO_UPDATE)

with st.expander("Лечение", expanded=True):
    st.subheader("Лечение для умерших пациентов из тестовой выборки")
    bad_X = test_xy.X[test_xy.y == 0]  # .iloc[:1]
    with st.spinner("Вычисление рекомендаций"):
        res = get_recommendation(core, bad_X)
        res = fill_highlight(res.reset_index(), core)

    st.dataframe(res, use_container_width=True, height=1000, hide_index=True)
    # AgGrid(res, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS, update_mode=GridUpdateMode.NO_UPDATE)

