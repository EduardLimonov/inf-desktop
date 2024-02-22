from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QProgressBar

from gui.custom_pb import CustomPB
from gui.design import Ui_MainWindow
from core.core import Core
from gui.plots import plot_figures
from gui.table_utils import ComboBoxDelegate, DoubleDelegate, getItem, CustomTableView
from settings import settings

ID_COLUMN = "Идентификатор"
ALIVE_COLUMN = "Оценка выживания"
ALIVE_GT_COLUMN = "Госпит. летальность"
TYPE_COLUMN = "Тип записи"
TYPE_START_STATE = "Исходное сост."
TYPE_START_STATE_EMPTY = "Исходное сост. *"
TYPE_DESIRED_STATE = "Жел. сост."
TYPE_DELTA = "Дельта"


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    core: Core
    standardItemModel: Optional[QStandardItemModel]
    standardItemModelOutput: Optional[QStandardItemModel]

    def __init__(self, core: Core):
        super().__init__()
        self.standardItemModel = None
        self.standardItemModelOutput = None
        self.core = core
        self.setupUi(self)
        self.postSetup()
        self.initConnections()
        self.__init_graph()

        self.showMaximized()

    def postSetup(self):
        # self.gridLayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignJustify)
        self.gridLayout_8.addWidget(self.groupBox, 1, 1, 2, 1,
                                  QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft |
                                  QtCore.Qt.AlignmentFlag.AlignTop)
        self.setResultsVisible(False)
        self.progressBar.setVisible(False)
        self.progressBar_2.setVisible(False)
        self.initTables()

    def initConnections(self):
        self.pushButton_6.clicked.connect(lambda x: self.__update_example_recs(force_reload=True))
        self.horizontalSlider_2.valueChanged.connect(lambda x: self.label_8.setText("%.2f" % round(x / 100, 2)))
        self.horizontalSlider.valueChanged.connect(lambda x: self.label_5.setText("%.2f" % round(x / 100, 2)))
        self.pushButton_5.clicked.connect(lambda x: self.predict())
        self.pushButton_4.clicked.connect(lambda x: self.setResultsVisible(True))
        self.pushButton_3.clicked.connect(lambda x: self.setResultsVisible(False))
        self.pushButton.clicked.connect(lambda x: self.optimResult())
        self.button_incr.clicked.connect(lambda x: self.__changeNumRows(True))
        self.button_decr.clicked.connect(lambda x: self.__changeNumRows(False))

    def __changeNumRows(self, increase: bool):
        sign = 1 if increase else -1
        self.standardItemModel.setRowCount(self.standardItemModel.rowCount() + sign)
        if increase:
            self.__set_default_values(self.standardItemModel)

    def predict(self):
        ids, inps = self.__getInput()
        if len(inps) == 0:
            return
        preds = self.core.predict(inps)
        c_idx = self.__get_input_columns().index(ALIVE_COLUMN)
        for row_idx in range(self.standardItemModel.rowCount()):
            self.standardItemModel.setItem(row_idx, c_idx, QStandardItem(str(preds[row_idx])))

        self.tableView.resizeColumnsToContents()

    def optimResult(self, table: CustomTableView = None, pbar: QProgressBar = None):
        if table is None:
            table = self.tableView_2
        if pbar is None:
            pbar = self.progressBar

        ids, inp = self.__getInput()
        if len(inp) == 0:
            return
        target_p = self.horizontalSlider.value() / 100
        result = self.core.get_recommendations(
            inp,
            target_p=target_p,
            _pb=lambda x: CustomPB(x, pbar, "Определение рекомендаций"),
        )
        self.setResultsVisible(True)
        self.__outputResult(ids, inp, result, table=table)
        pbar.setVisible(False)

    def __outputResult(self, ids: List[str], inp: pd.DataFrame, result: pd.DataFrame,
                       table: Optional[CustomTableView] = None):
        if table is None:
            table = self.tableView_2

        table.model().setRowCount(0)
        for id_str, row_inp, row_res in zip(ids, inp.values, result.values):
            self.__addResultRows(id_str, row_inp, row_res, table.model())

        table.resizeColumnsToContents()

    def __getDfForRecord(self, id_str: str, row_inp: np.ndarray, row_res: np.ndarray) -> pd.DataFrame:
        new_df_data = [row_inp]
        if any([r is None for r in row_inp]):
            new_df_data.append(self.core.fill_empty(pd.DataFrame([row_inp], columns=self.core.get_columns())).values[0])

        new_df_data.append(row_res)

        new_df = pd.DataFrame(new_df_data, columns=self.core.get_columns())
        new_df[ALIVE_COLUMN] = self.core.predict(new_df)
        df_for_delta = self.core.system.encoder.encode_df(new_df.iloc[-2:]).astype(float)

        delta_row = df_for_delta.iloc[-1] - df_for_delta.iloc[-2]
        new_df = pd.concat((new_df, pd.DataFrame([delta_row])))

        new_df[TYPE_COLUMN] = ([TYPE_START_STATE_EMPTY] if len(new_df) == 4 else []) + \
                              [TYPE_START_STATE, TYPE_DESIRED_STATE, TYPE_DELTA]
        new_df[ID_COLUMN] = id_str
        return new_df[[ID_COLUMN, TYPE_COLUMN, ALIVE_COLUMN, *self.core.get_columns()]]

    def __addResultRows(self, id_str: str, row_inp: np.ndarray, row_res: np.ndarray, model: QStandardItemModel):
        new_df = self.__getDfForRecord(id_str, row_inp, row_res)
        for i, row in new_df.iterrows():
            model.appendRow([getItem(item) for item in row])

    def __getInput(self) -> Tuple[List[str], pd.DataFrame]:
        # TODO: double validator comma sep
        res = []
        for r in range(self.standardItemModel.rowCount()):
            row = [self.standardItemModel.data(self.standardItemModel.index(r, c))
                   for c in range(self.standardItemModel.columnCount())]
            row = [r if r != "None" else None for r in row]
            res.append(row)

        res = pd.DataFrame(res, columns=self.__get_input_columns())
        return res[ID_COLUMN].tolist(), res[self.core.get_columns()]

    def setResultsVisible(self, visible: bool):
        # for w in (self.label_2, self.tableView_2, self.pushButton_3):
        #     w.setVisible(visible)
        self.widgetTrBottom.setVisible(visible)

    def showResults(self):
        self.label_2.setVisible(True)

    def __get_input_columns(self) -> List[str]:
        return [ID_COLUMN, ALIVE_COLUMN] + self.core.get_columns()

    def __get_output_columns(self) -> List[str]:
        return [ID_COLUMN, TYPE_COLUMN, ALIVE_COLUMN] + self.core.get_columns()

    def __get_example_predict_columns(self) -> List[str]:
        return [ID_COLUMN, ALIVE_GT_COLUMN, ALIVE_COLUMN] + self.core.get_columns()

    def initTables(self):
        self.tableView_2.setEditable(False)

        input_columns = self.__get_input_columns()
        output_columns = self.__get_output_columns()
        example_columns = self.__get_example_predict_columns()
        self.standardItemModel = QStandardItemModel(0, len(input_columns))
        self.__changeNumRows(True)
        self.standardItemModelOutput = QStandardItemModel(0, len(output_columns))
        self.standardItemModelExample = QStandardItemModel(0, len(example_columns))
        self.standardItemModelExampleTr = QStandardItemModel(0, len(output_columns))

        for table, model, columns, init_delegates in zip(
                (self.tableView, self.tableView_2, self.tableView_3, self.tableView_4),
                (self.standardItemModel, self.standardItemModelOutput, self.standardItemModelExample,
                 self.standardItemModelExampleTr),
                (input_columns, output_columns, example_columns, output_columns),
                (True, False, False, False)
        ):
            self.__initTable(table, model, columns, init_delegates)

    def __init_column(self, tableView: QtWidgets.QTableView, columns):
        cat_columns = self.core.get_data_info().cat_features
        all_features = self.core.get_columns()

        for idx in range(len(columns)):
            feature_name = columns[idx]
            idx = columns.index(feature_name)
            if feature_name in cat_columns:
                options = list(self.core.system.encoder.encoders[feature_name].classes_)
                tableView.setItemDelegateForColumn(idx, ComboBoxDelegate(tableView, options))
            elif feature_name in all_features:
                min_l, max_l = self.core.get_decoded_limits(feature_name)
                bottom, top, decimals = min_l, max_l, 4
                tableView.setItemDelegateForColumn(idx, DoubleDelegate(tableView, bottom, top, decimals))

    def __set_default_values(self, model: QStandardItemModel, row_num: Optional[int] = None):
        if row_num is None:
            row_num = model.rowCount() - 1

        columns = self.__get_input_columns()
        cat_columns = self.core.get_data_info().cat_features

        for idx in range(len(columns)):
            feature_name = columns[idx]
            idx = columns.index(feature_name)
            if feature_name in cat_columns:
                options = list(self.core.system.encoder.encoders[feature_name].classes_)
                if None not in options:
                    model.setItem(row_num, idx, QStandardItem(options[0]))
            elif feature_name == ALIVE_COLUMN:
                item = model.item(row_num, idx)
                if item is None:
                    item = QStandardItem(None)
                    item.setFlags(item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

                model.setItem(row_num, idx, item)

                # item.setFlags(item.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

    def __initTable(self, tableView: CustomTableView, model: QStandardItemModel, columns: List[str],
                    initDelegates: bool = True):
        model.setHorizontalHeaderLabels(columns)
        tableView.setHeaders(columns)
        tableView.setModel(model)

        if initDelegates:
            self.__init_column(tableView, columns)
        self.__set_default_values(model)
        tableView.setCore(self.core)

        tableView.resizeColumnsToContents()

    def __init_graph(self):
        test_xy = self.core.get_test_data()
        test_df = test_xy.X.copy()
        preds = self.core.predict(test_df)
        ground_true = test_xy.y.values

        cols = list(test_df.columns)
        test_df[ALIVE_COLUMN] = preds
        test_df[ALIVE_GT_COLUMN] = ground_true
        test_df[ID_COLUMN] = test_df.index.values
        test_df = test_df[[ID_COLUMN, ALIVE_GT_COLUMN, ALIVE_COLUMN, *cols]]

        plotly_graph, plotly_bar = plot_figures(preds, ground_true, ALIVE_GT_COLUMN)
        self.webEngineView.setHtml(plotly_graph.to_html(include_plotlyjs='cdn'))
        self.webEngineView_2.setHtml(plotly_bar.to_html(include_plotlyjs='cdn'))

        for row in test_df.values:
            self.standardItemModelExample.appendRow([getItem(r) for r in row])

        self.tableView_3.resizeColumnsToContents()

        self.__update_example_recs()

    def __update_example_recs(self, force_reload: bool = False):
        test_xy = self.core.get_test_data()

        bad_X = test_xy.X[test_xy.y == 0]
        ids = bad_X.index.tolist()
        self.standardItemModelExampleTr.setRowCount(0)
        recs = self.__get_example_recommendations(bad_X, force_reload=force_reload)
        self.__outputResult(ids, inp=bad_X[self.core.get_columns()], result=recs, table=self.tableView_4)

    def __get_example_recommendations(self, df: pd.DataFrame, force_reload=False):
        if not settings.recalc_test and not force_reload:
            rec = pd.read_csv(settings.test_recom_path)

        else:
            pd.options.mode.chained_assignment = None

            self.progressBar_2.show()
            rec = self.core.get_recommendations(
                df,
                _pb=lambda x: CustomPB(x, self.progressBar_2, "Определение рекомендаций"),
            )
            self.progressBar_2.hide()

            rec.to_csv(settings.test_recom_path, index=False)

        return rec
