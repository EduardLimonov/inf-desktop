from typing import List, Tuple

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView, QWidget, QLabel, \
    QPushButton, QErrorMessage

from core.base_core import CoreInterface
from gui.core_callback_secure import core_callback_secure
from settings.network import network_settings


class CoreMgmtWindow(QMainWindow):

    core: CoreInterface
    tableModel: List[Tuple[str, str]]

    def __init__(self, parent, core: CoreInterface):
        super().__init__(parent)  # parent)
        self.setWindowTitle("Управление ядрами")
        self.tableModel = []
        self.core = core

        self.setupUI()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setMinimumSize(700, 400)
        self.setEnabled(True)

    @core_callback_secure
    def setupUI(self):
        self.tableModel.clear()
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)

        self.table = QTableWidget(self.centralWidget)
        vlayout = QVBoxLayout(self.centralWidget)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        vlayout.addWidget(QLabel("Список подключений (локальное невозможно отредактировать)", self.centralWidget))
        vlayout.addWidget(self.table)
        self.centralWidget.setLayout(vlayout)

        url_known = self.core.get_urls_known()
        self.table.setRowCount(len(url_known) + 1)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Имя", "Адрес", "Действие"])

        for idx, (_name, _url) in enumerate(url_known.items()):
            self.__add_item(idx, _name, _url)

        self.__add_item(len(url_known), "", "", row_for_new=True)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(False)

        self.table.cellChanged.connect(self.cellChanged)

    @core_callback_secure
    def cellChanged(self, row, column):
        if row >= len(self.tableModel):
            return

        name_old, url_old = self.tableModel[row]
        name_new, url_new = self.table.item(row, 0).text(), self.table.item(row, 1).text()

        may_change = True
        if name_old != name_new and not self.__check_connection_name(url_new):
            self.__show_error_msg(f"Недопустимое имя: {name_old}")
            may_change = False
        elif url_old != url_new and not self.__check_connection_url(url_new):
            if not self.__check_connection_url(url_old):
                self.__show_error_msg(f"Не удается подключиться к хосту: {url_new} (введенный) и к хосту {url_old} "
                                      f"(предыдущий). Будет применен новый хост {url_new}")
            else:
                self.__show_error_msg(f"Не удается подключиться к хосту: {url_new}")
                may_change = False

        if may_change:
            self.core.remove_url(name_old)
            self.core.add_url(url_new, name_new)
        else:
            self.table.setItem(row, column, QTableWidgetItem(name_old if column == 0 else url_old))

    @core_callback_secure
    def __check_connection_name(self, name: str) -> bool:
        return name not in self.core.get_urls_known()

    @core_callback_secure
    def __check_connection_url(self, url: str) -> bool:
        return self.core.check_connection(url)

    def __add_item(self, row: int, _name: str, _url: str, row_for_new: bool = False):
        if row_for_new:
            _name, _url = "", ""
            btn = QPushButton("Сохранить", self.table)
            btn.clicked.connect(lambda e, idx=row: self.__add_connection(row))
        else:
            self.tableModel.append((_name, _url))
            btn = QPushButton("Удалить", self.table)
            btn.clicked.connect(lambda e, idx=row: self.__remove_connection(row))

        it_name, it_url = QTableWidgetItem(_name), QTableWidgetItem(_url)
        if _name == network_settings.DEFAULT_LOCAL_NAME:
            it_name.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            it_url.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            btn.setEnabled(False)

        self.table.setItem(row, 0, it_name)
        self.table.setItem(row, 1, it_url)
        self.table.setCellWidget(row, 2, btn)

    def __show_error_msg(self, error_text: str):
        msg = QErrorMessage(self)
        msg.setWindowTitle("Ошибка")
        msg.showMessage(error_text)

    @core_callback_secure
    def __add_connection(self, row: int):
        url_name, url = self.table.item(row, 0).text(), self.table.item(row, 1).text()
        if not url.endswith("/"):
            url += "/"

        if not self.__check_connection_name(url_name):
            self.__show_error_msg(f"Подключение с именем {url_name} уже существует")

        elif not self.__check_connection_url(url):
            self.__show_error_msg(f"Не удается подключиться к хосту '{url}'")

        else:
            self.core.add_url(url, url_name)
            self.centralWidget.hide()
            self.setupUI()

    @core_callback_secure
    def __remove_connection(self, row: int):
        url_name = self.table.item(row, 0).text()
        result = self.core.remove_url(url_name)
        if not result:
            self.__show_error_msg(f"Невозможно удалить подключение '{url_name}'")
        else:
            self.centralWidget.hide()
            self.setupUI()

    def focusOutEvent(self, event):
        self.setFocus()
        self.activateWindow()
        # self.raise_()
        self.show()

    def closeEvent(self, event):
        self.parent().setEnabled(True)
        self.parent().update_connections_items()
