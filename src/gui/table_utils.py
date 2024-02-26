from typing import Union, Optional, List

import pandas as pd
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import QLocale
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtGui import QBrush, QColor
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QStandardItem
from PyQt5.QtWidgets import QTableView

from core.base_core import CoreInterface


class ComboBoxDelegate(QtWidgets.QItemDelegate):
    def __init__(self, owner, choices):
        self.items = choices
        super().__init__(owner)

    def paint(self, painter, option, index):
        if isinstance(self.parent(), QtWidgets.QAbstractItemView):
            self.parent().openPersistentEditor(index)
        else:
            super(ComboBoxDelegate, self).paint(painter, option, index)

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QComboBox(parent)
        editor.currentIndexChanged.connect(self.commit_editor)
        editor.addItems(self.items)
        return editor

    def commit_editor(self):
        editor = self.sender()
        self.commitData.emit(editor)

    def setEditorData(self, editor, index):
        value = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        num = self.items.index(value)
        editor.setCurrentIndex(num)

    def setModelData(self, editor, model, index):
        value = editor.currentText()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class DoubleDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, owner, bottom: float, top: float, decimals: int):
        super().__init__(owner)
        self.bottom, self.top, self.decimals = bottom, top, decimals
        self.validator = QtGui.QDoubleValidator(self.bottom, self.top, self.decimals)
        self.validator.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        # self.validator.setParent(editor)
        if isinstance(editor, QtWidgets.QLineEdit):
            editor.setValidator(self.validator)
        return editor

    def checkData(self, data: str):
        return self.validator.validate(data, 0) == QtGui.QValidator.State.Acceptable


def getItem(item: Union[float, str, None], highlightColor: str = "#d4f5ff") -> QStandardItem:
    if pd.isnull(item):
        res = QStandardItem("")
        res.setBackground(QBrush(QColor(highlightColor)))
    else:
        if type(item) == float:
            item = round(item, 4)
        res = QStandardItem(str(item))

    res.setFlags(res.flags() ^ QtCore.Qt.ItemFlag.ItemIsEditable)

    return res


class CustomTableView(QTableView):
    isEditable: bool
    core: Optional[CoreInterface]
    headers: Optional[List[str]]

    def __init__(self, parent, isEditable: bool = True):
        self.isEditable = isEditable
        self.core = None
        self.headers = None
        super().__init__(parent)

    def setHeaders(self, headers: List[str]):
        self.headers = headers

    def setEditable(self, editable: bool):
        self.isEditable = editable

    def setCore(self, core: CoreInterface):
        self.core = core

    def keyPressEvent(self, event: QKeyEvent):
        super().keyPressEvent(event)

        selectedIndexes = self.selectedIndexes()

        if len(selectedIndexes):
            if event.matches(QKeySequence.StandardKey.Copy):
                text = ""
                top_row = min(selectedIndexes, key=lambda x: x.row()).row()
                bottom_row = max(selectedIndexes, key=lambda x: x.row()).row()
                left_column = min(selectedIndexes, key=lambda x: x.column()).column()
                right_column = max(selectedIndexes, key=lambda x: x.column()).column()

                for r in range(top_row, bottom_row + 1):
                    rowContents = [str(self.model().index(r, c).data()) for c in range(left_column, right_column + 1)]
                    text += "\t".join(rowContents) + "\n"

                QApplication.clipboard().setText(text)

            elif event.matches(QKeySequence.StandardKey.Paste) and self.isEditable:
                text = QApplication.clipboard().text()
                rowContents = text.split("\n")
                idx = selectedIndexes[0]

                for i in range(len(rowContents)):
                    columnContents = rowContents[i].split("\t")
                    for j in range(len(columnContents)):
                        if self.validate(idx.column() + j, columnContents[j]):
                            self.model().setData(self.model().index(idx.row() + i, idx.column() + j), columnContents[j])

    def validate(self, column: int, content: str) -> bool:
        if self.core is None or self.headers is None:
            return True

        col_name = self.headers[column]
        if col_name not in self.core.get_columns():
            return True
        return self.core.check_correct_feature(col_name, content)
