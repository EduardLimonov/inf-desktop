from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow

from core.core_manager import CoreManager


class CoreMgmtWindow(QMainWindow):

    core: CoreManager

    def __init__(self, parent, core: CoreManager):
        super().__init__(parent)
        self.setWindowTitle("Управление ядрами")
        self.core = core

        self.setupUI()
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        # self.resize(200, 215)

    def setupUI(self):
        pass

    def focusOutEvent(self, event):
        self.setFocus()
        self.activateWindow()
        # self.raise_()
        self.show()

    def closeEvent(self, event):
        self.parent().setEnabled(True)




