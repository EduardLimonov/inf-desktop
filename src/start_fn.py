import sys
import os

from PyQt5 import QtCore, QtWidgets

from core.core_factory import CoreFactory
from gui.AppWindow import MainWindow


def restart_fn(init_manager: bool = False):
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication(sys.argv)
    with open("resource/ui/styles.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)

    core = CoreFactory.create_core(create_core_manager=init_manager)

    window = MainWindow(core, restart_fn)
    window.show()
    app.exec()

    core.shutdown_server_if_started()
