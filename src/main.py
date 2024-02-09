import sys

from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets

from gui.AppWindow import MainWindow
from core.core import Core


def main():
    from qt_material import apply_stylesheet
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    with open("resource/ui/styles.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    # apply_stylesheet(app, theme='light_lightgreen_500.xml')#, invert_secondary=True)
    window = MainWindow(Core())  # Создаём объект класса ExampleApp

    window.show()  # Показываем окно
    app.exec()  # и запускаем приложение


if __name__ == "__main__":
    main()
