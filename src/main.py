import sys
import os
import PyQt5
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow


def main():
    plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    os.environ["QT_PLUGIN_PATH"] = plugin_path

    app = QApplication(sys.argv)
    app.setApplicationName("Video Recorder")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
