# Test PyQt5 GUI

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore


class mainUI(QMainWindow):

    def __init__(self):
        
        super().__init__()
        self.title = "Brain Tumor Detection"
        self.initUI()

        return

    def initUI(self):

        # Basic settings of window
        self.showMaximized()
        self.setWindowFlags(QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon("icons\\brain.png"))

        # Set status bar
        self.statusBar().showMessage("Ready")

        # Set menu - File
        impoAct = QAction("&Import", self)
        impoAct.setShortcut("Ctrl+O")

        exitAct = QAction("&Exit", self)
        exitAct.setShortcut("Ctrl+Q")
        exitAct.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(impoAct)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAct)

        self.show()

        return


def main():
    
    app = QApplication(sys.argv)
    ui = mainUI()
    sys.exit(app.exec_())

    return


if __name__ == "__main__":
    main()