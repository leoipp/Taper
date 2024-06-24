from PyQt5 import QtCore
import pandas as pd


class ThreadReader(QtCore.QThread):
    finished = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, filepath, sheetname):
        super().__init__()
        self.filePath = filepath
        self.sheetName = sheetname
        self.result = None

    def run(self):
        self.result = pd.read_excel(self.filePath, sheet_name=self.sheetName)
        self.finished.emit(self.result)
