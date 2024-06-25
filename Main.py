import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QVBoxLayout, \
    QComboBox, QPushButton, QCheckBox, QGridLayout, QWidget, QLabel, QMessageBox
from PyQt5 import uic
import os
from Threads import ThreadReader
from Recursos import output_df, plot_layout, plot_layout_hist
from Recursos import Garay
import numpy as np


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class Mainui(QMainWindow):
    def __init__(self):
        super(Mainui, self).__init__()
        uic.loadUi(resource_path("GUI/MainGui.ui"), self)

        self.b1.clicked.connect(lambda: self.openFileDialog(1))
        self.b3.clicked.connect(lambda: self.openFileDialog(2))
        self.b5.clicked.connect(lambda: self.openFileDialog(3))
        self.b2.clicked.connect(lambda: self.on_click(1))
        self.b4.clicked.connect(lambda: self.on_click(2))
        self.b6.clicked.connect(lambda: self.on_click(3))
        self.bexp.clicked.connect(lambda: self.saveFileDialog(self.qd_ajustes))
        self.thread = None
        self.worker = None

        self.horizontalSlider.valueChanged.connect(self.update_label)
        self.horizontalSlider_2.valueChanged.connect(self.update_label_2)

        # Ensure alvo is initialized
        if not hasattr(self, 'alvo'):
            self.alvo = QWidget(self)
            self.setCentralWidget(self.alvo)

        # Set a layout for alvo if not already set
        if not self.alvo.layout():
            self.alvo_layout = QGridLayout(self.alvo)
            self.alvo.setLayout(self.alvo_layout)
        else:
            self.alvo_layout = self.alvo.layout()

        self.checkboxes_1 = []

        self.path, self.sheet = None, None
        self.qd_ajustes = None
        self.bexp.setDisabled(True)

    def update_label(self, value):
        self.label_6.setText(f'{value}')

    def update_label_2(self, value):
        self.label_7.setText(f'{value}')

    def openFileDialog(self, botao: int):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Abrir arquivo de cubagem", "", "Arquivos excel (*.xlsx);;All Files (*)", options=options)
        if filePath:
            self.loadExcel(filePath, botao)

    def loadExcel(self, filePath: str, botao: int):
        xls = pd.ExcelFile(filePath)
        sheets = xls.sheet_names

        sheetDialog = SheetSelectionDialog(sheets)
        if sheetDialog.exec_():
            selectedSheet = sheetDialog.getSelectedSheet()
            self.sheet = selectedSheet
            self.path = filePath
            self.startWorker(filePath, selectedSheet, botao)

    def startWorker(self, filePath, sheetName, botao: int):
        self.worker = ThreadReader(filePath, sheetName)
        self.worker.finished.connect(lambda: self.onWorkerFinished(botao))
        self.worker.start()

    def onWorkerFinished(self, botao):
        self.df = self.worker.result
        self.showColumnSelection(botao)

    def showColumnSelection(self, botao):
        if botao == 1:
            label = QLabel("Variáveis da base de dados:")
            self.alvo_layout.addWidget(label, 0, 0, 1, 2)
            for i, col in enumerate(self.df.columns):
                checkbox = QCheckBox(col)
                checkbox.setVisible(True)
                self.checkboxes_1.append(checkbox)
                self.alvo_layout.addWidget(checkbox, (i // 3) + 1, i % 3)
        if botao == 2:
            self.comboBox.addItems(self.df.columns)
            self.comboBox_2.addItems(self.df.columns)
            self.comboBox_3.addItems(self.df.columns)
            self.comboBox_4.addItems(self.df.columns)


        if botao == 3:
            self.comboBox_5.addItems(self.df.columns)
            self.comboBox_6.addItems(self.df.columns)
            self.comboBox_7.addItems(self.df.columns)

    def on_click(self, botao):
        if botao == 1:
            checkeds = []
            n_checked = []
            for i, checkbox in enumerate(self.checkboxes_1):
                if checkbox.isChecked():
                    checkeds.append(checkbox.text())
                else:
                    n_checked.append(checkbox.text())
            self.melted = pd.melt(self.df, id_vars=n_checked, value_vars=checkeds)
            self.melted['seccao'] = self.melted['variable'].str[1:]
            return self.saveFileDialog(self.melted)

        if botao == 2:
            taper = Garay(self.df, self.comboBox_2.currentText(), self.comboBox.currentText(),
                          self.comboBox_3.currentText(), self.comboBox_4.currentText())
            self.horizontalSlider.setMaximum(int(taper.dap.max()))
            self.horizontalSlider_2.setMaximum(int(taper.ht.max()))

            try:
                initial_params = taper.find_initial_parameters()
                optimized_params = taper.fit_model()
                d_est = taper.predict()

                # Calculate residuals
                residuals = taper.d - d_est

                # Calculate sum of squared residuals (SQRES)
                sqres = np.sum(residuals ** 2)

                # Calculate R^2
                ss_tot = np.sum((taper.d - np.mean(taper.d)) ** 2)
                r2 = 1 - (sqres / ss_tot)

                # Calculate RQEM
                rqem = np.sqrt(np.mean(residuals ** 2))

                bias = np.mean(residuals)

                self.label_19.setText(f"{optimized_params[0]:.8f}")
                self.label_20.setText(f"{optimized_params[1]:.8f}")
                self.label_21.setText(f"{optimized_params[2]:.8f}")
                self.label_22.setText(f"{optimized_params[3]:.8f}")

                self.label_30.setText(f"{sqres:.4f}")
                self.label_27.setText(f"{r2:.4f}")
                self.label_28.setText(f"{rqem:.4f}")
                self.label_31.setText(f"{bias:.4f}")

                plot_layout(self.plotLayout, taper.d, d_est, self.comboBox_4.currentText(), 'd estimado')

                self.qd_ajustes = pd.DataFrame({"Stats": ['sqres', 'r2', 'rqem', 'bias'],
                                                "Vals": [sqres, r2, rqem, bias],
                                                "params": ['b0', 'b1', 'b2', 'b3'],
                                                "params_vals": optimized_params})
                self.bexp.setDisabled(False)

            except Exception as e:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle('Error de parâmetro')
                msg.setText(f'Erro de ajuste {e}')
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()

        if botao == 3:
            taper = Garay(self.df, id=self.comboBox_6.currentText(), dap=self.comboBox_5.currentText(), ht=self.comboBox_7.currentText())
            _d_chapeu_, _h_, ht_, dap_, dif, as_, asmed, vol, id = Garay(self.df).aplicacao_real(taper.id, taper.dap, taper.ht,
                                                                                                 float(self.lineEdit_5.text()), float(self.lineEdit_6.text()), float(self.lineEdit_7.text()),
                                                                                                 [float(self.lineEdit_8.text()), float(self.lineEdit_9.text()),
                                                                                                  float(self.lineEdit_10.text()), float(self.lineEdit_11.text())])
            df = pd.DataFrame(
                {'arv': id, 'd_est': _d_chapeu_, 'h_est': _h_, 'ht': ht_, 'dap': dap_, 'dif': dif, 'as': as_,
                 'asmed': asmed, 'vol': vol})

            grouped = df.groupby('arv').sum().reset_index()
            bins = pd.interval_range(start=0, end=5, freq=0.25, closed='right')
            grouped['classe'] = pd.cut(grouped['vol'], bins, include_lowest=True)
            plot_layout_hist(self.plotLayout, grouped.classe, grouped.vol, 'bins', 'vol')

            return self.saveFileDialog(df)




    def saveFileDialog(self, df: pd.DataFrame):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getSaveFileName(self, "Salvar Arquivo", "",
                                                  "Arquivos excel (*.xlsx);;CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        if filePath:
            if filePath.endswith('.xlsx'):
                output_df(filePath, df)
            elif filePath.endswith('.csv'):
                output_df(filePath, df)
            else:
                output_df(filePath, df)


class SheetSelectionDialog(QDialog):
    def __init__(self, sheets):
        super().__init__()
        self.sheets = sheets
        self.selectedSheet = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Selecione a planilha')
        self.setGeometry(120, 120, 300, 150)

        layout = QVBoxLayout()

        self.comboBox = QComboBox(self)
        self.comboBox.addItems(self.sheets)
        layout.addWidget(self.comboBox)

        selectButton = QPushButton('+', self)
        selectButton.clicked.connect(self.selectSheet)
        layout.addWidget(selectButton)

        self.setLayout(layout)

    def selectSheet(self):
        self.selectedSheet = self.comboBox.currentText()
        self.accept()

    def getSelectedSheet(self):
        return self.selectedSheet


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = Mainui()
    viewer.show()
    sys.exit(app.exec_())
