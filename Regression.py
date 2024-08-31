from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from Linear_Regression import LinearRegressionWindow
from PyQt5.QtGui import QIcon

class ThirdGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('Linear_Regression')
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        self.btn1 = QtWidgets.QPushButton('Linear Regression', self)
        self.btn1.setGeometry(360, 210, 200, 150)
        self.btn1.setStyleSheet('background-color:#f5cd79;color;font-size:20px')
        self.btn1.clicked.connect(self.run_linear_regression)

        self.setStyleSheet('background-color:#006266;')

        self.show()

    def run_linear_regression(self):
        self.hide()
        self.linear_regression_window = LinearRegressionWindow()
        self.linear_regression_window.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    third_gui = ThirdGUI()
    sys.exit(app.exec_())

