import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from Classification import SecondGUI
from Regression import ThirdGUI
from PyQt5.QtGui import QIcon

class MainGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('Classification')
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        self.setStyleSheet('background-color:#006266;')

        self.btn1 = QtWidgets.QPushButton('Regression', self)
        self.btn1.setGeometry(500, 210, 150, 100)
        self.btn1.setStyleSheet('background-color:#f5cd79;color;font-size:20px')
        self.btn1.clicked.connect(self.show_third_gui)

        self.btn2 = QtWidgets.QPushButton('Classification', self)
        self.btn2.setGeometry(250, 210, 180, 100)
        self.btn2.setStyleSheet('background-color:#f5cd79;color;font-size:20px')
        self.btn2.clicked.connect(self.show_second_gui)

        self.show()

    def show_second_gui(self):
        self.hide()
        self.Classification = SecondGUI()
        self.Classification.show()

    def show_third_gui(self):
        self.hide()
        self.Regression = ThirdGUI()
        self.Regression.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_win = MainGUI()
    sys.exit(app.exec_())
