from PyQt5 import QtWidgets
from KNN import KNNApplication
from SVM import SVMApp
from Decision_Tree import DecisionTreeApp
from Randome_Forest import RandomForestApp
from PyQt5.QtGui import QIcon


class SecondGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('Classification')
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        btn1 = QtWidgets.QPushButton('Decision tree', self)
        btn1.setGeometry(400, 210, 150, 100)
        btn1.setStyleSheet('background-color:#f5cd79;color;font-size:20px')

        btn2 = QtWidgets.QPushButton('SVM', self)
        btn2.setGeometry(250, 210, 90, 100)
        btn2.setStyleSheet('background-color:#f5cd79;color;font-size:20px')

        btn3 = QtWidgets.QPushButton('KNN', self)
        btn3.setGeometry(100, 210, 90, 100)
        btn3.setStyleSheet('background-color:#f5cd79;color;font-size:20px')

        btn4 = QtWidgets.QPushButton('Random Forest', self)
        btn4.setGeometry(620, 210, 200, 100)
        btn4.setStyleSheet('background-color:#f5cd79;color;font-size:20px')

        self.setStyleSheet('background-color:#006266;')

        btn1.clicked.connect(self.run_decision_tree)
        btn2.clicked.connect(self.run_svm)
        btn3.clicked.connect(self.run_knn)
        btn4.clicked.connect(self.run_random_forest)

    def run_decision_tree(self):
        self.decision_tree_window = DecisionTreeApp()
        self.decision_tree_window.show()

    def run_svm(self):
        self.svm_window = SVMApp()
        self.svm_window.show()

    def run_knn(self):
        self.knn_window = KNNApplication()
        self.knn_window.show()

    def run_random_forest(self):
        self.random_forest_window = RandomForestApp()
        self.random_forest_window.show()
