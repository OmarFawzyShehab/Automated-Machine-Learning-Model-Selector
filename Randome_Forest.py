import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog
from tkinter import *
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from PyQt5.QtGui import QIcon

class RandomForestApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('Random Forest Tree Application')
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        self.Text1 = QLineEdit(self)
        self.Text1.setGeometry(150, 30, 500, 50)
        self.Text1.setPlaceholderText("Enter CSV file path")

        self.Text2 = QLineEdit(self)
        self.Text2.setGeometry(200, 120, 50, 40)
        self.Text2.setText("5")  # Set default KFold value to 5

        self.btn1 = QPushButton('Browse', self)
        self.btn1.setGeometry(700, 30, 100, 50)
        self.btn1.setToolTip('Browse')
        self.btn1.clicked.connect(self.browse_csv)

        self.btn2 = QPushButton('Run', self)
        self.btn2.setGeometry(700, 400, 100, 50)
        self.btn2.clicked.connect(self.run_random_forest)
        self.btn2.setEnabled(False)

        self.name1 = QLabel('  CSV file', self)
        self.name1.setGeometry(30, 30, 100, 50)
        self.name1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.name2 = QLabel('   KFold', self)
        self.name2.setGeometry(30, 120, 100, 50)
        self.name2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.btn1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.btn2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.Text1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.Text2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.setStyleSheet('background-color:#006266;')

        self.csv_filename = None

    def browse_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        if filename:
            self.Text1.setText(filename)
            self.csv_filename = filename
            self.btn2.setEnabled(True)

    def run_random_forest(self):
        if self.csv_filename:
            try:
                kfold_value = int(self.Text2.text())
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please enter a valid integer for KFold.")
                return

            data = pd.read_csv(self.csv_filename)
            x = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            new_data = SimpleImputer(missing_values=np.nan, strategy="mean")
            new_data = new_data.fit_transform(data)
            x = new_data[:, :-1]
            y = new_data[:, -1]

            scale = StandardScaler()
            x = scale.fit_transform(x)

            rf = RandomForestClassifier(n_estimators=10)

            kfold = KFold(n_splits=kfold_value, random_state=42, shuffle=True)

            acclist = []

            for train_index, test_index in kfold.split(x):
                X_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                rf.fit(X_train, y_train)
                y_predict = rf.predict(X_test)


                acc = accuracy_score(y_test, y_predict)
                acclist.append(acc)

            acc = sum(acclist) / kfold_value

            cm = confusion_matrix(y_test, y_predict, labels=[0, 1])
            confusion = pd.DataFrame(cm, index=['actual 0', 'actual 1'], columns=['predicted 0', 'predicted 1'])

            sns.heatmap(confusion, center=True)
            plt.show()

            window = Tk()
            window.title("Result")
            window.geometry("1000x500")
            window.configure(bg='#006266')
            acc_label = Label(window, text=f"Accuracy: {acc}", height=2, font=("Arial", 20), bg='#ccae62', fg='#fff')
            acc_label.pack()
            classification = Label(window,
                                    text=f"Classification Report: {metrics.classification_report(y_test, y_predict)}",
                                    font=("Arial", 20), bg='#ccae62', fg='#fff')
            classification.pack()
            window.mainloop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myApp = RandomForestApp()
    myApp.show()
    sys.exit(app.exec_())
