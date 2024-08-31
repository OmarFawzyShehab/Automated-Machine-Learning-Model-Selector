import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, median_absolute_error,mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
from PyQt5.QtGui import QIcon

class LinearRegressionWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle("Linear_Regression")
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        self.Text1 = QLineEdit(self)
        self.Text1.setGeometry(150, 30, 500, 50)

        self.btn1 = QPushButton("Browse", self)
        self.btn1.setGeometry(700, 30, 100, 50)
        self.btn1.setToolTip("Browse")
        self.btn1.clicked.connect(self.browse_file)

        self.TestSize = QLineEdit(self)
        self.TestSize.setGeometry(300, 100, 30, 50)

        self.btn2 = QPushButton("Run", self)
        self.btn2.setGeometry(700, 400, 100, 50)
        self.btn2.clicked.connect(self.run)

        self.name1 = QLabel("  CSV file", self)
        self.name1.setGeometry(30, 30, 100, 50)
        self.name1.setStyleSheet("background-color:#f5cd79;color;font-size:20px")

        self.name2 = QLabel("  Test-data size(%):", self)
        self.name2.setGeometry(30, 100, 180, 50)
        self.name2.setStyleSheet("background-color:#f5cd79;color;font-size:20px")



        self.btn1.setStyleSheet("background-color:#f5cd79;color;font-size:20px")
        self.btn2.setStyleSheet("background-color:#f5cd79;color;font-size:20px")
        self.Text1.setStyleSheet("background-color:#f5cd79;color;font-size:20px")
        self.TestSize.setStyleSheet("background-color:#f5cd79;color;font-size:20px")
        self.setStyleSheet("background-color:#006266;")

        self.show()
        self.btn1.show()
        self.Text1.show()
        self.TestSize.show()

    def browse_file(self):
        # file_dialog = QFileDialog()
        # file_path, _ = file_dialog.getOpenFileName(self, "Select CSV File")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                  options=options)
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))
        if filename:
            self.Text1.setText(filename)

    def run(self):
        file_name = self.Text1.text()
        if not file_name:
            return

        data = pd.read_csv(file_name)

        new_data = SimpleImputer(missing_values=np.nan, strategy="mean")
        new_data = new_data.fit_transform(data)
        x = new_data[:, :-1]
        y = new_data[:, -1]

        scale = StandardScaler()
        x = scale.fit_transform(x)

        observation_count = 100
        x_var = np.linspace(start=0, stop=10, num=observation_count)
        np.random.seed(22)
        y_var = x_var + np.random.normal(size=observation_count, loc=1, scale=2)
        sns.scatterplot(x=x_var, y=y_var)
        plt.show()

        test_size = float(self.TestSize.text()) if self.TestSize.text() else 30
        if test_size > 40:
            print("Test size should not exceed 40%")
            return

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size/100, random_state=0
        )

        linear = LinearRegression()
        linear = linear.fit(x_train, y_train)
        y_predict = linear.predict(x_test)

        error1 = median_absolute_error(y_test, y_predict)
        error2 = mean_squared_error(y_test, y_predict)
        error3 = mean_absolute_error(y_test,y_predict)

        window = Tk()
        window.title("Result")
        window.geometry("600x500")
        window.configure(bg="#006266")
        acc = Label(
            window, text=f"Median absolute error: {error1}", font=("Arial", 20), bg="#ccae62", fg="#fff"
        )
        acc.pack()
        acc1 = Label(
            window, text=f"Mean squared error: {error2}", font=("Arial", 20), bg="#ccae62", fg="#fff"
        )
        acc1.pack()
        acc2 = Label(
            window, text=f"Mean absolute error: {error3}", font=("Arial", 20), bg="#ccae62", fg="#fff"
        )
        acc2.pack()
        window.mainloop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    linear_regression_win = LinearRegressionWindow()
    sys.exit(app.exec_())


