import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QFileDialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
from PyQt5.QtGui import QIcon

class KNNApplication(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(600, 200, 900, 600)
        self.setWindowTitle('KNN')
        self.setWindowIcon(QIcon('C:\\Users\\Hp\\Downloads\\2857360.png'))

        self.Text1 = QLineEdit(self)
        self.Text1.setGeometry(150, 30, 500, 50)

        self.btn1 = QPushButton('Browse', self)
        self.btn1.setGeometry(700, 30, 100, 50)
        self.btn1.setToolTip('Browse')
        self.btn1.clicked.connect(self.browse_file)

        self.btn2 = QPushButton('Run', self)
        self.btn2.setGeometry(700, 400, 100, 50)
        self.btn2.clicked.connect(self.run)

        self.name1 = QLabel('  CSV file', self)
        self.name1.setGeometry(30, 30, 100, 50)
        self.name1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.btn1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.btn2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.Text1.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.Text2 = QLineEdit(self)
        self.Text2.setGeometry(300, 120, 50, 40)
        self.Text2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.Text2.setText('5')  # Default value for K

        self.name2 = QLabel('   K', self)
        self.name2.setGeometry(30, 120, 100, 40)
        self.name2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.name2 = QLabel('   Test_data size(%): ', self)
        self.name2.setGeometry(30, 200, 190, 40)
        self.name2.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')

        self.Text3 = QLineEdit(self)
        self.Text3.setGeometry(300, 200, 50, 40)
        self.Text3.setStyleSheet('background-color:#f5cd79;color ; font-size:20px')
        self.Text3.setText('30')  # Default value for test data size

        self.setStyleSheet('background-color:#006266;')

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
            return  # No file selected

        test_data_size = float(self.Text3.text())
        if test_data_size > 40:
            print("Warning: Your test data size is too high.")
            return

        data = pd.read_csv(file_name)
        x = data.drop(columns=['target'])
        y = data['target'].values

        new_data = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_data = new_data.fit_transform(data)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_data_size / 100, random_state=0, stratify=None
        )

        new_x_train = StandardScaler()
        x_train = new_x_train.fit_transform(x_train)
        new_x_test = StandardScaler()
        x_test = new_x_test.fit_transform(x_test)

        knn = KNeighborsClassifier(n_neighbors=int(self.Text2.text()))
        knn = knn.fit(x_train, y_train)
        y_predict = knn.predict(x_test)

        ac = accuracy_score(y_test, y_predict)
        print(ac)

        cr = classification_report(y_test, y_predict)
        print(cr)

        cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
        confusion = pd.DataFrame(cm, index=['actual 0', 'actual 1'], columns=['predicted 0', 'predicted 1'])

        sns.heatmap(confusion, center=True)
        plt.show()

        error_rate = []
        for i in range(1, 40):
            knn = KNeighborsClassifier(n_neighbors=i) # n_neighbors = k
            knn.fit(x_train, y_train)
            pred_i = knn.predict(x_test)
            error_rate.append(np.mean(pred_i != y_test))

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',
                 markersize=10)
        plt.title('ErrorRate vs. K-value')
        plt.xlabel('K')
        plt.ylabel('ErrorRate')
        plt.show()

        window = Tk()
        window.title("Result")
        window.geometry("1000x500")
        window.configure(bg='#006266')
        result = Label(window, text=f"Accuracy: {metrics.accuracy_score(y_test, y_predict)}", font=("Arial", 20),
                       bg='#ccae62', fg='#fff')
        result.pack()
        classification = Label(window,
                               text=f"classification Report: {metrics.classification_report(y_test, y_predict)}",
                               font=("Arial", 20), bg='#ccae62', fg='#fff')
        classification.pack()
        window.mainloop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    knn_app = KNNApplication()
    knn_app.show()
    sys.exit(app.exec_())
