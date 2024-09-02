import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from gc import Ui_GarbageClassification
import time

class PredictionThread(QThread):
    prediction_complete = pyqtSignal(str)

    def __init__(self, model, file_path):
        super().__init__()
        self.model = model
        self.file_path = file_path

    def run(self):
        try:
            
            img = load_img(self.file_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 
            
            
            prediction = self.model.predict(img_array)
            class_idx = np.argmax(prediction, axis=1)[0]

            
            class_names = ["Glass", "Metal", "Paper", "Plastic"]
            result_text = class_names[class_idx]

        except Exception as e:
            result_text = f"Error: {str(e)}"

        self.prediction_complete.emit(result_text)

class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_GarbageClassification()
        self.ui.setupUi(self)

      
        try:
            self.model = load_model('/dataset/trained_model.h5')
        except Exception as e:
            print(f"Error: {e}")

        
        self.ui.pushButton.clicked.connect(self.predict_image)

        
        self.ui.progressBar.setValue(0)
        self.setAcceptDrops(True)
        self.file_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.file_path = event.mimeData().urls()[0].toLocalFile()

            pixmap = QPixmap(self.file_path).scaled(self.ui.label_3.width(), self.ui.label_3.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.label_3.setPixmap(pixmap)

    def predict_image(self):
        if self.file_path:
            self.ui.progressBar.setValue(0)
            self.ui.textBrowser.setText("Analyzing...")
            self.thread = PredictionThread(self.model, self.file_path)
            self.thread.prediction_complete.connect(self.on_prediction_complete)
            self.thread.start()
            for i in range(100):
                 self.ui.progressBar.setValue(i)
                 time.sleep(0.05)

    def on_prediction_complete(self, result_text):
        self.ui.textBrowser.setText(f"Result: {result_text}")
        self.ui.progressBar.setValue(100)
        QThread.msleep(500)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
