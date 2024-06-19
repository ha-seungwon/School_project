from image_viewer import *
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
import os
import sys

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from PyQt5 import QtWidgets

class My_Application(QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.checkPath)

    def checkPath(self):
        image_path = self.ui.lineEdit.text()
        if os.path.isfile(image_path):
            model = VGG16()

            image = load_img(image_path, target_size=(224, 224))
            image_array = img_to_array(image)
            image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
            image_array = preprocess_input(image_array)

            predictions = model.predict(image_array)
            labels = decode_predictions(predictions, top=3)  # Get top 3 predictions

            result_text = 'Top 3 Predictions:'
            for i, label_info in enumerate(labels[0]):
                result_text += f'\n{i + 1}. {label_info[1]}: {label_info[2]*100:.2f}%'


            # Display the result_text in the QLabel
            self.ui.result_label.setText(result_text)
            # Display the image in QGraphicsView
            scene = QtWidgets.QGraphicsScene(self)
            pixmap = QPixmap(image_path)
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            self.ui.graphicsView.setScene(scene)


# The rest of your code remains unchanged

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    class_instance = My_Application()
    class_instance.show()
    sys.exit(app.exec_())
