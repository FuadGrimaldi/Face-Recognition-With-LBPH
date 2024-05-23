import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import dlib
import face_recognition
from PIL import Image
import os

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.pushButton.clicked.connect(self.fungsi)
        self.pushButton_2.clicked.connect(self.video)
        self.pushButton_3.clicked.connect(self.train_face)
        self.lineEdit = self.findChild(QLineEdit, 'lineEdit')

        # Load known image and encoding
        # self.known_image = face_recognition.load_image_file("crop2.jpg")
        # self.known_encoding = face_recognition.face_encodings(self.known_image)[0]

    def train_face(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        imagePaths = [os.path.join('Dataset', f) for f in os.listdir('Dataset') if
                      os.path.isfile(os.path.join('Dataset', f))]
        faceSamples = []
        Ids = []

        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)

        recognizer.train(faceSamples, np.array(Ids))
        recognizer.save(f'Dataset/training.xml')
        print(f"Training completed and saved as f'Dataset/training.xml'")

        # self.train_face(dataset_path, f'{dataset_path}/training.xml')

    def fungsi(self):
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not video.isOpened():
            print("Error: Could not open camera.")
            return

        faceDeteksi = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if faceDeteksi.empty():
            print("Error: Could not load Haar Cascade classifier.")
            return
        id_text = self.lineEdit.text()
        if not id_text:
            self.show_error_message("Error: ID input is empty.")
            return

        try:
            id = int(id_text)
        except ValueError:
            self.show_error_message("Error: ID input must be an integer.")
            return

        a = 0
        while True:
            a += 1
            check, frame = video.read()
            if not check:
                print("Error: Failed to read frame from camera.")
                break

            abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            wajah = faceDeteksi.detectMultiScale(abu, 1.3, 5)
            for (x, y, w, h) in wajah:
                cv2.imwrite(f'Dataset/Buronan.{id}.{a}.jpg', abu[y:y + h, x:x + w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)
            if a > 29:
                break

        video.release()
        cv2.destroyAllWindows()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape) == 3:  # rows[0], cols[1], channels[2]
            if self.Image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)

    def video(self):
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        Recognizer = cv2.face.LBPHFaceRecognizer_create()
        Recognizer.read(r'Dataset/training.xml')
        a = 0
        buronan_ids = [1, 2]
        while True:
            a = a + 1
            check, frame = vid.read()  # Corrected variable name here
            abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            wajah = detector.detectMultiScale(abu, 1.3, 5)  # Corrected variable name here
            for (x, y, w, h) in wajah:
                id, conf = Recognizer.predict(abu[y:y + h, x:x + w])  # Corrected variable name here
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if (id == 1):
                    label = 'asep'
                    print("ID: ", id)
                elif (id == 2):
                    label = 'fuad'
                    print("ID: ", id)
                elif (id == 3):
                    label = 'abel'
                elif (id == 4):
                    label = 'rere'
                else:
                    label = 'uknown'
                cv2.putText(frame, label, (x + 40, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0))
            cv2.imshow("Face Recognition", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        vid.release()
        cv2.destroyAllWindows()


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Face Recognizier Siswa')
window.show()
sys.exit(app.exec_())