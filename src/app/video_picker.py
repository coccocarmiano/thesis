from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QCoreApplication, QTimer
from time import sleep
from PIL import Image
import cv2


def list_video_sources():
    idx = 0
    arr = []
    for idx in range(10):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L)
        ok, _ = cap.read()
        if ok:
            arr.append(idx)
        cap.release()
    return arr


class ThumbnailUpdater(QThread):
    emit_image = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.video_input = cv2.VideoCapture(self.source, cv2.CAP_V4L)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timeout)
        self.timer.setInterval(0)

    def timeout(self):
        _, frame = self.video_input.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, d = frame.shape
        qimg = QImage(frame, w, h, w * d,
                      QImage.Format_RGB888).scaled(360, 240)
        self.emit_image.emit(qimg)

    def stop(self):
        self.timer.stop()
        print("Relesing video source", self.source)
        self.video_input.release()

    def __del__(self):
        print(f"Closing video source {self.source}...")


class VideoInputItem(QWidget):

    def update_image(self, qimg):
        self.thumbnail.setPixmap(QPixmap.fromImage(qimg))

    def __init__(self, source_idx, father):
        super().__init__()
        self.source_idx = source_idx
        self.father = father

        self.thumbnail = QLabel()
        self.text = QLabel()
        self.btn = QPushButton("Usa")
        self.layout = QHBoxLayout(self)

        self.layout.addWidget(self.thumbnail)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.btn)
        self.setLayout(self.layout)

        # Set Text
        self.text.setStyleSheet("padding: 10px;")
        self.text.setText(f"Sorgente Video #{source_idx+1}")
        font = self.text.font()
        font.setPointSize(20)
        self.text.setFont(font)

        # Set Button
        self.btn.setStyleSheet("padding: 10px;")
        font = self.btn.font()
        font.setPointSize(20)
        self.btn.setFont(font)
        self.btn.pressed.connect(
            lambda: self.father.notify_source(self.source_idx))

        # Thumbnail
        self.worker = ThumbnailUpdater(self.source_idx)
        self.worker.emit_image.connect(self.update_image)
        self.worker.started.connect(self.worker.run)
        self.worker.timer.start()
        self.worker.start()


class VideoInputPicker(QMainWindow):
    emit_video_source = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seleziona Ingresso Video")
        self.root = QWidget()

        vl = QVBoxLayout()

        self.sources = [
            VideoInputItem(source, self) for source in list_video_sources()
        ]

        for source in self.sources:
            vl.addWidget(source)
            source.btn.pressed.connect(self.stop_workers)

        self.root.setLayout(vl)
        self.setCentralWidget(self.root)
        self.show()

    def notify_source(self, source):
        self.emit_video_source.emit(source)

    def stop_workers(self):
        for source in self.sources:
            source.worker.stop()
        self.close()
