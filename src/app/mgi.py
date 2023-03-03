import time
import logging
from PIL import Image
import numpy as np
from scipy import ndimage as spnd
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer, Qt
from PIL import Image
import cv2


class VideoPoller(QThread):
    emit_qimg = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.video_input = None
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.update_image)
        self.video_input = None

        self.__load_fake_mask()
        self.mask_width = 1
        self.mask_opacity = 50

    def __load_fake_mask(self):
        self.mask = np.array(Image.open("../data/fake-mask.png").convert("L"))

    def update_image(self):
        frame = self.poll_video_source()
        h, w, d = frame.shape
        # mask = self.infer_mask(frame)
        mask = self.process_mask(h, w)
        mask = self.blend_mask(frame, mask)
        qimg = QImage(mask, w, h, w * d,
                      QImage.Format_RGB888).scaled(1920 // 3, 1080 // 3)
        self.emit_qimg.emit(qimg)

    def blend_mask(self, frame, mask):
        # start = time.time()
        frame[mask] = [255, 0, 0]
        end = time.time()
        # logging.debug(f"Blend mask took {(end - start)*1000} ms")
        return frame

    def process_mask(self, h, w):
        self.mask = cv2.resize(self.mask, (w, h))
        arr = np.array(self.mask)
        arr = arr.astype(bool)
        grow = spnd.binary_dilation(arr, iterations=self.mask_width)
        grow[arr == True] = 0
        return grow

    def update_mask_width(self, value):
        self.mask_width = value

    def poll_video_source(self):
        frame = None
        while frame is None:
            _, frame = self.video_input.read()
            self.connect_video()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def connect_video(self):
        if self.video_input is not None:
            if self.video_input.isOpened():
                self.video_input.release()
        self.video_input = cv2.VideoCapture(self.source_idx, cv2.CAP_V4L)
        self.timer.start()

    def run(self):
        self.connect_video()


class Controls(QWidget):
    emit_mask_opacity = pyqtSignal(int)
    emit_mask_width = pyqtSignal(int)

    def __init__(self):
        super().__init__()

        self.setup_ui()
        self.mask_opacity_slider.setValue(20)
        self.mask_width_slider.setValue(20)

    def setup_ui(self):
        self.layout = QHBoxLayout()

        self.mask_opacity_slider = QSlider(Qt.Vertical)
        self.mask_width_slider = QSlider(Qt.Vertical)

        self.mask_opacity_slider.setMaximum(90)
        self.mask_opacity_slider.setMinimum(20)

        self.mask_width_slider.setMaximum(20)
        self.mask_width_slider.setMinimum(2)

        self.mask_opacity_slider.valueChanged.connect(self.update_mask_opacity)
        self.mask_width_slider.valueChanged.connect(self.update_mask_width)

        self.layout.addWidget(self.mask_width_slider,
                              alignment=Qt.AlignmentFlag.AlignRight)
        self.layout.addWidget(self.mask_opacity_slider)
        self.setLayout(self.layout)

    def update_mask_width(self, value):
        self.mask_width = value
        self.emit_mask_width.emit(value)

    def update_mask_opacity(self, value):
        self.mask_opacity = value
        self.emit_mask_opacity.emit(value)


class MGIWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.hide()
        self.setWindowTitle("MGI")
        self.resize(800, 600)
        self.idx = 0

        self.video_poller = VideoPoller()
        self.video_poller.emit_qimg.connect(self.update_video_display)

        self.video_display = QLabel()
        self.video_display.setAlignment(Qt.AlignCenter)
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.video_display)

        self.mask_controls = Controls()
        self.layout.addWidget(self.mask_controls)
        self.mask_controls.emit_mask_width.connect(
            self.video_poller.update_mask_width)

        self.root = QWidget()
        self.root.setLayout(self.layout)
        self.setCentralWidget(self.root)

    def update_video_display(self, qimg):
        self.video_display.setPixmap(QPixmap.fromImage(qimg))

    def select_source(self, idx):
        self.source = idx
        self.video_poller.source_idx = idx
        self.video_poller.run()
        self.show()
