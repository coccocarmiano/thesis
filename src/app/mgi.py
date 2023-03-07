from PIL import Image
import requests as reqs
import io
import numpy as np
from scipy import ndimage as spnd
from matplotlib import pyplot as plt

from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import pyqtSignal, QObject, QThread, QTimer, Qt, QBuffer
import cv2
from inferrer import Inferrer


class VideoPoller(QThread):
    emit_qimg = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.video_input = None
        self.timer = QTimer(self)
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.update_image)
        self.video_input = None
        self.inferrer = Inferrer()

        self.__load_fake_mask()
        self.mask_width = 1
        self.mask_opacity = 50

    def __load_fake_mask(self):
        self.mask = np.array(Image.open("../data/fake-mask.png").convert("L"))

    def update_image(self):
        frame = self.poll_video_source()
        h, w, d = frame.shape
        # mask = self.infer_mask(frame)
        qimg = QImage(frame, w, h, w * d,
                      QImage.Format_RGBA8888).scaled(1920 // 3, 1080 // 3)
        self.emit_qimg.emit(qimg)

    def update_mask_width(self, value):
        self.mask_width = value

    def update_mask_opacity(self, value):
        self.mask_opacity = value

    def poll_video_source(self):
        _, frame = self.video_input.read()
        while frame is None:
            print("Frame is none")
            self.connect_video()
            _, frame = self.video_input.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return frame

    def connect_video(self):
        self.timer.stop()
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

    def setup_ui(self):
        self.layout = QHBoxLayout()

        self.mask_opacity_slider = GUISlider(
            label="Opacità", min_value=10, max_value=100)
        self.mask_width_slider = GUISlider(
            label="Spessore", min_value=2, max_value=15)

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

        self.layout = QHBoxLayout()

        self.video_poller = VideoPoller()
        self.video_poller.emit_qimg.connect(self.update_video_display)

        self.video_display = QLabel()
        self.layout.addWidget(self.video_display, alignment=Qt.AlignCenter)

        self.mask_controls = Controls()
        self.mask_controls.setMaximumWidth(400)
        self.layout.addWidget(self.mask_controls)
        self.mask_controls.emit_mask_width.connect(
            self.video_poller.update_mask_width)
        self.mask_controls.emit_mask_opacity.connect(
            self.video_poller.update_mask_opacity)

        self.root = QWidget()
        self.root.setLayout(self.layout)
        self.setCentralWidget(self.root)

    def update_video_display(self, qimg):
        mask = self.predict_mask(qimg)
        mask = self.process_mask(mask)
        blend = self.blend_mask(qimg, mask)
        pixmap = self.npa_to_qpixmap(blend)
        self.video_display.setPixmap(pixmap)

    def process_mask(self, mask):
        width = self.mask_controls.mask_width_slider.slider.value()
        grow = spnd.binary_dilation(mask, iterations=width)
        grow[mask == True] = 0
        return grow

    def npa_to_qpixmap(self, npa):
        h, w, d = npa.shape
        return QPixmap.fromImage(QImage(npa, w, h, w * d, QImage.Format_RGBA8888))

    def predict_mask(self, qimg):
        qbuf = QBuffer()
        qbuf.open(QBuffer.ReadWrite)
        qimg.save(qbuf, "JPG")
        to_send = io.BytesIO(qbuf.data())
        h, w = qimg.height(), qimg.width()

        resp = reqs.post(
            "http://localhost:8080/predictions/mgi",
            data=to_send
        )

        if resp.status_code == 200:
            mask = np.array(Image.open(io.BytesIO(resp.content)).convert("L"))
            mask = cv2.resize(mask, (w, h))
            mask = mask.astype(bool, copy=False)
        else:
            mask = np.zeros((h, w), dtype=bool)
        return mask

    def qimg_to_np(self, qimg):
        qbuf = QBuffer()
        qbuf.open(QBuffer.ReadWrite)
        qimg.save(qbuf, "JPG")
        to_send = io.BytesIO(qbuf.data())
        return np.array(Image.open(to_send).convert("RGBA"))

    def blend_mask(self, qimg, mask):
        frame = self.qimg_to_np(qimg)
        op = self.mask_controls.mask_opacity_slider.slider.value() / 100
        frame[mask] = np.array([255, 0, 0, 255]) * op + frame[mask] * (1 - op)
        return frame

    def select_source(self, idx):
        self.source = idx
        self.video_poller.source_idx = idx
        self.video_poller.run()
        self.show()


class GUISlider(QWidget):
    def __init__(self, **kwargs):
        super().__init__()
        self.min_value = kwargs.get("min_value", 0)
        self.max_value = kwargs.get("max_value", 100)
        self.label_text = kwargs.get("label", "Slider")

        self.label = QLabel(self.label_text)
        self.label.setFont(QFont("Arial", 15))

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(self.min_value)
        self.slider.setMaximum(self.max_value)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        self.layout.addWidget(self.slider, alignment=Qt.AlignHCenter)
        self.setLayout(self.layout)

    def setMaximum(self, value):
        self.slider.setMaximum(value)

    def setMinimum(self, value):
        self.slider.setMinimum(value)

    def setValue(self, value):
        self.slider.setValue(value)

    @property
    def valueChanged(self):
        return self.slider.valueChanged
