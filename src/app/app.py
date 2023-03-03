import logging

from PyQt5.QtWidgets import QApplication
from video_picker import VideoInputPicker
from mgi import MGIWindow

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    app = QApplication([])
    displayer = MGIWindow()
    source_picker = VideoInputPicker()
    source_picker.emit_video_source.connect(displayer.select_source)
    app.exec_()