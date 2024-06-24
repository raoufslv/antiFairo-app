import sys
import cv2
import threading
import json
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject, Qt
from PyQt5 import QtWidgets, QtGui, QtCore
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load
import os
from datetime import datetime

import pathlib
from pathlib import Path

logging.basicConfig(filename='logs/fire_detection.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class CameraStream(QThread):
    frame_received = pyqtSignal(str, object)
    connection_failed = pyqtSignal(str)

    def __init__(self, camera_label, rtsp_url):
        super().__init__()
        self.camera_label = camera_label
        self.rtsp_url = rtsp_url
        self.running = True
        self.cap = None

    def run(self):
        if not self.open_camera_with_timeout(self.rtsp_url, timeout=7):
            self.connection_failed.emit(self.camera_label)
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_received.emit(self.camera_label, frame)
            # Adjust the sleep time to control the frame rate
            QThread.msleep(30)
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def open_camera_with_timeout(self, url, timeout):
        def attempt_open():
            self.cap = cv2.VideoCapture(url)

        thread = threading.Thread(target=attempt_open)
        thread.start()
        thread.join(timeout)

        if not thread.is_alive() and self.cap and self.cap.isOpened():
            return True
        else:
            return False


class FireSmokeDetectorApp(QtWidgets.QMainWindow):
    fire_detected_signal = pyqtSignal()  # Signal for fire detection

    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.current_camera_label = ""
        self.streams = {}

        red_pixmap = QtGui.QPixmap(10, 10)
        red_pixmap.fill(QtGui.QColor('red'))
        self.red_bullet = QtGui.QIcon(red_pixmap)

        green_pixmap = QtGui.QPixmap(10, 10)
        green_pixmap.fill(QtGui.QColor('green'))
        self.green_bullet = QtGui.QIcon(green_pixmap)

        orange_pixmap = QtGui.QPixmap(10, 10)
        orange_pixmap.fill(QtGui.QColor('orange'))
        self.orange_bullet = QtGui.QIcon(orange_pixmap)

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Fire and Smoke Detection')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)

        left_sidebar_layout = QtWidgets.QVBoxLayout()

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setText("Status: Not connected")
        left_sidebar_layout.addWidget(self.status_label)

        self.camera_list = QtWidgets.QListWidget(self)
        left_sidebar_layout.addWidget(self.camera_list)
        self.camera_list.itemClicked.connect(self.select_camera)

        main_layout.addLayout(left_sidebar_layout)

        middle_layout = QtWidgets.QVBoxLayout()

        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: grey;")
        middle_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        buttons_layout = QtWidgets.QHBoxLayout()

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_streaming)
        buttons_layout.addWidget(self.start_button)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_streaming)
        buttons_layout.addWidget(self.stop_button)

        self.add_camera_button = QtWidgets.QPushButton('Add Camera', self)
        self.add_camera_button.clicked.connect(self.add_camera)
        buttons_layout.addWidget(self.add_camera_button)

        self.delete_camera_button = QtWidgets.QPushButton(
            'Delete Camera', self)
        self.delete_camera_button.clicked.connect(self.delete_camera)
        buttons_layout.addWidget(self.delete_camera_button)

        self.stop_alarm_button = QtWidgets.QPushButton('Stop Alarm', self)
        self.stop_alarm_button.clicked.connect(self.stop_alarm)
        buttons_layout.addWidget(self.stop_alarm_button)

        middle_layout.addLayout(buttons_layout)
        main_layout.addLayout(middle_layout)

        self.load_camera_config()
        self.show()

    def show_alert(self, message):
        alert = QtWidgets.QMessageBox()
        alert.setText(message)
        stop_alarm_button = alert.addButton(
            "Stop Alarm", QtWidgets.QMessageBox.ActionRole)
        stop_alarm_button.clicked.connect(self.stop_alarm)
        alert.exec_()

    def log_event(self, event):
        logging.info(event)

    def save_camera_config(self):
        with open('assets/camera_config.json', 'w') as f:
            json.dump(self.cameras, f)

    def load_camera_config(self):
        try:
            with open('assets/camera_config.json', 'r') as f:
                self.cameras = json.load(f)
                for label, address in self.cameras.items():
                    self.camera_list.addItem(f'{label}: {address}')
                    # set the icon to red bullet
                    self.camera_list.item(self.camera_list.count() - 1).setIcon(
                        self.red_bullet)

        except FileNotFoundError:
            pass

    def add_camera(self):
        camera_label, ok = QtWidgets.QInputDialog.getText(
            self, 'Add Camera', 'Enter camera label:')
        if ok and camera_label:
            camera_address, ok = QtWidgets.QInputDialog.getText(
                self, 'Add Camera', 'Enter RTSP address:')
            if ok and camera_address:
                self.cameras[camera_label] = camera_address
                self.camera_list.addItem(f'{camera_label}: {camera_address}')
                self.save_camera_config()

    def delete_camera(self):
        selected_items = self.camera_list.selectedItems()
        if not selected_items:
            return

        confirm_dialog = QtWidgets.QMessageBox()
        confirm_dialog.setIcon(QtWidgets.QMessageBox.Question)
        confirm_dialog.setText(
            "Are you sure you want to delete the selected camera?")
        confirm_dialog.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        confirm_dialog.setWindowTitle("Confirmation")

        response = confirm_dialog.exec_()
        if response == QtWidgets.QMessageBox.Yes:
            for item in selected_items:
                del self.cameras[item.text().split(':')[0]]
                self.camera_list.takeItem(self.camera_list.row(item))
            self.save_camera_config()
            self.stop_streaming()
            self.video_label.clear()

    def select_camera(self, item):
        self.current_camera_label = item.text().split(':')[0]
        self.start_streaming()

    def start_streaming(self):
        if not self.current_camera_label:
            print("No camera selected.")
            return

        self.status_label.setText(
            f"Status: Connecting to {self.current_camera_label}...")

        # Check if the stream is already running
        if self.current_camera_label in self.streams:
            # Directly update the UI with the existing feed
            self.streams[self.current_camera_label].frame_received.connect(
                self.update_frame)
            self.status_label.setText(
                f"Status: Connected to {self.current_camera_label}")
            return

        stream = CameraStream(self.current_camera_label,
                              self.cameras[self.current_camera_label])
        stream.frame_received.connect(self.update_frame)
        stream.connection_failed.connect(self.handle_connection_failure)
        stream.start()
        self.streams[self.current_camera_label] = stream
        

    def stop_streaming(self):
        if self.current_camera_label in self.streams:
            self.streams[self.current_camera_label].stop()
            del self.streams[self.current_camera_label]
        self.status_label.setText("Status: Not connected")

    def handle_connection_failure(self, camera_label):
        if camera_label in self.streams:
            self.streams[camera_label].stop()
            del self.streams[camera_label]
        self.status_label.setText(f"Status: Can't connect to {camera_label}")
        # update the placeholder image
        self.video_label.clear()
        self.video_label.setText(f'No signal for the camera {camera_label}')

    def update_frame(self, camera_label, frame):
        if camera_label == self.current_camera_label:
            qt_image = QtGui.QImage(
                frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def stop_alarm(self):
        self.status_label.setText("Alarm stopped")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FireSmokeDetectorApp()
    sys.exit(app.exec_())
