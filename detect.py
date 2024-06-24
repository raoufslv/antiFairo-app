import sys
import cv2
import threading
import json
import logging
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject, Qt
from PyQt5 import QtWidgets, QtGui, QtCore
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load
import os
from datetime import datetime
from PyQt5.QtMultimedia import QSound

import pathlib
from pathlib import Path

logging.basicConfig(filename='logs/fire_detection.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Ensure PosixPath is replaced by WindowsPath on Windows
pathlib.PosixPath = pathlib.WindowsPath

# Add the YOLOv5 repository to the Python path
yolov5_path = Path('yolov5')
utils_path = Path('yolov5/utils')
sys.path.append(str(yolov5_path))
sys.path.append(str(utils_path))

# Path to your custom YOLOv5 model weights
model_path = Path('models/EfficientDet200.pt')


# Select device
device = select_device('cpu')

# Load the YOLOv5 model
try:
    model = attempt_load(model_path)
    model.to(device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define confidence thresholds for fire and smoke classes
fire_conf_threshold = 0.35  # Adjust as needed
smoke_conf_threshold = 0.35  # Adjust as needed

# Define ALERT VARIABLE GLOBAL
ALERT_GLOBAL = False

# Ensure 'dataset' directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')
if not os.path.exists('dataset/images'):
    os.makedirs('dataset/images')
if not os.path.exists('dataset/labels'):
    os.makedirs('dataset/labels')
if not os.path.exists('dataset/predicts'):
    os.makedirs('dataset/predicts')
if not os.path.exists('logs'):
    os.makedirs('logs')
if not os.path.exists('logs/images'):
    os.makedirs('logs/images')


class CameraStream(QThread):
    frame_received = pyqtSignal(str, object)
    connection_success = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    fire_detected_signal = pyqtSignal(object)
    colorid_bg_label = pyqtSignal(object)

    def __init__(self, camera_label, rtsp_url):
        super().__init__()
        self.camera_label = camera_label
        self.rtsp_url = rtsp_url
        self.running = True
        self.cap = None
        self.processed_frame = None
        self.frame_skip = 5  # Process every 10th frame
        self.frame_count = 0
        # Flag to track alarm status
        self.alarm_on = False

        self.save_frame_count = 0  # Counter for frames saved
        # Interval to skip frames (adjust as needed)
        self.save_frame_interval = 10

        # Initialize variables for temporal consistency
        self.fire_detected = False
        self.smoke_detected = False
        self.fire_frame_count = 0
        self.smoke_frame_count = 0
        self.consistent_detection_threshold = 4  # Adjust as needed

    def run(self):
        if not self.open_camera_with_timeout(self.rtsp_url, timeout=10):
            self.connection_failed.emit(self.camera_label)
            return

        self.connection_success.emit(self.camera_label)
        # Start processing frames
        self.process_frames()

    def process_frames(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_count % self.frame_skip == 0:
                    # Convert frame color space from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize frame to a lower resolution
                    target_size = (320, 320)
                    resized_frame = cv2.resize(frame_rgb, target_size)

                    # Convert frame to a torch tensor
                    img = torch.from_numpy(resized_frame).to(device)
                    img = img.permute(2, 0, 1).float()
                    img /= 255.0  # Normalize to [0, 1]
                    img = img.unsqueeze(0)  # Add batch dimension [1, 3, H, W]

                    # Run the model
                    with torch.no_grad():
                        results = model(img)[0]

                    # Apply NMS (non-maximum suppression)
                    results = non_max_suppression(results)

                    # Process results and save frame with bounding boxes
                    self.processed_frame = self.plot_boxes(
                        results, resized_frame, img)

                    if self.alarm_on:
                        if self.fire_detected or self.smoke_detected:
                            if self.save_frame_count % self.save_frame_interval == 0:
                                self.save_detection_frame(
                                    frame_rgb, self.processed_frame, results)
                            self.save_frame_count += 1

                    self.check_consistent_detection()
                    self.frame_received.emit(
                        self.camera_label, self.processed_frame)
                self.frame_count += 1

    def plot_boxes(self, results, frame, img):
        # Reset detection flags
        self.fire_detected = False
        self.smoke_detected = False
        for det in results:
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    if int(cls) == 0 and conf >= fire_conf_threshold:  # Fire class
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        self.draw_box(frame, xyxy, label, color=(
                            0, 0, 255))  # Red color in BGR
                        self.fire_detected = True

                    elif int(cls) == 1 and conf >= smoke_conf_threshold:  # Smoke class
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        self.draw_box(frame, xyxy, label, color=(
                            128, 128, 128))  # Gray color in BGR
                        self.smoke_detected = True

        return frame

    def draw_box(self, img, xyxy, label=None, color=(255, 0, 0), line_thickness=2):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, color,
                      thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            tf = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(
                label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [
                        225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def check_consistent_detection(self):
        if self.fire_detected:
            self.fire_frame_count += 1
            if self.fire_frame_count >= self.consistent_detection_threshold:
                self.handle_fire_detection()
        else:
            self.fire_frame_count = 0

        if self.smoke_detected:
            self.smoke_frame_count += 1
            if self.smoke_frame_count >= self.consistent_detection_threshold:
                self.handle_fire_detection()
        else:
            self.smoke_frame_count = 0

    def log_event(self, event):
        logging.info(event)

    def handle_fire_detection(self):
        # Check if alarm is already on
        if not self.alarm_on:
            # Set alarm flag to true
            self.alarm_on = True
            self.colorid_bg_label.emit(self.camera_label)
            print("Fire detected! Terminal alert")
            # Additional actions (logging, sending alerts, etc.) can be added here
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_event(
                f'Fire detected! image link: logs/images/frame_{timestamp}.jpg')
            cv2.imwrite(
                f'logs/images/frame_{timestamp}.jpg', cv2.cvtColor(self.processed_frame, cv2.COLOR_RGB2BGR))

            global ALERT_GLOBAL
            if not ALERT_GLOBAL:
                ALERT_GLOBAL = True
                self.fire_detected_signal.emit(self.camera_label)

    def save_detection_frame(self, original_frame, processed_frame, results):
        # Reconvert to BGR
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'dataset/images/frame_{timestamp}.jpg', original_frame)
        cv2.imwrite(f'dataset/predicts/frame_{timestamp}.jpg', processed_frame)

        with open(f'dataset/labels/frame_{timestamp}.txt', 'w') as f:
            for det in results:
                if len(det):
                    for *xyxy, conf, cls in det:
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2
                        width = xyxy[2] - xyxy[0]
                        height = xyxy[3] - xyxy[1]
                        f.write(
                            f"{int(cls)} {x_center/processed_frame.shape[1]} {y_center/processed_frame.shape[0]} {width/processed_frame.shape[1]} {height/processed_frame.shape[0]}\n")

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

    def __init__(self):
        super().__init__()
        self.cameras = {}
        self.current_camera_label = ""
        self.streams = {}

        self.alert_sound = QSound("assets/alert.wav")

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
        self.video_label.setFixedSize(320, 320)
        self.video_label.setStyleSheet("background-color: grey; color: white; font-weight: bold;")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)  # Ensure text is centered
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
        self.connect_all_cameras()  # Connect to all cameras when the interface is launched
        self.show()

    def show_alert(self, camera_label):
        # Play an alert sound in a loop
        self.alert_sound.setLoops(QSound.Infinite)
        self.alert_sound.play()
        alert = QtWidgets.QMessageBox()
        alert.setText(f'{camera_label} detected fire!')
        stop_alarm_button = alert.addButton(
            "Stop Alarm", QtWidgets.QMessageBox.ActionRole)
        stop_alarm_button.clicked.connect(self.stop_alarm)
        alert.exec_()

    def save_camera_config(self):
        with open('assets/camera_config.json', 'w') as f:
            json.dump(self.cameras, f)

    def load_camera_config(self):
        try:
            with open('assets/camera_config.json', 'r') as f:
                self.cameras = json.load(f)
                for label, address in self.cameras.items():
                    self.camera_list.addItem(f'{label}: {address}')
                    self.camera_list.item(
                        self.camera_list.count() - 1).setIcon(self.red_bullet)
        except FileNotFoundError:
            pass

    def connect_all_cameras(self):
        for camera_label in self.cameras.keys():
            self.current_camera_label = camera_label
            self.start_streaming()
        # check if

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

        # Update the icon to orange (connecting)
        self.update_camera_icon(self.current_camera_label, self.orange_bullet)

        # Check if the stream is already running
        if self.current_camera_label in self.streams:
            self.streams[self.current_camera_label].frame_received.connect(
                self.update_frame)
            self.status_label.setText(
                f"Status: Connected to {self.current_camera_label}")
            # Update the icon to green (connected)
            self.update_camera_icon(
                self.current_camera_label, self.green_bullet)
            return

        stream = CameraStream(self.current_camera_label,
                              self.cameras[self.current_camera_label])
        stream.frame_received.connect(self.update_frame)
        stream.connection_success.connect(self.camera_started)

        stream.connection_failed.connect(self.handle_connection_failure)
        stream.fire_detected_signal.connect(self.show_alert)
        stream.colorid_bg_label.connect(self.update_label_bg)
        stream.start()
        self.streams[self.current_camera_label] = stream

    def camera_started(self, camera_label):
        self.status_label.setText(f"Status: Connected to {camera_label}")
        # Update the icon to green (connected)
        self.update_camera_icon(camera_label, self.green_bullet)
        # make it the one selected and start streaming it
        self.current_camera_label = camera_label
        self.start_streaming()

    def update_label_bg(self, label_camera):
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            if item.text().startswith(label_camera):
                item.setBackground(QtGui.QColor(255, 0, 0, 50))
                # make it the one selected and start streaming it
                self.current_camera_label = label_camera
                self.start_streaming()
                break

    def stop_streaming(self):
        if self.current_camera_label in self.streams:
            self.streams[self.current_camera_label].stop()
            del self.streams[self.current_camera_label]
        self.status_label.setText("Status: Not connected")
        # Update the icon to red (disconnected)
        self.update_camera_icon(self.current_camera_label, self.red_bullet)

    def handle_connection_failure(self, camera_label):
        if camera_label in self.streams:
            self.streams[camera_label].stop()
            del self.streams[camera_label]
        self.status_label.setText(f"Status: Can't connect to {camera_label}")
        self.video_label.clear()
        self.video_label.setText(f'No signal for the camera {camera_label}')
        # Update the icon to red (connection failed)
        self.update_camera_icon(camera_label, self.red_bullet)

    def update_frame(self, camera_label, frame):
        if camera_label == self.current_camera_label:
            qt_image = QtGui.QImage(
                frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            # Update the icon to green (connected)
            self.update_camera_icon(camera_label, self.green_bullet)

    def stop_alarm(self):
        self.status_label.setText("Alarm stopped")
        self.alert_sound.stop()
        global ALERT_GLOBAL
        ALERT_GLOBAL = False
        # Set the alarm_on variable to False for all active camera streams
        for stream in self.streams.values():
            stream.alarm_on = False
        # clear the background color of the camera label
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            item.setBackground(QtGui.QColor('white'))

    def update_camera_icon(self, camera_label, icon):
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            if item.text().startswith(camera_label):
                item.setIcon(icon)
                break


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FireSmokeDetectorApp()
    sys.exit(app.exec_())
