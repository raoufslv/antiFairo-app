from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.models.experimental import attempt_load
import pathlib
from pathlib import Path
import sys
import torch
import cv2
import threading
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtMultimedia import QSound
import json
import logging

logging.basicConfig(filename='fire_detection.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


# Ensure PosixPath is replaced by WindowsPath on Windows
pathlib.PosixPath = pathlib.WindowsPath
defaultCamera = "http://192.168.213.21:4747"

# Add the YOLOv5 repository to the Python path
yolov5_path = Path('yolov5')
utils_path = Path('yolov5/utils')
sys.path.append(str(yolov5_path))
sys.path.append(str(utils_path))


# Path to your custom YOLOv5 model weights
model_path = Path('bestbestbest.pt')

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
fire_conf_threshold = 0.7  # Adjust as needed
smoke_conf_threshold = 0.7  # Adjust as needed


class FireSmokeDetectorApp(QtWidgets.QMainWindow):
    fire_detected_signal = pyqtSignal()  # Signal for fire detection

    def __init__(self):
        super().__init__()
        # List to store RTSP addresses of cameras
        self.cameras = ["Laptop Camera (Default)"]
        self.current_camera_index = 0
        self.capture = None
        self.initUI()
        self.lock = threading.Lock()
        self.current_frame = None
        self.processed_frame = None
        self.processing_thread = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frame_skip = 10  # Process every 10th frame
        self.frame_count = 0
        self.background_processing = False

        # Flag to track alarm status
        self.alarm_on = False

        # Initialize variables for temporal consistency
        self.fire_detected = False
        self.smoke_detected = False
        self.fire_frame_count = 0
        self.smoke_frame_count = 0
        self.consistent_detection_threshold = 5  # Adjust as needed

        self.fire_detected_signal.connect(
            self.handle_fire_detection)  # Connect signal to slot

    def initUI(self):
        self.setWindowTitle('Fire and Smoke Detection')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        self.grid_layout = QtWidgets.QGridLayout(central_widget)

        self.status_label = QtWidgets.QLabel(self)
        self.status_label.setText("Status: Not connected")
        self.grid_layout.addWidget(self.status_label, 0, 0)

        self.video_label = QtWidgets.QLabel(self)
        self.grid_layout.addWidget(self.video_label, 1, 0)

        self.start_button = QtWidgets.QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_streaming)
        self.grid_layout.addWidget(self.start_button, 2, 0)

        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop_streaming)
        self.grid_layout.addWidget(self.stop_button, 2, 1)

        self.add_camera_button = QtWidgets.QPushButton('Add Camera', self)
        self.add_camera_button.clicked.connect(self.add_camera)
        self.grid_layout.addWidget(self.add_camera_button, 3, 0)

        self.delete_camera_button = QtWidgets.QPushButton(
            'Delete Camera', self)
        self.delete_camera_button.clicked.connect(self.delete_camera)
        self.grid_layout.addWidget(self.delete_camera_button, 3, 1)

        self.camera_list = QtWidgets.QListWidget(self)
        self.grid_layout.addWidget(self.camera_list, 1, 1, 2, 1)
        self.camera_list.addItem("Laptop Camera (Default)")
        self.camera_list.itemClicked.connect(self.select_camera)

        self.load_camera_config()
        self.show()

    def show_alert(self, message):
        alert = QtWidgets.QMessageBox()
        alert.setWindowTitle("Alert")
        alert.setText(message)
        alert.setIcon(QtWidgets.QMessageBox.Information)
        
        stop_alarm_button = alert.addButton("Stop Alarm", QtWidgets.QMessageBox.AcceptRole)
        alert.setDefaultButton(stop_alarm_button)
        stop_alarm_button.clicked.connect(self.stop_alarm)
        
        alert.exec_()


    def log_event(self, event):
        logging.info(event)

    def save_camera_config(self):
        with open('camera_config.json', 'w') as f:
            json.dump(self.cameras, f)

    def load_camera_config(self):
        try:
            with open('camera_config.json', 'r') as f:
                self.cameras = json.load(f)
                for camera in self.cameras:
                    self.camera_list.addItem(camera)
        except FileNotFoundError:
            pass

    def add_camera(self):
        camera_address, ok = QtWidgets.QInputDialog.getText(
            self, 'Add Camera', 'Enter RTSP address:')
        if ok and camera_address:
            self.cameras.append(camera_address)
            self.camera_list.addItem(camera_address)

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
                self.cameras.remove(item.text())
                self.camera_list.takeItem(self.camera_list.row(item))

    def select_camera(self, item):
        self.current_camera_index = self.camera_list.row(item)
        self.stop_streaming()
        self.start_streaming()

    def start_streaming(self):
        if self.current_camera_index == -1:
            print("No camera selected.")
            return

        if self.current_camera_index == 0:
            self.video_path = defaultCamera  # Use laptop camera
        else:
            self.video_path = self.cameras[self.current_camera_index]

        if self.capture is None or not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.video_path)
            if not self.capture.isOpened():
                print("Error: Unable to open video stream")
                return

        if not self.timer.isActive() and not self.background_processing:
            self.timer.start(30)  # Update every 30 ms

        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(
                target=self.process_frames)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def stop_streaming(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()

    def restart_streaming(self):
        self.stop_streaming()
        self.start_streaming()

    def toggle_background_processing(self, state):
        if state == QtCore.Qt.Checked:
            self.background_processing = True
            if self.timer.isActive():
                self.timer.stop()
        else:
            self.background_processing = False
            if not self.timer.isActive() and self.capture is not None and self.capture.isOpened():
                self.timer.start(30)

    def update_frame(self):
        if self.capture is not None:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.current_frame = frame
                if not self.background_processing and self.processed_frame is not None:
                    qt_image = QtGui.QImage(
                        self.processed_frame.data, self.processed_frame.shape[1], self.processed_frame.shape[0], self.processed_frame.strides[0], QtGui.QImage.Format_RGB888)
                    self.video_label.setPixmap(
                        QtGui.QPixmap.fromImage(qt_image))

    def process_frames(self):
        while self.capture is not None and self.capture.isOpened():
            if self.current_frame is not None:
                with self.lock:
                    frame = self.current_frame.copy()
                if self.frame_count % self.frame_skip == 0:
                    # Convert frame color space from BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Resize frame to a lower resolution
                    target_size = (320, 320)
                    resized_frame = cv2.resize(frame_rgb, target_size)

                    # Convert frame to a torch tensor
                    img = torch.from_numpy(resized_frame).to(device)
                    # Change shape to [3, H, W]
                    img = img.permute(2, 0, 1).float()
                    img /= 255.0  # Normalize to [0, 1]
                    img = img.unsqueeze(0)  # Add batch dimension [1, 3, H, W]

                    # Run the model
                    with torch.no_grad():
                        results = model(img)[0]

                    # Apply NMS (non-maximum suppression)
                    results = non_max_suppression(results)

                    # Process results
                    self.processed_frame = self.plot_boxes(
                        results, resized_frame, img)

                    self.check_consistent_detection()
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
                self.fire_detected_signal.emit()  # Emit fire detection signal
        else:
            self.fire_frame_count = 0

        if self.smoke_detected:
            self.smoke_frame_count += 1
            if self.smoke_frame_count >= self.consistent_detection_threshold:
                self.fire_detected_signal.emit()  # Emit fire detection signal
        else:
            self.smoke_frame_count = 0

    def handle_fire_detection(self):
        # Check if alarm is already on
        if not self.alarm_on:
            # Set alarm flag to true
            self.alarm_on = True
            print("Fire detected! terminal")
            # Play an alert sound (make sure alert.wav exists in the working directory)
            QSound.play("alert.wav")
            # Additional actions (logging, sending alerts, etc.) can be added here
            self.log_event("Fire detected! file logs")
            # send alert
            self.show_alert("Fire detected!")

    def stop_alarm(self):
        # Turn off alarm
        self.alarm_on = False

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FireSmokeDetectorApp()
    sys.exit(app.exec_())
