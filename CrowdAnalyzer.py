import sys
import os
import cv2
import torch
import random
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import markdown
from datetime import datetime
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from pedpy import (
    TrajectoryData, plot_trajectories, WalkableArea, MeasurementArea,
    compute_classic_density, compute_individual_voronoi_polygons, compute_voronoi_density,
    Cutoff, SpeedCalculation, compute_individual_speed, compute_mean_speed_per_frame,
    compute_voronoi_speed, PEDPY_BLUE, PEDPY_GREY, PEDPY_ORANGE, PEDPY_RED
)
from groq import Groq
from dotenv import load_dotenv
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QFileDialog, QInputDialog, QLineEdit, QComboBox, QDialog, QFormLayout,
    QDialogButtonBox, QMessageBox, QFrame, QTabWidget, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.layout = QFormLayout(self)
        self.layout.addRow(QLabel("<b>YOLO parameters</b>"))
        self.yolo_model_combo = QComboBox(self)
        self.yolo_model_combo.addItems(["yolo11x", "yolo11l", "yolo11m", "yolo11s", "yolo11n"])
        self.yolo_model_combo.setCurrentText(self.parent().settings.get('yolo_model', 'yolo11x'))
        self.layout.addRow("YOLO Model:", self.yolo_model_combo)
        self.track_model_combo = QComboBox(self)
        self.track_model_combo.addItems(["bytetrack.yaml", "botsort.yaml"])
        self.track_model_combo.setCurrentText(self.parent().settings.get('track_model', 'bytetrack.yaml'))
        self.layout.addRow("Track Model:", self.track_model_combo)
        self.output_folder_button = QPushButton("Choose Folder", self)
        self.output_folder_button.clicked.connect(self.choose_folder)
        self.layout.addRow("Output Folder:", self.output_folder_button)
        current_output_folder = self.parent().settings.get('output_folder', '')
        if current_output_folder:
            self.output_folder = current_output_folder
            self.output_folder_button.setText(f"Folder: {current_output_folder}")
        else:
            self.output_folder = None

        self.conf_threshold_input = QLineEdit(self)
        self.conf_threshold_input.setText(str(self.parent().settings.get('conf_threshold', 0.3)))
        self.layout.addRow("Confidence Threshold:", self.conf_threshold_input)
        self.iou_threshold_input = QLineEdit(self)
        self.iou_threshold_input.setText(str(self.parent().settings.get('iou_threshold', 0.5)))
        self.layout.addRow("IoU Threshold:", self.iou_threshold_input)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addRow(line)
        self.layout.addRow(QLabel("<b>PedPy parameters</b>"))

        self.frame_rate_input = QLineEdit(self)
        self.frame_rate_input.setText(str(self.parent().settings.get('frame_rate', 30)))
        self.layout.addRow("Frame Rate:", self.frame_rate_input)

        self.walkable_area_input = QLineEdit(self)
        walkable_area = self.parent().settings.get('walkable_area', None)
        if walkable_area is not None:
            walkable_area_text = ' '.join([f"{x},{y}" for x, y in walkable_area])
            self.walkable_area_input.setText(walkable_area_text)
        else:
            self.walkable_area_input.setText("NA")
        self.layout.addRow("Walkable Area (x1,y1 x2,y2 ...):", self.walkable_area_input)

        self.measurement_area_input = QLineEdit(self)
        measurement_area = self.parent().settings.get('measurement_area', None)
        if measurement_area is not None:
            measurement_area_text = ' '.join([f"{x},{y}" for x, y in measurement_area])
            self.measurement_area_input.setText(measurement_area_text)
        else:
            self.measurement_area_input.setText("NA")
        self.layout.addRow("Measurement Area (x1,y1 x2,y2 ...):", self.measurement_area_input)

        self.frame_step_input = QLineEdit(self)
        self.frame_step_input.setText(str(self.parent().settings.get('frame_step', 25)))
        self.layout.addRow("Frame Step:", self.frame_step_input)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_button.setText(f"Folder: {self.output_folder}")

    def get_settings(self):
        walkable_area = self.parse_polygon(self.walkable_area_input.text())
        measurement_area = self.parse_polygon(self.measurement_area_input.text())
        if walkable_area is None or measurement_area is None:
            return None

        return {
            "yolo_model": self.yolo_model_combo.currentText(),
            "track_model": self.track_model_combo.currentText(),
            "output_folder": self.output_folder,
            "conf_threshold": float(self.conf_threshold_input.text()),
            "iou_threshold": float(self.iou_threshold_input.text()),
            "frame_rate": int(self.frame_rate_input.text()),
            "walkable_area": walkable_area,
            "measurement_area": measurement_area,
            "frame_step": int(self.frame_step_input.text())
        }

    def parse_polygon(self, text):
        if text == "NA":
            return None
        try:
            points = [tuple(map(float, point.split(','))) for point in text.strip().split()]
            if len(points) < 3:
                QMessageBox.critical(self, "Error", "A polygon must have at least 3 points.")
                return None
            return points
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid polygon format: {e}")
            return None
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crowd Analyzer")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "logo.png")))
        icon_path = os.path.join(os.path.dirname(__file__), "img", "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Icon not found at {icon_path}")

        self.layout = QVBoxLayout()

        self.video_label = QLabel("No video loaded", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.load_button = QPushButton("Load Video", self)
        self.load_button.clicked.connect(self.load_video)
        self.layout.addWidget(self.load_button)

        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self.settings_button = QPushButton("Settings", self)
        self.settings_button.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_button)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.video_path = ""
        self.settings = {
            "yolo_model": "yolo11x",
            "track_model": "bytetrack.yaml",
            "output_folder": "",
            "conf_threshold": 0.3,
            "iou_threshold": 0.5,
            "frame_rate": 30,
            "walkable_area": None,
            "measurement_area": None,
            "frame_step": 25
        }

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.estimator = None
        self.frame_number = 0
        self.video_writer = None
        self.output_path = None
        self.output_filename = None

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.video_label.setText(f"Loaded video: {self.video_path}")
            if not self.settings["output_folder"]:
                self.settings["output_folder"] = os.path.dirname(self.video_path)

    def start_processing(self):
        if not self.video_path:
            self.video_label.setText("Please load a video first")
            return

        if not self.settings['output_folder']:
            self.settings['output_folder'] = os.path.dirname(self.video_path)

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.video_label.setText("Error opening video stream or file")
            return

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.output_filename = f"{base_name}_processed_{timestamp}.mp4"
        self.output_path = os.path.join(self.settings['output_folder'], self.output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            fps,
            (frame_width, frame_height),
            isColor=True
        )

        if not self.video_writer.isOpened():
            self.video_label.setText("Failed to create video writer")
            return

        self.video_label.setText(f"Processing video... Output will be saved to: {self.output_path}")

        ret, frame = self.cap.read()
        if not ret:
            self.video_label.setText("Error reading first frame")
            return

        points_image, points_world = self.select_points_and_distances(frame)
        if points_image is None or points_world is None:
            self.video_label.setText("Need 4 points and distances for perspective transform")
            self.video_writer.release()
            return

        self.estimator = CrowdDensityEstimation(
            model_path=f"{self.settings['yolo_model']}.pt",
            conf_threshold=self.settings["conf_threshold"],
            iou_threshold=self.settings["iou_threshold"]
        )
        self.estimator.set_homography_matrix(points_image, points_world)

        self.frame_number = 0
        self.timer.start(30) 

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.release()
            if self.video_writer and self.video_writer.isOpened():
                self.video_writer.release()
                print(f"Video saved to: {self.output_path}")
                self.video_label.setText(f"Processing complete.\nVideo saved to: {self.output_path}")
            self.estimator.save_trajectories(f"{self.settings['output_folder']}/trajectories.csv")
            self.calculate_default_areas()
            self.show_plots()
            return
        
        results, resized_frame = self.estimator.extract_tracks(frame)

        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            conf_scores = results[0].boxes.conf.cpu()
            
            for box, conf in zip(boxes, conf_scores):
                if conf > self.settings["conf_threshold"]:
                    detections.append(box.numpy())

        self.estimator.update_tracks(detections)

        processed_frame, density_info = self.estimator.process_frame(frame, self.frame_number)

        if processed_frame.shape[:2] != (frame.shape[0], frame.shape[1]):
            processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))

        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        self.video_writer.write(processed_frame)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))
        
        self.frame_number += 1

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            new_settings = dialog.get_settings()
            if new_settings:
                self.settings.update(new_settings)

    def select_points_and_distances(self, frame):
        selector = PointSelector("Select 4 points")
        selector.image = frame.copy()
        cv2.namedWindow(selector.window_name)
        cv2.setMouseCallback(selector.window_name, selector.mouse_callback)

        print("Select 4 points in clockwise order")
        cv2.imshow(selector.window_name, frame)

        while len(selector.points) < 4:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(selector.points) == 4:
            distances = []
            for i in range(4):
                p1 = selector.points[i]
                p2 = selector.points[(i+1)%4]
                distance, ok = QInputDialog.getDouble(
                    self,
                    "Enter Distance",
                    f"Enter distance between point {i+1} and point {(i+1)%4 + 1} in meters:",
                    value=1.0,
                    min=0.1,
                    max=100.0,
                    decimals=2
                )
                if not ok:
                    return None, None
                distances.append(distance)
            points_world = [[0,0]] 
            current_x, current_y = 0, 0
            for i in range(4):
                if i == 0:  
                    current_x += distances[0]
                    points_world.append([current_x, current_y])
                elif i == 1:  
                    current_y += distances[1]
                    points_world.append([current_x, current_y])
                elif i == 2:  
                    current_x -= distances[2]
                    points_world.append([current_x, current_y])

            return selector.points, points_world

        return None, None
    
    def calculate_default_areas(self):
        trajectory_file = os.path.join(self.settings['output_folder'], 'trajectories.csv')
        df = pd.read_csv(trajectory_file)
        margin = 0.1
        minx, miny = df[['x', 'y']].min().values
        maxx, maxy = df[['x', 'y']].max().values
        minx -= margin
        miny -= margin
        maxx += margin
        maxy += margin
        self.settings["walkable_area"] = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        width = (maxx - minx) * 0.25
        height = (maxy - miny) * 0.25
        self.settings["measurement_area"] = [(center_x - width / 2, center_y - height / 2), (center_x + width / 2, center_y - height / 2), (center_x + width / 2, center_y + height / 2), (center_x - width / 2, center_y + height / 2)]

    def show_plots(self):
        trajectory_file = os.path.join(self.settings['output_folder'], 'trajectories.csv')
        df = pd.read_csv(trajectory_file)
        frame_rate = self.settings["frame_rate"]
        traj = TrajectoryData(data=df, frame_rate=frame_rate)
        fig1, ax1 = plt.subplots()
        plot_trajectories(traj=traj, ax=ax1).set_aspect("equal")
        ax1.set_title("Pedestrian Trajectories")
        ax1.set_xlabel("X Position (m)")
        ax1.set_ylabel("Y Position (m)")
        trajectory_plot_path = os.path.join(self.settings['output_folder'], 'trajectories_plot.png')
        fig1.savefig(trajectory_plot_path)

        polygon = self.settings["walkable_area"]
        walkable_area = WalkableArea(polygon=polygon)
        measurement_area_polygon = self.settings["measurement_area"]
        measurement_area = MeasurementArea(measurement_area_polygon)
        classic_density = compute_classic_density(traj_data=traj, measurement_area=measurement_area)
        individual = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area)
        density_voronoi, intersecting = compute_voronoi_density(individual_voronoi_data=individual, measurement_area=measurement_area)
        individual_cutoff = compute_individual_voronoi_polygons(traj_data=traj, walkable_area=walkable_area, cut_off=Cutoff(radius=12.0, quad_segments=1))
        density_voronoi_cutoff, intersecting_cutoff = compute_voronoi_density(individual_voronoi_data=individual_cutoff, measurement_area=measurement_area)
        fig2, ax2 = plt.subplots()
        ax2.set_title("Comparison of different density methods")
        ax2.plot(classic_density.reset_index().frame, classic_density.values, label="classic", color=PEDPY_BLUE)
        ax2.plot(density_voronoi.reset_index().frame, density_voronoi, label="voronoi", color=PEDPY_ORANGE)
        ax2.plot(density_voronoi_cutoff.reset_index().frame, density_voronoi_cutoff, label="voronoi with cutoff", color=PEDPY_GREY)
        ax2.set_xlabel("frame")
        ax2.set_ylabel("$\\rho$ / 1/$m^2$")
        ax2.grid()
        ax2.legend()
        density_plot_path = os.path.join(self.settings['output_folder'], 'density_plot.png')
        fig2.savefig(density_plot_path)

        frame_step = self.settings["frame_step"]
        individual_speed_single_sided = compute_individual_speed(traj_data=traj, frame_step=frame_step, compute_velocity=True, speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED)
        mean_speed = compute_mean_speed_per_frame(traj_data=traj, measurement_area=measurement_area, individual_speed=individual_speed_single_sided)
        individual_speed_direction = compute_individual_speed(traj_data=traj, frame_step=5, movement_direction=np.array([0, -1]), compute_velocity=True, speed_calculation=SpeedCalculation.BORDER_SINGLE_SIDED)
        mean_speed_direction = compute_mean_speed_per_frame(traj_data=traj, measurement_area=measurement_area, individual_speed=individual_speed_direction)
        voronoi_speed = compute_voronoi_speed(traj_data=traj, individual_voronoi_intersection=intersecting, individual_speed=individual_speed_single_sided, measurement_area=measurement_area)
        voronoi_speed_direction = compute_voronoi_speed(traj_data=traj, individual_voronoi_intersection=intersecting, individual_speed=individual_speed_direction, measurement_area=measurement_area)
        fig3, ax3 = plt.subplots()
        ax3.set_title("Comparison of different speed methods")
        ax3.plot(voronoi_speed.reset_index().frame, voronoi_speed, label="Voronoi", color=PEDPY_ORANGE)
        ax3.plot(voronoi_speed_direction.reset_index().frame, voronoi_speed_direction, label="Voronoi direction", color=PEDPY_GREY)
        ax3.plot(mean_speed.reset_index().frame, mean_speed, label="classic", color=PEDPY_BLUE)
        ax3.plot(mean_speed_direction.reset_index().frame, mean_speed_direction, label="classic direction", color=PEDPY_RED)
        ax3.set_xlabel("frame")
        ax3.set_ylabel("v / m/s")
        ax3.legend()
        ax3.grid()
        speed_plot_path = os.path.join(self.settings['output_folder'], 'speed_plot.png')
        fig3.savefig(speed_plot_path)

        plot_window = PlotWindow(density_plot_path, speed_plot_path, trajectory_plot_path, self)
        plot_window.show()

class PointSelector:
    def __init__(self, window_name):
        self.points = []
        self.window_name = window_name
        self.image = None
        self.distances = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.image, str(len(self.points)), (x+10, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if len(self.points) > 1:
                cv2.line(self.image, tuple(self.points[-2]), tuple(self.points[-1]), (0,255,0), 2)
            if len(self.points) == 4:
                cv2.line(self.image, tuple(self.points[-1]), tuple(self.points[0]), (0,255,0), 2)
            cv2.imshow(self.window_name, self.image)

class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2

class Track:
    def __init__(self, box, track_id):
        self.id = track_id
        self.state = TrackState.NEW
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  
        x,y = (box[0] + box[2])/2, (box[1] + box[3])/2
        self.kf.x = np.array([[x], [y], [0], [0]])
        self.kf.F = np.array([[1,0,1,0], 
                             [0,1,0,1],
                             [0,0,1,0],
                             [0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],
                             [0,1,0,0]])
        self.kf.P *= 10
        self.kf.R *= 1
        self.kf.Q *= 0.1

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def update(self, measurement):
        self.kf.update(measurement)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.NEW and self.hits >= 3:
            self.state = TrackState.TRACKED

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.TRACKED and self.time_since_update > 10:
            self.state = TrackState.LOST

class CrowdDensityEstimation:
    def __init__(self, model_path='yolo11x.pt', conf_threshold=0.15, iou_threshold=0.45):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)
        self.density_history = []
        self.track_colors = {}
        self.unique_persons = set()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None

        self.tracks = []
        self.next_id = 0

    def get_color(self, track_id):
        if track_id not in self.track_colors:
            self.track_colors[track_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return self.track_colors[track_id]

    def extract_tracks(self, frame):
        # Resize frame to dimensions divisible by 32 (e.g., 640x640)
        height, width, _ = frame.shape
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        resized_frame = cv2.resize(frame, (new_width, new_height))
        tensor_frame = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor_frame /= 255.0  # Normalize to [0, 1]

        results = self.model.track(
            tensor_frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[0],  # Person class
            tracker="bytetrack.yaml",
            augment=True,
            verbose=False
        )
        return results, resized_frame
    
    def draw_detections(self, frame, results):
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            self.unique_persons.update(track_ids)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                color = self.get_color(track_id)
                cv2.rectangle(frame,
                             (int(x1), int(y1)),
                             (int(x2), int(y2)),
                             color, 2)
                cv2.putText(frame,
                           f"ID: {track_id}",
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           color,
                           2)
                if track_id in self.track_history:
                    points = self.track_history[track_id]
                    for i in range(1, len(points)):
                        if points[i - 1] is None or points[i] is None:
                            continue
                        # Convert world coordinates back to image coordinates
                        if self.homography_matrix is not None:
                            H_inv = np.linalg.inv(self.homography_matrix)
                            pt1 = self.world_to_image(points[i - 1], H_inv)
                            pt2 = self.world_to_image(points[i], H_inv)
                        else:
                            pt1 = points[i - 1]
                            pt2 = points[i]
                        
                        cv2.line(frame, 
                                (int(pt1[0]), int(pt1[1])), 
                                (int(pt2[0]), int(pt2[1])), 
                                color, 2)

            cv2.putText(frame,
                       f"Total Unique Persons: {len(self.unique_persons)}",
                       (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)

        return frame

    def process_frame(self, frame, frame_number):
        results, resized_frame = self.extract_tracks(frame)
        processed_frame = self.draw_detections(resized_frame, results)
        current_count = len(results[0].boxes.id) if results[0].boxes.id is not None else 0

        cv2.putText(processed_frame,
                    f"Current Count: {current_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        tracking_info = {
            'frame_number': frame_number,
            'current_count': current_count,
            'unique_count': len(self.unique_persons)
        }
        self.update_trajectories(results, frame_number)
        return processed_frame, tracking_info

    def display_output(self, frame, tracking_info):
        cv2.putText(frame,
                    f"Current Count: {tracking_info['current_count']}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)
        cv2.imshow('Crowd Tracking', frame)
        return cv2.waitKey(1) & 0xFF

    def update_trajectories(self, results, frame_number):
        if results and len(results) > 0:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                world_center = self.transform_point(center)

                if len(self.track_history[track_id]) >= 2:
                    prev = self.track_history[track_id][-1]
                    smoothed_x = (world_center[0] + prev[0]) / 2
                    smoothed_y = (world_center[1] + prev[1]) / 2
                    world_center = (smoothed_x, smoothed_y)

                self.track_history[track_id].append(world_center)

                if len(self.track_history[track_id]) > 45:
                    self.track_history[track_id].pop(0)

    def save_trajectories(self, output_path='trajectories.csv'):
        trajectory_data = []
        for person_id, positions in self.track_history.items():
            for frame_idx, pos in enumerate(positions):
                trajectory_data.append({
                    'id': person_id,
                    'frame': frame_idx,
                    'x': pos[0],
                    'y': pos[1]
                })

        try:
            if trajectory_data:
                df = pd.DataFrame(trajectory_data)
                df.to_csv(output_path, index=False, float_format='%.6f')
                print(f"Trajectories saved to {output_path}")
            else:
                print("No trajectory data to save")
        except PermissionError:
            temp_path = os.path.join(os.environ.get('TEMP'), 'trajectories.csv')
            print(f"Permission denied, saving to temp directory: {temp_path}")
            df = pd.DataFrame(trajectory_data)
            df.to_csv(temp_path, index=False)
        except Exception as e:
            print(f"Error saving trajectories: {e}")

    def set_camera_calibration(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_homography_matrix(self, points_image, points_world):
        """Calculate homography matrix from corresponding points"""
        assert len(points_image) >= 4, "Need at least 4 point correspondences"
        points_image = np.float32(points_image)
        points_world = np.float32(points_world)
        self.homography_matrix, _ = cv2.findHomography(points_image, points_world)

    def transform_point(self, point):
        """Transform image point to world coordinates using homography matrix"""
        if self.homography_matrix is None:
            return point
        point_homo = np.array([point[0], point[1], 1.0]).reshape(3, 1)

        transformed = self.homography_matrix @ point_homo

        transformed = transformed / transformed[2]

        x_world = float(transformed[0][0])
        y_world = float(transformed[1][0])

        return (x_world, y_world)

    def world_to_image(self, point, H_inv):
        """Convert world coordinates back to image coordinates"""
        point_homo = np.array([point[0], point[1], 1.0]).reshape(3, 1)
        transformed = H_inv @ point_homo
        transformed = transformed / transformed[2]
        return (transformed[0][0], transformed[1][0])

    def update_tracks(self, detections):
        for track in self.tracks:
            track.predict()

        if len(detections) > 0 and len(self.tracks) > 0:
            det_centers = np.array([
                [(d[0] + d[2])/2, (d[1] + d[3])/2] 
                for d in detections
            ])
            track_centers = np.array([
                t.kf.x[:2].flatten() 
                for t in self.tracks
            ])
            
            cost_matrix = np.linalg.norm(
                det_centers[:, np.newaxis] - track_centers, 
                axis=2
            )
            matched, unmatched_dets, unmatched_tracks = self._hungarian_match(cost_matrix)
            for d_idx, t_idx in matched:
                self.tracks[t_idx].update(det_centers[d_idx])

            for t_idx in unmatched_tracks:
                self.tracks[t_idx].mark_missed()

            for d_idx in unmatched_dets:
                if d_idx < len(detections):  
                    self._initiate_track(detections[d_idx])
        
        elif len(detections) > 0:
            for det in detections:
                self._initiate_track(det)

        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]

    def _hungarian_match(self, cost_matrix):
        """
        Perform Hungarian matching between tracks and detections
        """
        
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
            
        # Use Hungarian algorithm
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matched = []
        unmatched_tracks = []
        unmatched_detections = []
        
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] > 30.0:  # Maximum distance threshold
                unmatched_tracks.append(r)
                unmatched_detections.append(c)
            else:
                matched.append((r, c))

        for i in range(cost_matrix.shape[0]):
            if i not in row_idx:
                unmatched_tracks.append(i)
                
        for i in range(cost_matrix.shape[1]):
            if i not in col_idx:
                unmatched_detections.append(i)
                
        return matched, unmatched_detections, unmatched_tracks

    def _initiate_track(self, detection):
        """
        Initialize a new track from detection
        """
        self.tracks.append(Track(detection, self.next_id))
        self.next_id += 1

class PlotWindow(QMainWindow):
    def __init__(self, density_plot_path, speed_plot_path, trajectory_plot_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Interpretations")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.density_tab = QWidget()
        self.speed_tab = QWidget()
        self.trajectory_tab = QWidget()

        self.tab_widget.addTab(self.density_tab, "Density Plot")
        self.tab_widget.addTab(self.speed_tab, "Speed Plot")
        self.tab_widget.addTab(self.trajectory_tab, "Trajectory Plot")

        # Set layouts for tabs
        self.density_layout = QVBoxLayout(self.density_tab)
        self.speed_layout = QVBoxLayout(self.speed_tab)
        self.trajectory_layout = QVBoxLayout(self.trajectory_tab)

        # Add plot images
        self.density_image = QLabel(self)
        self.density_image.setPixmap(QPixmap(density_plot_path))
        self.density_layout.addWidget(self.density_image)

        self.speed_image = QLabel(self)
        self.speed_image.setPixmap(QPixmap(speed_plot_path))
        self.speed_layout.addWidget(self.speed_image)

        self.trajectory_image = QLabel(self)
        self.trajectory_image.setPixmap(QPixmap(trajectory_plot_path))
        self.trajectory_layout.addWidget(self.trajectory_image)

        # Add interpretation text
        self.density_text = QTextEdit(self)
        self.density_text.setReadOnly(True)
        self.density_layout.addWidget(self.density_text)

        self.speed_text = QTextEdit(self)
        self.speed_text.setReadOnly(True)
        self.speed_layout.addWidget(self.speed_text)

        # Generate interpretations
        self.generate_interpretation(density_plot_path, self.density_text, "density")
        self.generate_interpretation(speed_plot_path, self.speed_text, "speed")

    def generate_interpretation(self, image_path, text_widget, plot_type):
        base64_image = self.encode_image(image_path)
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api_key)

        if plot_type == "density":
            content = """
                You are a data scientist specializing in pedestrian and crowd mobility patterns. Your task is to analyze and interpret the attached pedestrian density plot, comparing the three density estimation methods (classic, Voronoi, and Voronoi with cutoff). Discuss their differences, and trends over frames in a scientific manner.
            """
        else:
            content = """
                You are a data scientist specializing in pedestrian and crowd mobility patterns. Your task is to analyze and interpret the attached pedestrian speed plot, comparing the four speed estimation methods (classic, classic direction, Voronoi, and Voronoi direction). Discuss their differences, and trends over frames in a scientific manner.
            """

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
        )
        html_content = markdown.markdown(chat_completion.choices[0].message.content)
        text_widget.setHtml(html_content)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
