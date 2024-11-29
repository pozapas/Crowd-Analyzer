import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict
import random
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter

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
        
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # x,y position and velocity
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

        # Add camera calibration parameters
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

        # Convert frame to torch tensor and move to device
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

                # Draw bounding box
                cv2.rectangle(frame,
                             (int(x1), int(y1)),
                             (int(x2), int(y2)),
                             color, 2)

                # Draw ID
                cv2.putText(frame,
                           f"ID: {track_id}",
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           color,
                           2)

                # Draw trajectory
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

                # Transform point to world coordinates
                world_center = self.transform_point(center)

                # Debug print
                print(f"Frame {frame_number}, Track ID {track_id}, Image Center: {center}, World Center: {world_center}")

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

        if trajectory_data:
            df = pd.DataFrame(trajectory_data)
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Trajectories saved to {output_path}")
        else:
            print("No trajectory data to save")

    def set_camera_calibration(self, camera_matrix, dist_coeffs):
        """Set camera calibration parameters"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_homography_matrix(self, points_image, points_world):
        """Calculate homography matrix from corresponding points"""
        assert len(points_image) >= 4, "Need at least 4 point correspondences"
        points_image = np.float32(points_image)
        points_world = np.float32(points_world)
        self.homography_matrix, _ = cv2.findHomography(points_image, points_world)
        #print("Points Image:", points_image)
        #print("Points World:", points_world)
        #print("Homography Matrix:", self.homography_matrix)

    def transform_point(self, point):
        """Transform image point to world coordinates using homography matrix"""
        if self.homography_matrix is None:
            return point

        # Convert to numpy array and reshape for matrix multiplication
        point_homo = np.array([point[0], point[1], 1.0]).reshape(3, 1)

        # Apply homography transformation
        transformed = self.homography_matrix @ point_homo

        # Normalize homogeneous coordinates
        transformed = transformed / transformed[2]

        # Convert to physical coordinates and return as tuple
        x_world = float(transformed[0][0])
        y_world = float(transformed[1][0])

        # Debug print to verify transformation
        #print(f"Image point: {point} -> World point: ({x_world:.2f}, {y_world:.2f})")

        return (x_world, y_world)

    def world_to_image(self, point, H_inv):
        """Convert world coordinates back to image coordinates"""
        point_homo = np.array([point[0], point[1], 1.0]).reshape(3, 1)
        transformed = H_inv @ point_homo
        transformed = transformed / transformed[2]
        return (transformed[0][0], transformed[1][0])

    def update_tracks(self, detections):
        # Predict new locations
        for track in self.tracks:
            track.predict()
            
        # Match detections to existing tracks
        if len(detections) > 0:
            det_centers = np.array([(d[0] + d[2])/2, (d[1] + d[3])/2] 
                                 for d in detections)
            
            if len(self.tracks) > 0:
                track_centers = np.array([t.kf.x[:2].flatten() 
                                        for t in self.tracks])
                cost_matrix = np.linalg.norm(det_centers[:, None] - 
                                           track_centers, axis=2)
                matched, unmatched_dets, unmatched_tracks = \
                    self._hungarian_match(cost_matrix)
                
                # Update matched tracks
                for d_idx, t_idx in matched:
                    self.tracks[t_idx].update(det_centers[d_idx])
                
                # Mark unmatched tracks as missed
                for t_idx in unmatched_tracks:
                    self.tracks[t_idx].mark_missed()
                    
                # Create new tracks
                for d_idx in unmatched_dets:
                    self._initiate_track(detections[d_idx])
            else:
                # Initialize tracks for all detections
                for det in detections:
                    self._initiate_track(det)
                    
        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]

class PointSelector:
    def __init__(self, window_name):
        self.points = []
        self.window_name = window_name
        self.image = None
        self.distances = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append([x, y])
            # Draw point
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)
            # Draw number
            cv2.putText(self.image, str(len(self.points)), (x+10, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if len(self.points) > 1:
                # Draw line between consecutive points
                cv2.line(self.image, tuple(self.points[-2]), tuple(self.points[-1]), (0,255,0), 2)
            if len(self.points) == 4:
                cv2.line(self.image, tuple(self.points[-1]), tuple(self.points[0]), (0,255,0), 2)
            cv2.imshow(self.window_name, self.image)

def select_points_and_distances(frame):
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
        # Get distances between consecutive points
        distances = []
        for i in range(4):
            p1 = selector.points[i]
            p2 = selector.points[(i+1)%4]
            print(f"Enter distance between point {i+1} and point {(i+1)%4 + 1} in meters:")
            dist = float(input())
            distances.append(dist)

        # Calculate world coordinates sequentially
        points_world = [[0,0]]  # First point at origin
        current_x, current_y = 0, 0

        # Calculate remaining points based on distances
        for i in range(4):
            if i == 0:  # Second point - move right
                current_x += distances[0]
                points_world.append([current_x, current_y])
            elif i == 1:  # Third point - move down
                current_y += distances[1]
                points_world.append([current_x, current_y])
            elif i == 2:  # Fourth point - move left
                current_x -= distances[2]
                points_world.append([current_x, current_y])

        return selector.points, points_world

    return None, None

def main():
    cap = cv2.VideoCapture(r"C:\Users\graduate\OneDrive\Pedestrian movies\3.mp4")

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    # Read first frame for point selection
    ret, frame = cap.read()
    if not ret:
        return

    # Get points and distances interactively
    points_image, points_world = select_points_and_distances(frame)
    if points_image is None or points_world is None:
        print("Need 4 points and distances for perspective transform")
        return

    estimator = CrowdDensityEstimation()
    estimator.set_homography_matrix(points_image, points_world)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'output.mp4',
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True
    )

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame and get results
            processed_frame, density_info = estimator.process_frame(frame, frame_number)

            # Ensure processed frame matches output dimensions
            if processed_frame.shape[:2] != (frame_height, frame_width):
                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            # Convert to BGR 
            if len(processed_frame.shape) == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

            # Write frame
            out.write(processed_frame)

            # Display
            estimator.display_output(processed_frame, density_info)

            frame_number += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        estimator.save_trajectories('trajectories.csv')

        print(f"Video saved to output.mp4")
        print(f"Total frames processed: {frame_number}")

if __name__ == "__main__":
    main()
