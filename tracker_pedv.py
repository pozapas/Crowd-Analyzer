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
import time


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
        self.kf.P *= 20
        self.kf.R *= 0.8
        self.kf.Q *= 0.2
        
        self.features = []
        self.last_box = box
        self.class_id = None

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def update(self, measurement):
        self.kf.update(measurement)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.NEW and self.hits >= 2:
            self.state = TrackState.TRACKED

    def mark_missed(self):
        self.time_since_update += 1
        if self.state == TrackState.TRACKED and self.time_since_update > 30:
            self.state = TrackState.LOST
            
    def add_feature(self, feature):
        if len(self.features) >= 10:
            self.features.pop(0)
        self.features.append(feature)
        
    def get_feature(self):
        if not self.features:
            return None
        return np.mean(self.features, axis=0)

class IntersectionAnalyzer:
    def __init__(self, model_path='yolov11x.pt', conf_threshold=0.1, iou_threshold=0.35):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        self.track_history = defaultdict(list)
        self.track_classes = {}
        self.track_colors = {}
        
        self.max_track_age = 120
        self.recovery_iou_threshold = 0.3
        self.recovery_distance_threshold = 100
        self.track_features = {}
        
        self.unique_persons = set()
        self.unique_vehicles = set()
        
        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus", 
            7: "truck"
        }
        
        self.vehicle_classes = [1, 2, 3, 5, 7]  
        self.pedestrian_classes = [0]  
        
        self.zones = {}
        
        self.objects_in_zones = defaultdict(lambda: {'pedestrians': set(), 'vehicles': set()})
        
        self.interactions = []
        self.proximity_threshold = 3.0  
        
        self.track_directions = {}  
        
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
        height, width, _ = frame.shape
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        resized_frame = cv2.resize(frame, (new_width, new_height))

        tensor_frame = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor_frame /= 255.0 

        target_classes = self.pedestrian_classes + self.vehicle_classes
        
        results = self.model.track(
            tensor_frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=target_classes,
            tracker="botsort.yaml",  
            augment=True,
            verbose=False
        )
        return results, resized_frame

    def draw_detections(self, frame, results):
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for track_id, class_id in zip(track_ids, class_ids):
                self.track_classes[track_id] = int(class_id)
                
                if class_id in self.pedestrian_classes:
                    self.unique_persons.add(track_id)
                elif class_id in self.vehicle_classes:
                    self.unique_vehicles.add(track_id)

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                color = self.get_color(track_id)
                
                class_name = self.class_names.get(int(class_id), "unknown")

                cv2.rectangle(frame,
                             (int(x1), int(y1)),
                             (int(x2), int(y2)),
                             color, 2)

                cv2.putText(frame,
                           f"ID:{track_id} {class_name}",
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

            for zone_name, zone_points in self.zones.items():
                if self.homography_matrix is not None:
                    H_inv = np.linalg.inv(self.homography_matrix)
                    image_points = [self.world_to_image(p, H_inv) for p in zone_points]
                else:
                    image_points = zone_points
                
                points = np.array(image_points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 255), 2)
                
                centroid_x = int(sum(p[0] for p in image_points) / len(image_points))
                centroid_y = int(sum(p[1] for p in image_points) / len(image_points))
                cv2.putText(frame, zone_name, (centroid_x, centroid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display counts
            cv2.putText(frame,
                       f"Pedestrians: {len(self.unique_persons)}",
                       (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
                       
            cv2.putText(frame,
                       f"Vehicles: {len(self.unique_vehicles)}",
                       (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)

        return frame

    def process_frame(self, frame, frame_number):
        results, resized_frame = self.extract_tracks(frame)
        processed_frame = self.draw_detections(resized_frame, results)
        
        current_pedestrians = 0
        current_vehicles = 0
        
        if results[0].boxes.id is not None:
            boxes_ids = results[0].boxes.id.cpu().numpy()
            boxes_cls = results[0].boxes.cls.cpu().numpy()
            
            for i, class_id in enumerate(boxes_cls):
                if int(class_id) in self.pedestrian_classes:
                    current_pedestrians += 1
                elif int(class_id) in self.vehicle_classes:
                    current_vehicles += 1

        current_interactions = sum(1 for interaction in self.interactions if interaction['frame'] == frame_number)

        tracking_info = {
            'frame_number': frame_number,
            'current_pedestrians': current_pedestrians,
            'current_vehicles': current_vehicles,
            'unique_pedestrians': len(self.unique_persons),
            'unique_vehicles': len(self.unique_vehicles),
            'current_interactions': current_interactions
        }
        
        self.update_trajectories(results, frame_number)
        return processed_frame, tracking_info

    def display_output(self, frame, tracking_info):
        frame_count_y = 120
        for key, value in tracking_info.items():
            if key == 'frame_number':
                continue 
            cv2.putText(frame,
                       f"{key}: {value}",
                       (20, frame_count_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
            frame_count_y += 40
            
        cv2.imshow('Intersection Analysis', frame)
        return cv2.waitKey(1) & 0xFF

    def update_trajectories(self, results, frame_number):
        if results and len(results) > 0 and results[0].boxes.id is not None:
            for i, box in enumerate(results[0].boxes):
                if box.id is None:
                    continue
                    
                track_id = int(box.id[0])
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                self.track_classes[track_id] = class_id
                
                world_center = self.transform_point(center)

                if len(self.track_history[track_id]) >= 2:
                    prev = self.track_history[track_id][-1]
                    smoothed_x = (world_center[0] + prev[0]) / 2
                    smoothed_y = (world_center[1] + prev[1]) / 2
                    world_center = (smoothed_x, smoothed_y)

                self.track_history[track_id].append(world_center)

                if len(self.track_history[track_id]) > 60:  
                    self.track_history[track_id].pop(0)
                    
                for zone_name, zone_points in self.zones.items():
                    if self.point_in_polygon(world_center, zone_points):
                        if class_id in self.pedestrian_classes:
                            self.objects_in_zones[zone_name]['pedestrians'].add(track_id)
                        elif class_id in self.vehicle_classes:
                            self.objects_in_zones[zone_name]['vehicles'].add(track_id)
            
            self.analyze_interactions(frame_number)

    def analyze_interactions(self, frame_number):
        """Analyze potential interactions between pedestrians and vehicles"""
        pedestrian_positions = {}
        vehicle_positions = {}
        
        for track_id, positions in self.track_history.items():
            if not positions:
                continue
                
            class_id = self.track_classes.get(track_id, -1)
            current_pos = positions[-1] 
            
            if class_id in self.pedestrian_classes:
                pedestrian_positions[track_id] = current_pos
            elif class_id in self.vehicle_classes:
                vehicle_positions[track_id] = current_pos

        for ped_id, ped_pos in pedestrian_positions.items():
            for veh_id, veh_pos in vehicle_positions.items():

                distance = np.sqrt((ped_pos[0] - veh_pos[0]) ** 2 + 
                                 (ped_pos[1] - veh_pos[1]) ** 2)
                
                if distance < self.proximity_threshold:
                    veh_class = self.class_names.get(self.track_classes.get(veh_id, -1), "unknown")
                    
                    ped_direction, ped_speed = self.calculate_direction_and_speed(ped_id)
                    veh_direction, veh_speed = self.calculate_direction_and_speed(veh_id)
                    
                    converging = False
                    if ped_direction is not None and veh_direction is not None:

                        angle = np.abs(np.degrees(np.arctan2(
                            np.cross(ped_direction, veh_direction),
                            np.dot(ped_direction, veh_direction)
                        )))
                        
                        converging = 30 < angle < 150
                    
                    interaction_zones = []
                    for zone_name, zone_points in self.zones.items():
                        if (self.point_in_polygon(ped_pos, zone_points) and 
                            self.point_in_polygon(veh_pos, zone_points)):
                            interaction_zones.append(zone_name)
                    
                    self.interactions.append({
                        'frame': frame_number,
                        'pedestrian_id': ped_id,
                        'vehicle_id': veh_id,
                        'vehicle_type': veh_class,
                        'distance': distance,
                        'pedestrian_pos': ped_pos,
                        'vehicle_pos': veh_pos,
                        'ped_speed': ped_speed,
                        'veh_speed': veh_speed,
                        'converging': converging,
                        'zones': interaction_zones
                    })
    
    def calculate_direction_and_speed(self, track_id, window_size=5):
        """Calculate movement direction and speed for a track"""
        positions = self.track_history.get(track_id, [])
        
        if len(positions) < window_size:
            return None, None
        
        recent_positions = positions[-window_size:]
        
        start_point = np.array(recent_positions[0])
        end_point = np.array(recent_positions[-1])
        displacement = end_point - start_point
        
        distance = np.linalg.norm(displacement)
        
        if distance > 0:
            direction = displacement / distance    
            # Assuming positions are updated every frame, and we know fps
            # speed = distance / (window_size / fps)
            # Since we don't have fps here, just return the distance as a relative speed
            speed = distance
            
            return direction, speed
        else:
            return np.array([0, 0]), 0

    def define_zone(self, name, points):
        """Define a zone for analysis (e.g., crosswalk, intersection area)"""
        self.zones[name] = points
        print(f"Zone '{name}' defined with {len(points)} points")
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def save_trajectories(self, output_path='trajectories.csv'):
        trajectory_data = []
        for track_id, positions in self.track_history.items():
            class_id = self.track_classes.get(track_id, -1)
            class_name = self.class_names.get(class_id, "unknown")
            
            for frame_idx, pos in enumerate(positions):
                trajectory_data.append({
                    'track_id': track_id,
                    'class_id': class_id,
                    'class_name': class_name,
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
    
    def save_vehicle_trajectories(self, output_path='vehicle_trajectories.csv'):
        """Save only vehicle trajectories to a separate file"""
        vehicle_data = []
        for track_id, positions in self.track_history.items():
            class_id = self.track_classes.get(track_id, -1)
            
            # Skip if not a vehicle
            if class_id not in self.vehicle_classes:
                continue
                
            class_name = self.class_names.get(class_id, "unknown")
            
            for frame_idx, pos in enumerate(positions):
                vehicle_data.append({
                    'track_id': track_id,
                    'vehicle_type': class_name,
                    'frame': frame_idx,
                    'x': pos[0],
                    'y': pos[1]
                })

        if vehicle_data:
            df = pd.DataFrame(vehicle_data)
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Vehicle trajectories saved to {output_path}")
        else:
            print("No vehicle trajectory data to save")
            
    def save_pedestrian_trajectories(self, output_path='pedestrian_trajectories.csv'):
        """Save pedestrian trajectories to a separate file and merge with vehicle trajectories"""
        pedestrian_data = []
        for track_id, positions in self.track_history.items():
            class_id = self.track_classes.get(track_id, -1)
            
            # Skip if not a pedestrian
            if class_id not in self.pedestrian_classes:
                continue
                
            for frame_idx, pos in enumerate(positions):
                pedestrian_data.append({
                    'track_id': track_id,
                    'type': 'pedestrian',
                    'frame': frame_idx,
                    'x': pos[0],
                    'y': pos[1],
                    'timestamp': frame_idx / 30.0  # Assuming 30 fps, adjust if different
                })

        if pedestrian_data:
            ped_df = pd.DataFrame(pedestrian_data)
            ped_df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Pedestrian trajectories saved to {output_path}")
        
            veh_output_path = 'vehicle_trajectories.csv'
            self.save_vehicle_trajectories(veh_output_path)
        
            try:
                veh_df = pd.read_csv(veh_output_path)
                veh_df = veh_df.rename(columns={'vehicle_type': 'type'})
                
                merged_df = pd.concat([ped_df, veh_df], ignore_index=True)
                
                merged_df = merged_df.sort_values(['frame', 'track_id'])
                
                merged_output = 'merged_trajectories.csv'
                merged_df.to_csv(merged_output, index=False, float_format='%.6f')
                print(f"Merged trajectories saved to {merged_output}")
            except Exception as e:
                print(f"Error merging trajectories: {str(e)}")
        else:
            print("No pedestrian trajectory data to save")

    def save_interactions(self, output_path='interactions.csv'):
        """Save recorded interactions to a CSV file"""
        if self.interactions:
            df = pd.DataFrame(self.interactions)
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"Interaction data saved to {output_path}")
        else:
            print("No interaction data to save")

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
        point_homo = np.array([point[0], point[1], 1.0]).reshape(3, 1)

        transformed = self.homography_matrix @ point_homo

        transformed = transformed / transformed[2]

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
        for track in self.tracks:
            track.predict()
            
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
                
                for d_idx, t_idx in matched:
                    self.tracks[t_idx].update(det_centers[d_idx])
                
                for t_idx in unmatched_tracks:
                    self.tracks[t_idx].mark_missed()
                    
                for d_idx in unmatched_dets:
                    self._initiate_track(detections[d_idx])
            else:
                for det in detections:
                    self._initiate_track(det)
                    
        self.tracks = [t for t in self.tracks if t.state != TrackState.LOST]

    def recover_lost_tracks(self, current_tracks):
        lost_tracks = {}
        recovery_threshold = 30  # frames
        
        for track_id, positions in self.track_history.items():
            if track_id not in current_tracks and len(positions) > 5:
                lost_tracks[track_id] = positions[-1]  # Last known position
        
        for new_id in current_tracks:
            if new_id not in self.track_history or len(self.track_history[new_id]) < 3:
                new_pos = self.track_history[new_id][-1] if new_id in self.track_history else None
                if new_pos:
                    for lost_id, lost_pos in lost_tracks.items():
                        distance = np.sqrt((new_pos[0] - lost_pos[0])**2 + (new_pos[1] - lost_pos[1])**2)
                        same_class = self.track_classes.get(new_id) == self.track_classes.get(lost_id)
                        
                        if distance < 5.0 and same_class:  # 5 meters threshold
                            self.track_history[lost_id].extend(self.track_history[new_id])
                            del self.track_history[new_id]
                            self.track_classes[lost_id] = self.track_classes[new_id]
                            break

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
        distances = []
        for i in range(4):
            p1 = selector.points[i]
            p2 = selector.points[(i+1)%4]
            print(f"Enter distance between point {i+1} and point {(i+1)%4 + 1} in meters:")
            dist = float(input())
            distances.append(dist)

        points_world = [[0,0]]  # First point at origin
        current_x, current_y = 0, 0

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

class EnhancedZoneSelector:
    def __init__(self, frame):
        self.original_frame = frame.copy()
        self.drawing_frame = frame.copy()
        self.zones = {}
        self.current_zone_name = ""
        self.current_points = []
        self.is_drawing = False
        self.drawing_complete = False
        
        self.window_name = "Zone Definition Tool"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.zone_colors = {
            "crosswalk": (0, 255, 255),      # Yellow
            "intersection": (0, 165, 255),    # Orange
            "sidewalk": (0, 255, 0),         # Green
            "road": (255, 0, 0),             # Blue
            "custom": (255, 255, 255)        # White
        }
        
        self.show_flashing = True
        self.flash_counter = 0
        
        self.show_instructions()
        
    def show_instructions(self):
        """Show instructions overlay on the image"""
        instructions = [
            "ZONE DEFINITION CONTROLS:",
            "- First, press a number key (1-5) to select zone type",
            "- Then click to place points and define zone boundaries",
            "- Press 'C' to close/complete current zone",
            "- Press 'Z' to undo last point",
            "- Press 'R' to reset current zone",
            "- Press 'D' to delete a zone",
            "- Press 'S' when finished with all zones",
            "",
            "ZONE TYPES (press number to select):",
            "1: Crosswalk",
            "2: Intersection",
            "3: Sidewalk",
            "4: Road",
            "5: Custom Zone (will prompt for name)"
        ]
        
        overlay = self.drawing_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.drawing_frame, 0.3, 0, self.drawing_frame)
        
        for i, line in enumerate(instructions):
            cv2.putText(self.drawing_frame, line, (20, 35 + i*20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                      
        if self.current_zone_name:
            cv2.putText(self.drawing_frame, f"Current zone: {self.current_zone_name}", 
                      (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.show_flashing and self.flash_counter < 30:
            cv2.putText(self.drawing_frame, "PRESS 1-5 TO SELECT ZONE TYPE", 
                      (int(self.drawing_frame.shape[1]/2 - 200), int(self.drawing_frame.shape[0]/2)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for interaction"""
        if self.current_zone_name == "":
            if event == cv2.EVENT_LBUTTONDOWN:
                print("Please select a zone type first (press keys 1-5)")
                temp_frame = self.drawing_frame.copy()
                cv2.putText(temp_frame, "Please select a zone type first (keys 1-5)", 
                          (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(self.window_name, temp_frame)
                cv2.waitKey(1000)  # Show message for 1 second
                self.update_display()
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append((x, y))
            self.is_drawing = True
            self.update_display()
            
    def update_display(self):
        """Update the display with current zone information"""
        self.drawing_frame = self.original_frame.copy()
        
        self.flash_counter = (self.flash_counter + 1) % 60
        
        for name, points in self.zones.items():
            zone_type = name.split('_')[0] if '_' in name else "custom"
            color = self.zone_colors.get(zone_type, self.zone_colors["custom"])
            
            if len(points) > 2:
                cv2.polylines(self.drawing_frame, [np.array(points)], 
                            True, color, 2)
                
                overlay = self.drawing_frame.copy()
                cv2.fillPoly(overlay, [np.array(points)], color)
                cv2.addWeighted(overlay, 0.3, self.drawing_frame, 0.7, 0, self.drawing_frame)

                centroid_x = int(sum(p[0] for p in points) / len(points))
                centroid_y = int(sum(p[1] for p in points) / len(points))
                cv2.putText(self.drawing_frame, name, (centroid_x, centroid_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self.current_points) > 0:
            for i, point in enumerate(self.current_points):
                cv2.circle(self.drawing_frame, point, 5, (0, 255, 255), -1)
                if i > 0:
                    cv2.line(self.drawing_frame, self.current_points[i-1], 
                            point, (0, 255, 255), 2)
            
            if len(self.current_points) >= 3:
                cv2.line(self.drawing_frame, self.current_points[-1], 
                        self.current_points[0], (0, 255, 255), 2, cv2.LINE_AA)
        
        self.show_instructions()
        
        cv2.imshow(self.window_name, self.drawing_frame)
    
    def select_zone_type(self, zone_type):
        """Set the current zone type and generate a name"""
        count = sum(1 for name in self.zones.keys() if name.startswith(zone_type))
        self.current_zone_name = f"{zone_type}_{count+1}"
        self.current_points = []
        self.show_flashing = False 
        self.update_display()
        print(f"Now defining zone: {self.current_zone_name}")
        print("Click on the image to add points to the zone boundary")
    
    def complete_current_zone(self):
        """Complete the current zone if it has enough points"""
        if len(self.current_points) >= 3:
            self.zones[self.current_zone_name] = self.current_points.copy()
            print(f"Zone '{self.current_zone_name}' defined with {len(self.current_points)} points")
            self.current_points = []
            self.current_zone_name = ""
            self.show_flashing = True  
            self.update_display()
            return True
        else:
            print("Need at least 3 points to define a zone")
            return False
            
    def undo_last_point(self):
        """Remove the last point added"""
        if self.current_points:
            self.current_points.pop()
            self.update_display()
            print("Removed last point")
            
    def reset_current_zone(self):
        """Reset the current zone being defined"""
        self.current_points = []
        self.update_display()
        print(f"Reset current zone: {self.current_zone_name}")
        
    def delete_zone(self):
        """Delete a zone by name"""
        if not self.zones:
            print("No zones defined yet")
            return
            
        print("Available zones to delete:")
        for i, name in enumerate(self.zones.keys()):
            print(f"{i+1}: {name}")
            
        choice = input("Enter zone number to delete (or 'c' to cancel): ")
        if choice.lower() == 'c':
            return
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(self.zones):
                zone_name = list(self.zones.keys())[idx]
                del self.zones[zone_name]
                print(f"Deleted zone: {zone_name}")
                self.update_display()
            else:
                print("Invalid zone number")
        except ValueError:
            print("Invalid input")
            
    def run(self):
        """Run the zone definition interface"""
        print("\n=== INTERSECTION ZONE DEFINITION TOOL ===")
        print("First, press a number key (1-5) to select a zone type")
        print("  1: Crosswalk")
        print("  2: Intersection")
        print("  3: Sidewalk")
        print("  4: Road")
        print("  5: Custom zone")
        print("\nThen click on the image to place points and define the zone")
        
        self.update_display()
        
        while True:
            if self.current_zone_name == "" and self.show_flashing:
                self.update_display()
                
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('1'):
                self.select_zone_type("crosswalk")
            elif key == ord('2'):
                self.select_zone_type("intersection")
            elif key == ord('3'):
                self.select_zone_type("sidewalk")
            elif key == ord('4'):
                self.select_zone_type("road")
            elif key == ord('5'):
                custom_name = input("Enter custom zone name: ")
                if custom_name:
                    self.current_zone_name = custom_name
                    self.current_points = []
                    self.show_flashing = False
                    self.update_display()
                    print(f"Now defining zone: {self.current_zone_name}")
                    print("Click on the image to add points to the zone boundary")
                    
            elif key == ord('c'):  
                self.complete_current_zone()
            elif key == ord('z'):  
                self.undo_last_point()
            elif key == ord('r'):  
                self.reset_current_zone()
            elif key == ord('d'):  
                self.delete_zone()
            elif key == ord('s'):  
                if self.current_points and len(self.current_points) >= 3:
                    print(f"Save current in-progress zone '{self.current_zone_name}'? (y/n)")
                    save_choice = input().lower()
                    if save_choice == 'y':
                        self.complete_current_zone()
                
                print(f"Zone definition complete: {len(self.zones)} zones defined")
                break
            elif key == ord('q'):  
                if self.zones:
                    print("Discard all zones and quit? (y/n)")
                    quit_choice = input().lower()
                    if quit_choice == 'y':
                        self.zones = {}
                        break
                else:
                    break
                    
        cv2.destroyAllWindows()
        return self.zones

def user_friendly_zone_definition(frame):
    """Start the enhanced zone definition interface"""
    zone_selector = EnhancedZoneSelector(frame)
    return zone_selector.run()

def main():
    # Start timing
    start_time = time.time()
    
    video_path = input("Enter the path to your video file (e.g., Sample 1.mp4): ")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Failed to read from video file")
        return

    print("\n=== DEFINING ANALYSIS ZONES ===")
    zones = user_friendly_zone_definition(frame.copy())
    
    print("\n=== CALIBRATING COORDINATE SYSTEM ===")
    print("Now select 4 points for coordinate calibration")
    points_image, points_world = select_points_and_distances(frame.copy())
    if points_image is None or points_world is None:
        print("Need 4 points and distances for perspective transform")
        return

    estimator = IntersectionAnalyzer(model_path='yolo11x.pt')
    
    estimator.set_homography_matrix(points_image, points_world)
    
    for zone_name, zone_points in zones.items():
        if estimator.homography_matrix is not None:
            world_points = [estimator.transform_point(p) for p in zone_points]
            estimator.define_zone(zone_name, world_points)
        else:
            estimator.define_zone(zone_name, zone_points)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        'output.mp4',
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, tracking_info = estimator.process_frame(frame, frame_number)

            if processed_frame.shape[:2] != (frame_height, frame_width):
                processed_frame = cv2.resize(processed_frame, (frame_width, frame_height))

            if len(processed_frame.shape) == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

            out.write(processed_frame)

            estimator.display_output(processed_frame, tracking_info)

            frame_number += 1
            if frame_number % 100 == 0:
                current_time = time.time()
                elapsed_minutes = (current_time - start_time) / 60.0
                print(f"Processed {frame_number} frames - Time elapsed: {elapsed_minutes:.2f} minutes")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        end_time = time.time()
        total_minutes = (end_time - start_time) / 60.0
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        estimator.save_trajectories('trajectories.csv')
        estimator.save_vehicle_trajectories('vehicle_trajectories.csv')
        estimator.save_pedestrian_trajectories('pedestrian_trajectories.csv')
        estimator.save_interactions('interactions.csv')

        print(f"\nExecution Summary:")
        print(f"Total frames processed: {frame_number}")
        print(f"Total execution time: {total_minutes:.2f} minutes")
        print(f"Average processing speed: {frame_number / (total_minutes * 60):.2f} frames per second")
        print(f"Video saved to output.mp4")
        print(f"Trajectories saved to CSV files")

if __name__ == "__main__":
    main()