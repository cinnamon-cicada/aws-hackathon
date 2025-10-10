# drone_simulator.py
import cv2
import numpy as np
import random


class DroneSimulator:
    def __init__(self, start_lat: float = 36.0331, start_lon: float = -86.7828, altitude: float = 100.0,
                 video_path: str = "assets/human_passby.mp4"):
        self.lat = start_lat
        self.lon = start_lon
        self.alt = altitude
        self.video_path = video_path
        print(f"[DRONE_SIM] Initializing with video path: {video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"[DRONE_SIM] WARNING: Could not open video file {video_path}")
            self.cap = None
            self.frame_count = 0
        else:
            count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.frame_count = count if count > 0 else 0
            print(f"[DRONE_SIM] Video loaded successfully - {self.frame_count} frames available")

    def get_coordinates(self):
        self.lat += random.uniform(-0.0005, 0.0005)
        self.lon += random.uniform(-0.0005, 0.0005)
        self.alt += random.uniform(-0.5, 0.5)
        return self.lat, self.lon, self.alt

    def get_random_frame(self):
        if self.cap is None or self.frame_count <= 0:
            print(f"[DRONE_SIM] Using random noise frame (no video available)")
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        idx = random.randint(0, self.frame_count - 1)
        print(f"[DRONE_SIM] Seeking to frame {idx} of {self.frame_count}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[DRONE_SIM] Failed to read frame {idx}, using random noise")
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"[DRONE_SIM] Successfully read frame {idx} with shape {frame.shape}")
        return frame

    def get_img_frame(self, path):
        """Load an image from file path and return as numpy array (same type as get_random_frame)"""
        frame = cv2.imread(path)
        if frame is None:
            print(f"[DRONE_SIM] Failed to load image: {path}, using random noise")
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"[DRONE_SIM] Successfully loaded image: {path} with shape {frame.shape}")
        return frame

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

 
