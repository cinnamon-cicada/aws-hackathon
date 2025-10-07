# human_detection.py
import asyncio
import cv2  # For video feed processing
import numpy as np
import random
from alert_system import alert_system, trigger_100_level_alert
from drone_simulator import DroneSimulator

# Mock AI model for demonstration
class MockAIModel:
    def __init__(self):
        self.detection_probability = 0.5
    
    def detect_humans(self, frame: np.ndarray, threshold: float = 0.6):
        """Return (detected: bool, boxes: list, avg_conf: float)."""
        hog = _ensure_hog()
        if frame is None or frame.size == 0:
            return False, [], 0.0
        resized = cv2.resize(frame, (640, 360))
        boxes, weights = hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        if len(weights) == 0:
            return False, [], 0.0
        avg_conf = float(np.mean(weights))
        return avg_conf > threshold, boxes, avg_conf

# Global model instance
model = MockAIModel()

_hog = None

def _ensure_hog():
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor()
        _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return _hog

# Persistent simulator instance for single-step detection calls
_sim = DroneSimulator()

async def monitor_drone_feed():
    """
    Continuously monitors a drone feed asynchronously.
    Sends alerts when humans are detected.
    """
    sim = DroneSimulator()
    
    while True:
        try:
            frame = sim.get_random_frame()
            lat, lon, alt = sim.get_coordinates()
            alert_system.update_drone_coordinates(lat, lon, altitude=alt)
            
            # Detect humans in frame
            humans_detected = model.detect_humans(frame)
            
            if humans_detected:
                # Calculate population density at drone location (mock)
                population_density = random.uniform(3000, 5000)
                
                # Trigger 100-level alert
                alert = trigger_100_level_alert(
                    human_detected=True,
                    population_density=population_density
                )
                
                if alert:
                    await send_alert(alert)
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            print(f"Error in drone monitoring: {e}")
            await asyncio.sleep(5)
    
    try:
        sim.close()
    except Exception:
        pass


def detect_humans_once_and_update() -> bool:
    """Grab a frame from the simulator, update coordinates, and run detection once.
    Returns True if a human is detected (and triggers an alert), else False.
    """
    frame = _sim.get_random_frame()
    lat, lon, alt = _sim.get_coordinates()
    alert_system.update_drone_coordinates(lat, lon, altitude=alt)
    humans_detected, _, _ = model.detect_humans(frame)
    if humans_detected:
        population_density = random.uniform(3000, 5000)
        trigger_100_level_alert(human_detected=True, population_density=population_density)
    return humans_detected

async def send_alert(alert_data):
    """
    Handles notifications/alerts when humans are detected.
    """
    print(f"ALERT SENT: {alert_data['type']}")
    print(f"Coordinates: {alert_data['coordinates']['lat']:.4f}, {alert_data['coordinates']['lon']:.4f}")
    print(f"Conditions: {', '.join(alert_data['conditions'])}")
    print(f"Time: {alert_data['timestamp']}")
    print("-" * 50)

