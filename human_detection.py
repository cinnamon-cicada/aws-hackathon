# human_detection.py
import asyncio
import cv2  # For video feed processing
import numpy as np
import random
from alert_system import alert_system, trigger_100_level_alert

# Mock AI model for demonstration
class MockAIModel:
    def __init__(self):
        self.detection_probability = 0.1  # 10% chance of detecting humans
    
    def detect_humans(self, frame):
        """Mock human detection - returns True/False"""
        return random.random() < self.detection_probability

# Global model instance
model = MockAIModel()

async def monitor_drone_feed():
    """
    Continuously monitors a drone feed asynchronously.
    Sends alerts when humans are detected.
    """
    # Mock drone coordinates (in practice, these would come from GPS/telemetry)
    drone_lat = 36.1627  # Nashville downtown
    drone_lon = -86.7816
    
    # Update drone coordinates in alert system
    alert_system.update_drone_coordinates(drone_lat, drone_lon, altitude=100)
    
    while True:
        try:
            # Mock frame processing (in practice, would read from actual drone feed)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
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

async def send_alert(alert_data):
    """
    Handles notifications/alerts when humans are detected.
    """
    print(f"ALERT SENT: {alert_data['type']}")
    print(f"Coordinates: {alert_data['coordinates']['lat']:.4f}, {alert_data['coordinates']['lon']:.4f}")
    print(f"Conditions: {', '.join(alert_data['conditions'])}")
    print(f"Time: {alert_data['timestamp']}")
    print("-" * 50)

