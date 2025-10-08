# human_detection.py
import asyncio
import cv2
import numpy as np
import random
import base64
import io
import os
import time
from PIL import Image
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
from alert_system import alert_system, trigger_100_level_alert
from drone_simulator import DroneSimulator
from utils import makeJSON

load_dotenv('aws_credentials.env')

# AWS Rekognition Human Detection
class AWSRekognitionDetector:
    def __init__(self, region_name: str = None):
        if region_name is None:
            region_name = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        try:
            session_token = os.getenv('AWS_SESSION_TOKEN')
            client_kwargs = {
                'service_name': 'rekognition',
                'region_name': region_name,
                'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY')
            }
            if session_token:
                client_kwargs['aws_session_token'] = session_token
            
            self.rekognition = boto3.client(**client_kwargs)
            self.region = region_name
            print(f"[AWS_REKOGNITION] Client initialized for region: {region_name}")
        except NoCredentialsError:
            print("[AWS_REKOGNITION] ERROR: AWS credentials not found!")
            print("[AWS_REKOGNITION] Please update aws_credentials.env with your credentials")
            self.rekognition = None
        except Exception as e:
            print(f"[AWS_REKOGNITION] ERROR: Failed to initialize client: {e}")
            self.rekognition = None
    
    def _numpy_to_bytes(self, frame: np.ndarray) -> bytes:
        """Convert numpy array to bytes for AWS Rekognition"""
        try:
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()
            
            return img_bytes
        except Exception as e:
            print(f"[AWS_REKOGNITION] Error converting frame to bytes: {e}")
            return None
    
    def detect_humans(self, frame: np.ndarray, min_confidence: float = 70.0) -> Tuple[bool, List[Dict], float]:
        """
        Detect humans in frame using AWS Rekognition
        Returns: (detected: bool, detections: list, max_confidence: float)
        """
        if self.rekognition is None:
            print("[AWS_REKOGNITION] Client not initialized, returning no detection")
            return False, [], 0.0
        
        if frame is None or frame.size == 0:
            print("[AWS_REKOGNITION] No frame data provided")
            return False, [], 0.0
        
        try:
            # Convert frame to bytes
            img_bytes = self._numpy_to_bytes(frame)
            if img_bytes is None:
                return False, [], 0.0
            
            print(f"[AWS_REKOGNITION] Analyzing frame ({len(img_bytes)} bytes)")
            
            # Call AWS Rekognition
            response = self.rekognition.detect_labels(
                Image={'Bytes': img_bytes},
                MinConfidence=min_confidence,
                MaxLabels=10
            )
            
            # Process results
            human_detections = []
            max_confidence = 0.0
            
            for label in response.get('Labels', []):
                if label['Name'].lower() in ['person', 'people', 'human']:
                    for instance in label.get('Instances', []):
                        confidence = instance.get('Confidence', 0.0)
                        bbox = instance.get('BoundingBox', {})
                        
                        detection = {
                            'confidence': confidence,
                            'bounding_box': bbox,
                            'label': label['Name']
                        }
                        human_detections.append(detection)
                        max_confidence = max(max_confidence, confidence)
            
            detected = len(human_detections) > 0
            print(f"[AWS_REKOGNITION] Found {len(human_detections)} humans, max confidence: {max_confidence:.2f}%")
            
            return detected, human_detections, max_confidence
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"[AWS_REKOGNITION] AWS Error ({error_code}): {error_message}")
            return False, [], 0.0
            
        except Exception as e:
            print(f"[AWS_REKOGNITION] Unexpected error: {e}")
            return False, [], 0.0

# AWS Rekognition Human Detection Model
class AWSHumanDetectionModel:
    def __init__(self, region_name: str = 'us-east-1', min_confidence: float = 70.0):
        """Initialize AWS Rekognition-based human detection model"""
        self.detector = AWSRekognitionDetector(region_name)
        self.min_confidence = min_confidence
        print(f"[AWS_MODEL] Initialized with min_confidence: {min_confidence}%")
    
    def detect_humans(self, frame: np.ndarray, threshold: float = None) -> Tuple[bool, List[Dict], float]:
        """
        Detect humans using AWS Rekognition
        Returns: (detected: bool, detections: list, max_confidence: float)
        """
        if threshold is None:
            threshold = self.min_confidence
        
        print(f"[AWS_MODEL] Detecting humans with confidence threshold: {threshold}%")
        
        # Use AWS Rekognition detector
        detected, detections, max_confidence = self.detector.detect_humans(frame, threshold)
        
        # Convert confidence to 0-1 scale for compatibility
        confidence_scale = max_confidence / 100.0 if max_confidence > 0 else 0.0
        
        print(f"[AWS_MODEL] Detection result: {detected}, Confidence: {max_confidence:.2f}%, Detections: {len(detections)}")
        
        return detected, detections, confidence_scale
    
# Global model instance - using AWS Rekognition
model = AWSHumanDetectionModel(min_confidence=70.0)

# Persistent simulator instance for single-step detection calls
_sim = DroneSimulator()

async def monitor_drone_feed():
    """
    Continuously monitors a drone feed asynchronously.
    Sends alerts when humans are detected.
    """
    sim = DroneSimulator()
    sim = DroneSimulator()
    
    while True:
        try:
            frame = sim.get_img_frame('assets/test1.png')
            lat, lon, alt = sim.get_coordinates()
            alert_system.update_drone_coordinates(lat, lon, altitude=alt)
            frame = sim.get_img_frame('assets/test1.png')
            lat, lon, alt = sim.get_coordinates()
            alert_system.update_drone_coordinates(lat, lon, altitude=alt)
            
            # Detect humans in frame
            print(f"[DRONE_MONITOR] Processing frame from coordinates: lat={lat:.4f}, lon={lon:.4f}, alt={alt:.2f}")
            humans_detected, boxes, confidence = model.detect_humans(frame)
            print(f"[DRONE_MONITOR] Detection result: {humans_detected}, Confidence: {confidence:.4f}")
            
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

def detect_human_heatmap_points() -> List[Dict[str, float]]:
    """Run one detection step and return all detection points for heatmap.
    Returns a list of dicts: [{ 'lat': float, 'lon': float, 'weight': float }] where weight in [0,1].
    Returns empty list if no detections or error.
    """
    try:
        frame = _sim.get_img_frame('assets/test1.png')
        lat, lon, alt = _sim.get_coordinates()
        # Do not update alert system here; this is a passive collection call
        humans_detected, boxes, confidence = model.detect_humans(frame)
        
        points = []
        if humans_detected and len(boxes) > 0:
            # Get image dimensions for relative positioning
            img_height, img_width = frame.shape[:2]
            
            # Define the area covered by the drone image (adjust these coordinates as needed)
            # This represents the geographic area that the image covers
            image_bounds = {
                'north': lat + 0.01,  # North boundary
                'south': lat - 0.01,  # South boundary  
                'east': lon + 0.01,   # East boundary
                'west': lon - 0.01    # West boundary
            }
            
            # Create a point for each detection box
            for i, box in enumerate(boxes):
                # Get bounding box coordinates from AWS Rekognition
                bbox = box.get('bounding_box', {})
                if bbox:
                    # AWS Rekognition returns normalized coordinates (0-1)
                    left = bbox.get('Left', 0.5)
                    top = bbox.get('Top', 0.5)
                    width = bbox.get('Width', 0.1)
                    height = bbox.get('Height', 0.1)
                    
                    # Calculate center of bounding box
                    center_x = left + width / 2
                    center_y = top + height / 2
                    
                    # Convert image coordinates to geographic coordinates
                    # X coordinate (left-right) maps to longitude
                    geo_lon = image_bounds['west'] + center_x * (image_bounds['east'] - image_bounds['west'])
                    # Y coordinate (top-bottom) maps to latitude (inverted)
                    geo_lat = image_bounds['north'] - center_y * (image_bounds['north'] - image_bounds['south'])
                else:
                    # Fallback to center of image with small offset
                    offset_lat = random.uniform(-0.0005, 0.0005)
                    offset_lon = random.uniform(-0.0005, 0.0005)
                    geo_lat = lat + offset_lat
                    geo_lon = lon + offset_lon
                
                # Use detection confidence as weight, with minimum for visibility
                weight = max(float(box.get('confidence', confidence)) / 100.0, 0.35)
                
                points.append({
                    'lat': float(geo_lat),
                    'lon': float(geo_lon),
                    'weight': weight
                })
        
        # Generate PNG heatmap if we have points
        if points:
            try:
                input_image_path = 'assets/test1.png'
                output_path = f'assets/heatmap_data_{int(time.time())}.json'
                json_path = makeJSON(points, input_image_path, output_path)
                if json_path:
                    print(f"[HEATMAP] PNG heatmap generated: {json_path}")
                else:
                    print("[HEATMAP] Failed to generate PNG heatmap")
            except Exception as e:
                print(f"[HEATMAP] Error generating PNG: {e}")
        
        return points
    except Exception as e:
        print(f"[HEATMAP] Error generating heatmap points: {e}")
        return []

async def send_alert(alert_data):
    """
    Handles notifications/alerts when humans are detected.
    """
    print(f"ALERT SENT: {alert_data['type']}")
    print(f"Coordinates: {alert_data['coordinates']['lat']:.4f}, {alert_data['coordinates']['lon']:.4f}")
    print(f"Conditions: {', '.join(alert_data['conditions'])}")
    print(f"Time: {alert_data['timestamp']}")
    print("-" * 50)

