# Generate sample building data for Nashville
import random
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime

def generate_nashville_buildings():
    """Generate sample building locations across Nashville"""
    buildings = []
    # Nashville bounds approximately
    lat_min, lat_max = 36.10, 36.22
    lon_min, lon_max = -86.85, -86.70
    
    # Generate grid of buildings
    for i in range(50):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        
        buildings.append({
            "lat": lat,
            "lon": lon,
            "name": f"Building {i+1}"
        })
    
    return buildings

def makeJSON(heatmap_points, input_image_path, output_path="detections_output.json"):
    """
    Create a JSON file with detection data in the specified format.
    
    Args:
        heatmap_points: List of detection points with lat, lon, weight
        input_image_path: Path to the input image to get dimensions
        output_path: Path to save the output JSON
    
    Returns:
        Path to the generated JSON file
    """
    try:
        # Load input image to get dimensions
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            print(f"[MAKEJSON] Error: Could not load input image {input_image_path}")
            return None
            
        height, width = input_img.shape[:2]
        print(f"[MAKEJSON] Input image dimensions: {width}x{height}")
        
        # Use input image bounds (same as in detection function)
        center_lat = 36.1627  # Nashville center
        center_lon = -86.7816
        
        image_bounds = {
            'north': center_lat + 0.01,
            'south': center_lat - 0.01,
            'east': center_lon + 0.01,
            'west': center_lon - 0.01
        }
        
        # Convert heatmap points to detection format
        detections = []
        for i, point in enumerate(heatmap_points):
            lat, lon, weight = point['lat'], point['lon'], point['weight']
            
            # Convert lat/lon to pixel coordinates (same as detection function)
            pixel_x = int((lon - image_bounds['west']) / (image_bounds['east'] - image_bounds['west']) * width)
            pixel_y = int((image_bounds['north'] - lat) / (image_bounds['north'] - image_bounds['south']) * height)
            
            # Ensure within bounds
            pixel_x = max(0, min(width-1, pixel_x))
            pixel_y = max(0, min(height-1, pixel_y))
            
            detections.append({
                "id": i + 1,
                "x": pixel_x,
                "y": pixel_y,
                "confidence": round(weight, 2)
            })
        
        # Create JSON structure
        json_data = {
            "detections": detections,
            "metadata": {
                "image_width": width,
                "image_height": height,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }
        
        # Save JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"[MAKEJSON] Detection data saved to: {output_path}")
        print(f"[MAKEJSON] Processed {len(heatmap_points)} detection points")
        
        return output_path
        
    except Exception as e:
        print(f"[MAKEJSON] Error creating JSON: {e}")
        return None
