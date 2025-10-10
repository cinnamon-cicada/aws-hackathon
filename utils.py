# Generate sample building data for Nashville
import random
import numpy as np
import cv2
from PIL import Image
import os
import json
from datetime import datetime

def findImageLocations(image_path, base_color):
    """Find locations of colored patches in the background image"""
    import cv2
    import numpy as np
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Define color ranges for each base color
    color_ranges = {
        "#d32f2f": {  # Red
            "lower": np.array([200, 0, 0]),
            "upper": np.array([255, 100, 100])
        },
        "#f57c00": {  # Orange
            "lower": np.array([200, 100, 0]),
            "upper": np.array([255, 200, 100])
        },
        "#fbc02d": {  # Yellow
            "lower": np.array([200, 200, 0]),
            "upper": np.array([255, 255, 150])
        },
        "#388e3c": {  # Green
            "lower": np.array([0, 150, 0]),
            "upper": np.array([100, 255, 100])
        }
    }
    
    if base_color not in color_ranges:
        return []
    
    # Create mask for the color
    mask = cv2.inRange(img_rgb, color_ranges[base_color]["lower"], color_ranges[base_color]["upper"])
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    locations = []
    center_lat, center_lon = get_center_coordinates()
    
    # Convert image coordinates to geographic coordinates
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Filter small areas
            # Get center of contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                img_height, img_width = img.shape[:2]
                norm_x = cx / img_width
                norm_y = cy / img_height
                lon = center_lon + (norm_x * (0.067 + 0.083) - 0.067)
                lat = center_lat - (norm_y * (0.067 + 0.083) - 0.067)
                
                locations.append({
                    "lat": lat,
                    "lon": lon,
                    "name": f"{base_color} Zone {i+1}",
                    "color": base_color
                })
    
    return locations[:15]  # Limit to 15 dots per color

def generate_heatmap_dots():
    """Generate dots corresponding to colored patches on the background heatmap"""
    dots = []
    
    # Define colors and their urgency levels
    colors = {
        "#d32f2f": (95, 5),    # Red: 95-100
        "#f57c00": (75, 15),   # Orange: 75-90
        "#fbc02d": (55, 15),   # Yellow: 55-70
        "#388e3c": (25, 20)    # Green: 25-45
    }
    
    # Find locations for each color
    for color, (base_urgency, variance) in colors.items():
        locations = findImageLocations("assets/background.png", color)
        
        for location in locations:
            location["urgency"] = base_urgency + random.uniform(0, variance)
            dots.append(location)
    
    return dots

def get_center_coordinates():
    """Get the center coordinates for the current area"""
    return 36.0331, -86.7828  # Brentwood center

def generate_nashville_buildings():
    """Generate sample building locations across Brentwood"""
    buildings = []
    # Brentwood bounds approximately
    lat_min, lat_max = 35.95, 36.10
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
        center_lat = 36.0331  # Brentwood center
        center_lon = -86.7828
        
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
