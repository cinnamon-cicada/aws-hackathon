# Generate sample building data for Nashville
import random
import numpy as np
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter

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

def makePng(heatmap_points, input_image_path, output_path="heatmap_output.png"):
    """
    Create a PNG heatmap using matplotlib with the same dimensions as the input image.
    
    Args:
        heatmap_points: List of detection points with lat, lon, weight
        input_image_path: Path to the input image to match dimensions
        output_path: Path to save the output PNG
    
    Returns:
        Path to the generated PNG file
    """
    try:
        # Load input image to get dimensions
        input_img = cv2.imread(input_image_path)
        if input_img is None:
            print(f"[MAKEPNG] Error: Could not load input image {input_image_path}")
            return None
            
        height, width = input_img.shape[:2]
        print(f"[MAKEPNG] Input image dimensions: {width}x{height}")
        
        if not heatmap_points:
            print("[MAKEPNG] No heatmap points provided")
            # Save empty canvas
            cv2.imwrite(output_path, np.zeros((height, width, 3), dtype=np.uint8))
            return output_path
        
        # Use input image bounds (same as in detection function)
        center_lat = 36.1627  # Nashville center
        center_lon = -86.7816
        
        image_bounds = {
            'north': center_lat + 0.01,
            'south': center_lat - 0.01,
            'east': center_lon + 0.01,
            'west': center_lon - 0.01
        }
        
        # Create heatmap data
        heatmap_data = np.zeros((height, width))
        
        # Add each point to the heatmap with Gaussian distribution
        for i, point in enumerate(heatmap_points):
            lat, lon, weight = point['lat'], point['lon'], point['weight']
            
            # Convert lat/lon to pixel coordinates (same as detection function)
            pixel_x = int((lon - image_bounds['west']) / (image_bounds['east'] - image_bounds['west']) * width)
            pixel_y = int((image_bounds['north'] - lat) / (image_bounds['north'] - image_bounds['south']) * height)
            
            # Ensure within bounds
            pixel_x = max(0, min(width-1, pixel_x))
            pixel_y = max(0, min(height-1, pixel_y))
            
            # Add heat with small radius (1/4 original size)
            radius = max(1, int(weight * 5))
            cv2.circle(heatmap_data, (pixel_x, pixel_y), radius, weight, -1)
        
        # Apply Gaussian blur for smooth gradients
        heatmap_data = gaussian_filter(heatmap_data, sigma=3)
        
        # Normalize to 0-1 range
        if heatmap_data.max() > 0:
            heatmap_data = heatmap_data / heatmap_data.max()
        
        # Create rainbow colormap
        colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
        n_bins = 256
        cmap = mcolors.LinearSegmentedColormap.from_list('rainbow', colors, N=n_bins)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot heatmap using image coordinates (0 to width, 0 to height)
        im = ax.imshow(heatmap_data, extent=[0, width, 0, height], 
                      cmap=cmap, origin='lower', alpha=0.8)
        
        # Remove axes and margins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Save as PNG
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
        
        print(f"[MAKEPNG] Heatmap saved to: {output_path}")
        print(f"[MAKEPNG] Processed {len(heatmap_points)} detection points")
        
        return output_path
        
    except Exception as e:
        print(f"[MAKEPNG] Error creating heatmap: {e}")
        return None
