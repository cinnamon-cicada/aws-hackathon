#!/usr/bin/env python3
"""
aerial_building_heatmap.py

Usage:
    python aerial_building_heatmap.py --input example.jpg --out out_overlay.png --heatmap out_heatmap.png

What it does:
 - Read an aerial image
 - Try to detect building-like regions using color / edge / morphology heuristics
 - Draw building contours / bounding boxes on a copy of the image
 - Create a density heatmap from building centroids (Gaussian blur / KDE-like)
 - Save overlay image (buildings + heatmap) and the plain heatmap image.

Limitations:
 - Heuristic detector works best when buildings have distinct roof colors/edges.
 - For robust results on diverse imagery, use a pretrained object detection / segmentation model.
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# (All global script logic has been moved into the if __name__ == "__main__" block)

def detect_buildings(img_bgr, debug=False):
    """
    Gray-based building detection:
    - Focuses on detecting gray areas which typically represent building roofs
    - Uses HSV color space for better gray detection
    - Multiple gray range detection for different building types
    Returns: list of contours and centroids
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # Convert to HSV for better color-based detection
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Method 1: Direct gray value detection in different ranges
    # Very white buildings (bright roofs, concrete)
    very_white_mask = cv2.inRange(img_gray, 220, 255)
    
    # White buildings
    white_mask = cv2.inRange(img_gray, 200, 255)
    
    # Light gray buildings (concrete, light roofs)
    light_gray_mask = cv2.inRange(img_gray, 150, 200)
    
    # Medium gray buildings
    medium_gray_mask = cv2.inRange(img_gray, 100, 150)
    
    # Dark gray buildings
    dark_gray_mask = cv2.inRange(img_gray, 50, 100)
    
    # Method 2: HSV-based gray detection (low saturation = gray)
    # Low saturation areas (grays, whites, blacks)
    low_sat_mask = cv2.inRange(hsv, (0, 0, 0), (180, 50, 255))
    
    # Method 3: Combined gray detection
    # Combine different gray ranges - prioritize white and light gray
    gray_combined = cv2.bitwise_or(very_white_mask, white_mask)
    gray_combined = cv2.bitwise_or(gray_combined, light_gray_mask)
    gray_combined = cv2.bitwise_or(gray_combined, medium_gray_mask)
    gray_combined = cv2.bitwise_or(gray_combined, dark_gray_mask)
    gray_combined = cv2.bitwise_or(gray_combined, low_sat_mask)
    
    # Method 4: Morphological operations to clean up and connect nearby gray areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))  # Smaller kernel
    gray_cleaned = cv2.morphologyEx(gray_combined, cv2.MORPH_OPEN, kernel, iterations=1)
    gray_cleaned = cv2.morphologyEx(gray_cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fewer iterations
    
    # Method 5: Additional filtering to remove noise
    # Remove very small areas
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Smaller kernel
    gray_final = cv2.morphologyEx(gray_cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(gray_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    valid_contours = []
    centers = []
    
    min_area = 20  # Even smaller minimum area to catch more buildings
    max_area = (w*h) // 2  # Larger maximum area
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Get bounding rectangle
        x, y, ww, hh = cv2.boundingRect(cnt)
        rect_area = ww * hh
        
        if rect_area == 0:
            continue
            
        # Calculate extent (how much of the bounding rect is filled)
        extent = area / rect_area
        
        # Filter out very thin or very sparse areas
        if extent < 0.1:  # Even more lenient extent threshold
            continue
            
        # Calculate aspect ratio
        aspect_ratio = max(ww, hh) / max(min(ww, hh), 1)
        
        # Filter out very elongated shapes (likely roads or linear features)
        if aspect_ratio > 15:  # More lenient aspect ratio
            continue
            
        # Calculate centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            valid_contours.append(cnt)
            centers.append((cx, cy))

    if debug:
        return valid_contours, centers, {
            'gray': img_gray,
            'hsv': hsv,
            'very_white_mask': very_white_mask,
            'white_mask': white_mask,
            'light_gray_mask': light_gray_mask,
            'medium_gray_mask': medium_gray_mask,
            'dark_gray_mask': dark_gray_mask,
            'low_sat_mask': low_sat_mask,
            'gray_combined': gray_combined,
            'gray_cleaned': gray_cleaned,
            'gray_final': gray_final
        }
    else:
        return valid_contours, centers

def detect_building_clusters(centers, img_shape, min_cluster_size=3, eps=50):
    """
    Detect building clusters using DBSCAN clustering algorithm.
    
    Args:
        centers: List of (x, y) building center points
        img_shape: Shape of the image (h, w)
        min_cluster_size: Minimum number of buildings to form a cluster
        eps: Maximum distance between buildings in the same cluster
    
    Returns:
        cluster_centers: List of cluster center points
        cluster_labels: Labels for each building (-1 for noise)
        cluster_sizes: Number of buildings in each cluster
    """
    if len(centers) < min_cluster_size:
        return [], [], []
    
    # Convert centers to numpy array
    points = np.array(centers)
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)
    labels = clustering.labels_
    
    # Calculate cluster centers and sizes
    unique_labels = set(labels)
    cluster_centers = []
    cluster_sizes = []
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
            
        # Get points in this cluster
        cluster_points = points[labels == label]
        cluster_size = len(cluster_points)
        
        # Calculate cluster center (centroid)
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_centers.append((int(cluster_center[0]), int(cluster_center[1])))
        cluster_sizes.append(cluster_size)
    
    return cluster_centers, labels, cluster_sizes

def create_cluster_heatmap(shape, cluster_centers, cluster_sizes, radius=80, sigma=40):
    """
    Create heatmap from building clusters with intensity based on cluster size.
    
    Args:
        shape: Image shape (h, w)
        cluster_centers: List of cluster center points
        cluster_sizes: Number of buildings in each cluster
        radius: Base radius for each cluster
        sigma: Blur sigma for density smoothing
    
    Returns:
        heat: Normalized heatmap (0-1)
    """
    h, w = shape[:2]
    canvas = np.zeros((h, w), dtype=np.float32)
    
    if not cluster_centers:
        return canvas
    
    # Normalize cluster sizes for intensity scaling
    max_size = max(cluster_sizes) if cluster_sizes else 1
    min_size = min(cluster_sizes) if cluster_sizes else 1
    
    for (x, y), size in zip(cluster_centers, cluster_sizes):
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
            
        # Scale radius and intensity based on cluster size
        scaled_radius = int(radius * (size / max_size))
        intensity = (size - min_size) / max(max_size - min_size, 1)
        
        # Draw cluster with scaled intensity
        rr = max(scaled_radius, 20)  # Minimum radius
        x0 = max(0, x - rr)
        x1 = min(w, x + rr + 1)
        y0 = max(0, y - rr)
        y1 = min(h, y + rr + 1)
        
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x)**2 + (yy - y)**2 <= rr*rr
        canvas[y0:y1, x0:x1][mask] += intensity
    
    # Gaussian blur to produce smooth heatmap
    ksize = int(max(3, (sigma*6)//1)) | 1  # ensure odd
    blurred = cv2.GaussianBlur(canvas, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    
    # Normalize to 0..1
    if blurred.max() > 0:
        heat = blurred / blurred.max()
    else:
        heat = blurred
    return heat.astype(np.float32)

def create_heatmap_from_points(shape, points, radius=50, sigma=30):
    """
    Create heatmap image (grayscale float 0..1) from list of (x,y) points.
    Approach: draw delta impulses at centroids, Gaussian blur to simulate density.
    radius: size of brush to stamp each centroid (int)
    sigma: blur sigma for density smoothing
    """
    h, w = shape[:2]
    canvas = np.zeros((h, w), dtype=np.float32)

    for (x, y) in points:
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        # stamp a small disk with value 1
        rr = radius
        x0 = max(0, x-rr)
        x1 = min(w, x+rr+1)
        y0 = max(0, y-rr)
        y1 = min(h, y+rr+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x)**2 + (yy - y)**2 <= rr*rr
        canvas[y0:y1, x0:x1][mask] += 1.0

    # Gaussian blur to produce smooth heatmap (sigma controls spread)
    # cv2.GaussianBlur kernel size should be odd and related to sigma
    ksize = int(max(3, (sigma*6)//1)) | 1  # ensure odd
    blurred = cv2.GaussianBlur(canvas, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    # Normalize to 0..1
    if blurred.max() > 0:
        heat = blurred / blurred.max()
    else:
        heat = blurred
    return heat.astype(np.float32)

def apply_colormap_to_heat(heat, colormap=cv2.COLORMAP_JET):
    """
    Convert grayscale heat (0..1) to BGR color image using OpenCV colormap.
    Enhanced to make heatmap more visible.
    """
    # Apply gamma correction to enhance visibility
    gamma = 0.5
    heat_corrected = np.power(heat, gamma)
    
    # Scale to 0-255
    heat_8u = np.clip((heat_corrected * 255), 0, 255).astype('uint8')
    heat_color = cv2.applyColorMap(heat_8u, colormap)
    return heat_color

def overlay_heatmap_on_image(img_bgr, heat_color, alpha=0.5):
    """
    Overlay colored heatmap onto BGR image with alpha blending.
    heat_color must be same shape as img_bgr.
    """
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heat_color, alpha, 0)
    return overlay

def create_light_gray_heatmap(img_bgr, radius=60, sigma=30):
    """
    Create heatmap directly from light gray areas in the image.
    This focuses on light gray building areas without needing individual building detection.
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    
    # Focus on light gray areas (buildings) with better weighting
    # Very white buildings (highest priority)
    very_white_mask = cv2.inRange(img_gray, 220, 255)
    # White buildings  
    white_mask = cv2.inRange(img_gray, 200, 255)
    # Light gray buildings
    light_gray_mask = cv2.inRange(img_gray, 150, 200)
    # Medium gray buildings
    medium_gray_mask = cv2.inRange(img_gray, 100, 150)
    
    # Combine all light areas with optimized weights
    # Give highest weight to whiter areas, lower weight to darker grays
    combined_mask = cv2.addWeighted(very_white_mask, 1.0, white_mask, 0.9, 0)
    combined_mask = cv2.addWeighted(combined_mask, 0.9, light_gray_mask, 0.7, 0)
    combined_mask = cv2.addWeighted(combined_mask, 0.8, medium_gray_mask, 0.5, 0)
    
    # Apply morphological operations to connect nearby areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Convert to float for heatmap processing
    heat_base = combined_mask.astype(np.float32) / 255.0
    
    # Apply Gaussian blur to create smooth heatmap
    ksize = int(max(3, (sigma*6)//1)) | 1  # ensure odd
    heat_blurred = cv2.GaussianBlur(heat_base, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    
    # Normalize to 0..1
    if heat_blurred.max() > 0:
        heat = heat_blurred / heat_blurred.max()
    else:
        heat = heat_blurred
    
    return heat.astype(np.float32)

def draw_building_contours(img_bgr, contours, color=(0,255,0), thickness=2):
    out = img_bgr.copy()
    cv2.drawContours(out, contours, -1, color, thickness)
    # also draw bounding rects and centers
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(out, (x,y), (x+w, y+h), color, 1)
    return out

def create_simple_group_overlay(img_bgr, cluster_centers, cluster_sizes):
    """
    Create a simple, clean overlay showing only building groups.
    """
    overlay = img_bgr.copy()
    
    # Draw cluster centers with clear, simple visualization
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    for i, ((cx, cy), size) in enumerate(zip(cluster_centers, cluster_sizes)):
        color = colors[i % len(colors)]
        
        # Draw large, clear circles for building groups
        radius = 25 + size * 4  # Even larger radius for better visibility
        cv2.circle(overlay, (cx, cy), radius, color, 4)  # Thick border
        cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), 2)  # White inner border
        
        # Clear, readable text
        cv2.putText(overlay, f'Group {i+1}', (cx-30, cy-radius-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay, f'{size} buildings', (cx-35, cy-radius+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

def main(args):
    # read
    img = cv2.imread(args.input)
    if img is None:
        print("Error: couldn't load image:", args.input)
        return
    orig = img.copy()

    # detect buildings
    contours, centers = detect_buildings(img, debug=False)
    print(f"Detected {len(contours)} building candidates")
    print(f"Building center points: {len(centers)}")

    # detect building clusters
    cluster_centers, cluster_labels, cluster_sizes = detect_building_clusters(
        centers, img.shape, min_cluster_size=2, eps=80
    )
    print(f"Detected {len(cluster_centers)} building clusters")
    total_clustered_buildings = sum(cluster_sizes)
    isolated_buildings = len(centers) - total_clustered_buildings
    
    for i, (center, size) in enumerate(zip(cluster_centers, cluster_sizes)):
        print(f"  Cluster {i+1}: {size} buildings at {center}")
    
    if isolated_buildings > 0:
        print(f"  Isolated buildings: {isolated_buildings}")
    
    print(f"  Total buildings in clusters: {total_clustered_buildings}")
    print(f"  Clustering efficiency: {total_clustered_buildings/len(centers)*100:.1f}%")

    # Create heatmap directly from light gray areas (no circles)
    det_drawn = orig.copy()  # Just use original image
    
    # Create heatmap based on light gray areas
    heat = create_light_gray_heatmap(img, radius=args.radius*2, sigma=args.sigma*1.5)
    heat_color = apply_colormap_to_heat(heat, colormap=cv2.COLORMAP_JET)
    print("Using light gray area-based heatmap")

    # optionally save heat-only
    heat_only_path = args.heatmap if args.heatmap else None
    if heat_only_path:
        # overlay heat on black background for saving; or save heat_color directly
        cv2.imwrite(heat_only_path, heat_color)
        print(f"Heatmap saved to: {heat_only_path}")

    # Create overlay with light gray heatmap
    overlay = overlay_heatmap_on_image(det_drawn, heat_color, alpha=0.5)  # Good visibility

    # Save the result
    out_path = args.out if args.out else "out_overlay.png"
    cv2.imwrite(out_path, overlay)
    
    print(f"Light gray heatmap overlay saved to: {out_path}")
    print("Processing complete! Generated light gray area-based heatmap")

    if args.debug:
        # save intermediate debug images into a folder
        import os
        ddir = args.debug_dir or "debug_outputs"
        os.makedirs(ddir, exist_ok=True)
        # rerun detector with debug info
        _, _, dbg = detect_buildings(img, debug=True)
        cv2.imwrite(os.path.join(ddir, "gray.png"), dbg['gray'])
        cv2.imwrite(os.path.join(ddir, "hsv.png"), dbg['hsv'])
        cv2.imwrite(os.path.join(ddir, "very_white_mask.png"), dbg['very_white_mask'])
        cv2.imwrite(os.path.join(ddir, "white_mask.png"), dbg['white_mask'])
        cv2.imwrite(os.path.join(ddir, "light_gray_mask.png"), dbg['light_gray_mask'])
        cv2.imwrite(os.path.join(ddir, "medium_gray_mask.png"), dbg['medium_gray_mask'])
        cv2.imwrite(os.path.join(ddir, "dark_gray_mask.png"), dbg['dark_gray_mask'])
        cv2.imwrite(os.path.join(ddir, "low_sat_mask.png"), dbg['low_sat_mask'])
        cv2.imwrite(os.path.join(ddir, "gray_combined.png"), dbg['gray_combined'])
        cv2.imwrite(os.path.join(ddir, "gray_cleaned.png"), dbg['gray_cleaned'])
        cv2.imwrite(os.path.join(ddir, "gray_final.png"), dbg['gray_final'])
        print(f"Debug images saved to: {ddir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="test_data/mapping examples/1post.png", help='input image')
    parser.add_argument('--out', default="overlay.png", help='output overlay image')
    parser.add_argument('--heatmap', default="heatmap.png", help='output heatmap image')
    parser.add_argument('--radius', type=int, default=15, help='heatmap point radius')
    parser.add_argument('--sigma', type=float, default=20.0, help='heatmap blur sigma')
    parser.add_argument('--alpha', type=float, default=0.6, help='overlay alpha')
    parser.add_argument('--debug', action='store_true', help='save debug images')
    parser.add_argument('--debug-dir', help='debug output dir')
    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ 找不到图片文件: {args.input}")
        exit()

    main(args)

