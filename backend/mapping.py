"""
Disaster Digital Twin — Mapping Comparison Module

This module implements patch-based change detection for before/after disaster imagery.
It processes satellite or drone images, computes damage scores per grid patch, and outputs
GeoJSON format results with optional Firebase integration and Folium visualization.

Author: AWS Hackathon Team
Date: October 2025
"""

import logging
import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import cv2
from skimage import exposure
from skimage.filters import sobel
from shapely.geometry import Polygon, Point, mapping
import folium
from folium import plugins

# Optional dependencies
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("rasterio not available. GeoTIFF support disabled.")

try:
    import firebase_admin
    from firebase_admin import credentials, db
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    logging.info("firebase-admin not available. Firebase integration disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PRESETS = {
    'default': {'edge': 0.5, 'mean': 0.3, 'var': 0.2},
    'earthquake': {'edge': 0.5, 'mean': 0.2, 'var': 0.3},
    'typhoon': {'edge': 0.6, 'mean': 0.1, 'var': 0.3},
    'flood': {'edge': 0.2, 'mean': 0.6, 'var': 0.2}
}


@dataclass
class PatchFeatures:
    """Container for patch-level image features"""
    mean_before: float
    mean_after: float
    var_before: float
    var_after: float
    edge_before: float
    edge_after: float
    edge_diff: float
    mean_diff: float
    var_diff: float
    cloud_mask: bool


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    grid_size: int = 32
    cloud_thresh: int = 230
    normalize: bool = True
    weights: Dict[str, float] = None
    preset: str = 'default'
    
    def __post_init__(self):
        if self.weights is None:
            if self.preset not in PRESETS:
                logger.warning(f"Preset '{self.preset}' not found. Using default.")
                self.preset = 'default'
            self.weights = PRESETS[self.preset]
        logger.info(f"Using damage score weights for preset '{self.preset}': {self.weights}")


class ImageProcessor:
    """Handles image loading, alignment, and preprocessing"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def load_image(self, path: str) -> Tuple[np.ndarray, Optional[Any], Optional[Any]]:
        """
        Load image from file. Returns (array, transform, crs).
        
        Args:
            path: Path to image file
            
        Returns:
            Tuple of (image_array, transform, crs)
            transform and crs are None for non-GeoTIFF images
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Try to load as GeoTIFF first
        if HAS_RASTERIO and path.suffix.lower() in ['.tif', '.tiff']:
            try:
                with rasterio.open(path) as src:
                    img = src.read()
                    # Convert to (H, W, C) format
                    if img.shape[0] <= 4:  # Assume channels-first
                        img = np.moveaxis(img, 0, -1)
                    transform = src.transform
                    crs = src.crs
                    logger.info(f"Loaded GeoTIFF: {path} with shape {img.shape}")
                    return img, transform, crs
            except Exception as e:
                logger.warning(f"Failed to read as GeoTIFF: {e}. Trying as regular image.")
        
        # Load as regular image
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        logger.info(f"Loaded regular image: {path} with shape {img.shape}")
        return img, None, None
    
    def align_images(self, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two images to the same shape.
        
        Args:
            before: Before image array
            after: After image array
            
        Returns:
            Tuple of aligned (before, after) images
        """
        if before.shape != after.shape:
            logger.info(f"Resizing images from {before.shape} and {after.shape}")
            target_shape = before.shape[:2]  # Use before image size as reference
            after = cv2.resize(after, (target_shape[1], target_shape[0]), 
                             interpolation=cv2.INTER_LINEAR)
            logger.info(f"Aligned to shape: {target_shape}")
        
        return before, after
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess image: convert to grayscale and normalize.
        
        Args:
            img: Input image array
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = img[:, :, 0]  # Take first channel
        else:
            gray = img
        
        # Ensure uint8 format
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return gray
    
    def apply_histogram_matching(self, before: np.ndarray, after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply histogram matching to reduce global brightness differences.
        
        Args:
            before: Before image (grayscale)
            after: After image (grayscale)
            
        Returns:
            Tuple of (before, matched_after)
        """
        try:
            matched = exposure.match_histograms(after, before, channel_axis=None)
            logger.info("Applied histogram matching")
            return before, matched.astype(np.uint8)
        except Exception as e:
            logger.warning(f"Histogram matching failed: {e}. Using original images.")
            return before, after


class ChangeDetector:
    """Performs patch-based change detection"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    def compute_edge_strength(self, patch: np.ndarray) -> float:
        """
        Compute edge strength using Sobel operator.
        
        Args:
            patch: Image patch
            
        Returns:
            Mean edge strength
        """
        edges = sobel(patch)
        return float(np.mean(np.abs(edges)))
    
    def detect_cloud(self, patch: np.ndarray) -> bool:
        """
        Detect if patch contains clouds/overexposed regions.
        
        Args:
            patch: Image patch
            
        Returns:
            True if cloud detected
        """
        mean_val = np.mean(patch)
        return mean_val > self.config.cloud_thresh
    
    def extract_patch_features(self, before_patch: np.ndarray, after_patch: np.ndarray) -> PatchFeatures:
        """
        Extract features from a pair of patches.
        
        Args:
            before_patch: Before image patch
            after_patch: After image patch
            
        Returns:
            PatchFeatures object
        """
        # Compute basic statistics
        mean_before = float(np.mean(before_patch))
        mean_after = float(np.mean(after_patch))
        var_before = float(np.var(before_patch))
        var_after = float(np.var(after_patch))
        
        # Compute edge strength
        edge_before = self.compute_edge_strength(before_patch)
        edge_after = self.compute_edge_strength(after_patch)
        
        # Compute differences
        edge_diff = abs(edge_after - edge_before)
        mean_diff = abs(mean_after - mean_before)
        var_diff = abs(var_after - var_before)
        
        # Cloud detection
        cloud_mask = self.detect_cloud(after_patch)
        
        return PatchFeatures(
            mean_before=mean_before,
            mean_after=mean_after,
            var_before=var_before,
            var_after=var_after,
            edge_before=edge_before,
            edge_after=edge_after,
            edge_diff=edge_diff,
            mean_diff=mean_diff,
            var_diff=var_diff,
            cloud_mask=cloud_mask
        )
    
    def compute_damage_score(self, features: PatchFeatures) -> float:
        """
        Compute damage score from patch features.
        
        Args:
            features: PatchFeatures object
            
        Returns:
            Damage score (unnormalized)
        """
        if features.cloud_mask:
            return 0.0  # Ignore clouded patches
        
        w = self.config.weights
        score = (
            w['edge'] * features.edge_diff +
            w['mean'] * features.mean_diff +
            w['var'] * features.var_diff
        )
        
        return float(score)
    
    def process_patches(self, before: np.ndarray, after: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process all patches in image pair.
        
        Args:
            before: Before image (grayscale)
            after: After image (grayscale)
            
        Returns:
            List of patch results with features and scores
        """
        h, w = before.shape
        grid_size = self.config.grid_size
        results = []
        
        num_rows = h // grid_size
        num_cols = w // grid_size
        
        logger.info(f"Processing {num_rows}x{num_cols} = {num_rows*num_cols} patches")
        
        for i in range(num_rows):
            for j in range(num_cols):
                # Extract patches
                y_start = i * grid_size
                y_end = y_start + grid_size
                x_start = j * grid_size
                x_end = x_start + grid_size
                
                before_patch = before[y_start:y_end, x_start:x_end]
                after_patch = after[y_start:y_end, x_start:x_end]
                
                # Extract features
                features = self.extract_patch_features(before_patch, after_patch)
                
                # Compute damage score
                damage_score = self.compute_damage_score(features)
                
                # Store result
                result = {
                    'grid_id': f"{i}_{j}",
                    'row': i,
                    'col': j,
                    'pixel_y': y_start + grid_size // 2,
                    'pixel_x': x_start + grid_size // 2,
                    'bbox_pixels': [x_start, y_start, x_end, y_end],
                    'features': features,
                    'damage_score': damage_score
                }
                
                results.append(result)
        
        return results
    
    def normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize damage scores using percentile scaling to handle outliers.
        
        Args:
            results: List of patch results
            
        Returns:
            Updated results with normalized scores
        """
        if not self.config.normalize:
            return results
        
        scores = [r['damage_score'] for r in results if not r['features'].cloud_mask]
        
        if not scores:
            logger.warning("No valid scores to normalize (all patches might be clouded).")
            return results
            
        min_score = np.min(scores)
        
        # Use 99th percentile as the max to avoid outliers skewing the scale
        max_score = np.percentile(scores, 99)
        
        if max_score <= min_score:
            logger.warning(f"Score range is too narrow or invalid (min: {min_score}, 99th percentile: {max_score}). All scores will be set to 0 or 1.")
            # Handle cases where all scores are the same
            for result in results:
                if not result['features'].cloud_mask:
                    result['damage_score'] = 0.0 if max_score <= min_score else 1.0
            return results

        logger.info(f"Normalizing scores using 99th percentile. Raw range min: {min_score:.2f}, max: {np.max(scores):.2f}. Effective max for scaling: {max_score:.2f}")
        
        for result in results:
            if not result['features'].cloud_mask:
                raw_score = result['damage_score']
                # Scale score and clip to [0, 1] range
                normalized_score = (raw_score - min_score) / (max_score - min_score)
                result['damage_score'] = float(np.clip(normalized_score, 0, 1))
            else:
                result['damage_score'] = 0.0
                
        return results


class GeoConverter:
    """Converts pixel coordinates to geographic coordinates"""
    
    def __init__(self, transform=None, bbox: Optional[List[float]] = None, image_shape: Optional[Tuple[int, int]] = None):
        """
        Initialize geo converter.
        
        Args:
            transform: Rasterio affine transform
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            image_shape: Image shape (height, width) for bbox mapping
        """
        self.transform = transform
        self.bbox = bbox
        self.image_shape = image_shape
        
        if transform is None and bbox is None:
            logger.warning("No georeference information provided. Coordinates will be in pixel space.")
    
    def pixel_to_geo(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple of (longitude, latitude)
        """
        if self.transform is not None and HAS_RASTERIO:
            # Use rasterio transform
            lon, lat = xy(self.transform, pixel_y, pixel_x)
            return float(lon), float(lat)
        
        elif self.bbox is not None and self.image_shape is not None:
            # Linear mapping from bbox
            min_lon, min_lat, max_lon, max_lat = self.bbox
            h, w = self.image_shape
            
            lon = min_lon + (pixel_x / w) * (max_lon - min_lon)
            lat = max_lat - (pixel_y / h) * (max_lat - min_lat)  # Y is inverted
            
            return float(lon), float(lat)
        
        else:
            # Return pixel coordinates as fallback
            return float(pixel_x), float(pixel_y)
    
    def pixel_bbox_to_polygon(self, x_start: int, y_start: int, x_end: int, y_end: int) -> Polygon:
        """
        Convert pixel bounding box to geographic polygon.
        
        Args:
            x_start, y_start: Top-left corner
            x_end, y_end: Bottom-right corner
            
        Returns:
            Shapely Polygon
        """
        corners = [
            self.pixel_to_geo(x_start, y_start),
            self.pixel_to_geo(x_end, y_start),
            self.pixel_to_geo(x_end, y_end),
            self.pixel_to_geo(x_start, y_end),
            self.pixel_to_geo(x_start, y_start)  # Close polygon
        ]
        
        return Polygon(corners)


def process_image_pair(
    before_path: str,
    after_path: str,
    output_geojson_path: str,
    grid_size: int = 32,
    weights: Optional[Dict[str, float]] = None,
    bbox: Optional[List[float]] = None,
    cloud_thresh: int = 230,
    normalize: bool = True,
    preset: str = 'default'
) -> Dict[str, Any]:
    """
    Main processing function: analyze before/after image pair and generate damage map.
    
    Args:
        before_path: Path to before image
        after_path: Path to after image
        output_geojson_path: Output path for GeoJSON file
        grid_size: Size of grid patches (default: 32)
        weights: Dict with 'edge', 'mean', 'var' weights (default: {edge:0.5, mean:0.3, var:0.2})
        bbox: Optional bounding box [min_lon, min_lat, max_lon, max_lat]
        cloud_thresh: Threshold for cloud detection (default: 230)
        normalize: Whether to normalize scores to [0,1] (default: True)
        preset: Damage analysis preset, e.g. 'earthquake', 'typhoon' (default: 'default')
        
    Returns:
        GeoJSON FeatureCollection dict
    """
    logger.info("=" * 60)
    logger.info("Starting image pair processing")
    logger.info(f"Before: {before_path}")
    logger.info(f"After: {after_path}")
    logger.info("=" * 60)
    
    # Create configuration
    config = ProcessingConfig(
        grid_size=grid_size,
        cloud_thresh=cloud_thresh,
        normalize=normalize,
        weights=weights,
        preset=preset
    )
    
    # Initialize processors
    img_processor = ImageProcessor(config)
    change_detector = ChangeDetector(config)
    
    # Load images
    before_img, before_transform, before_crs = img_processor.load_image(before_path)
    after_img, after_transform, after_crs = img_processor.load_image(after_path)
    
    # Align images
    before_img, after_img = img_processor.align_images(before_img, after_img)
    
    # Preprocess
    before_gray = img_processor.preprocess(before_img)
    after_gray = img_processor.preprocess(after_img)
    
    # Apply histogram matching
    before_gray, after_gray = img_processor.apply_histogram_matching(before_gray, after_gray)
    
    # Process patches
    results = change_detector.process_patches(before_gray, after_gray)
    
    # Normalize scores
    results = change_detector.normalize_scores(results)
    
    # Initialize geo converter
    geo_converter = GeoConverter(
        transform=before_transform,
        bbox=bbox,
        image_shape=before_gray.shape
    )
    
    # Generate GeoJSON
    features = []
    for result in results:
        # Get patch center coordinates
        lon, lat = geo_converter.pixel_to_geo(result['pixel_x'], result['pixel_y'])
        
        # Get patch polygon
        x_start, y_start, x_end, y_end = result['bbox_pixels']
        polygon = geo_converter.pixel_bbox_to_polygon(x_start, y_start, x_end, y_end)
        
        # Create feature
        f = result['features']
        feature = {
            'type': 'Feature',
            'geometry': mapping(polygon),
            'properties': {
                'grid_id': result['grid_id'],
                'lat': lat,
                'lon': lon,
                'damage_score': round(result['damage_score'], 4),
                'mean_before': round(f.mean_before, 2),
                'mean_after': round(f.mean_after, 2),
                'var_before': round(f.var_before, 2),
                'var_after': round(f.var_after, 2),
                'edge_diff': round(f.edge_diff, 4),
                'cloud_mask': bool(f.cloud_mask)  # Convert to Python bool
            }
        }
        
        features.append(feature)
    
    # Create FeatureCollection
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    # Write to file
    output_path = Path(output_geojson_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    logger.info(f"GeoJSON written to: {output_path}")
    logger.info(f"Total features: {len(features)}")
    
    # Statistics
    non_cloud = [f for f in features if not f['properties']['cloud_mask']]
    if non_cloud:
        scores = [f['properties']['damage_score'] for f in non_cloud]
        logger.info(f"Damage scores - min: {min(scores):.3f}, max: {max(scores):.3f}, mean: {np.mean(scores):.3f}")
    
    return geojson


def visualize_geojson_on_map(
    geojson_path: str,
    output_html_path: str,
    tile_style: str = 'OpenStreetMap'
) -> None:
    """
    Visualize GeoJSON damage map using Folium.
    
    Args:
        geojson_path: Path to GeoJSON file
        output_html_path: Output path for HTML map
        tile_style: Map tile style (default: 'OpenStreetMap')
    """
    logger.info(f"Creating map visualization: {output_html_path}")
    
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    if not geojson_data['features']:
        logger.warning("No features in GeoJSON")
        return
    
    # Calculate center
    lats = [f['properties']['lat'] for f in geojson_data['features']]
    lons = [f['properties']['lon'] for f in geojson_data['features']]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=tile_style
    )
    
    # Color mapping function
    def get_color(damage_score: float) -> str:
        """Map damage score to color"""
        if damage_score >= 0.8:
            return '#d32f2f'  # Red
        elif damage_score >= 0.6:
            return '#f57c00'  # Orange
        elif damage_score >= 0.4:
            return '#fbc02d'  # Yellow
        elif damage_score >= 0.2:
            return '#8bc34a'  # Light green
        else:
            return '#388e3c'  # Green
    
    # Add features to map
    for feature in geojson_data['features']:
        props = feature['properties']
        
        # Skip clouds
        if props.get('cloud_mask', False):
            continue
        
        color = get_color(props['damage_score'])
        
        # Create popup
        popup_html = f"""
        <div style="font-family: Arial; width: 250px;">
            <h4>Grid {props['grid_id']}</h4>
            <b>Damage Score:</b> {props['damage_score']:.3f}<br>
            <b>Location:</b> {props['lat']:.4f}, {props['lon']:.4f}<br>
            <hr>
            <b>Before:</b> mean={props['mean_before']:.1f}, var={props['var_before']:.1f}<br>
            <b>After:</b> mean={props['mean_after']:.1f}, var={props['var_after']:.1f}<br>
            <b>Edge diff:</b> {props['edge_diff']:.3f}
        </div>
        """
        
        # Add polygon
        folium.GeoJson(
            feature,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.6
            },
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p style="margin: 0; font-weight: bold;">Damage Level</p>
    <p style="margin: 5px 0;"><span style="background-color:#d32f2f; padding: 2px 8px;">&nbsp;</span> Critical (0.8-1.0)</p>
    <p style="margin: 5px 0;"><span style="background-color:#f57c00; padding: 2px 8px;">&nbsp;</span> High (0.6-0.8)</p>
    <p style="margin: 5px 0;"><span style="background-color:#fbc02d; padding: 2px 8px;">&nbsp;</span> Medium (0.4-0.6)</p>
    <p style="margin: 5px 0;"><span style="background-color:#8bc34a; padding: 2px 8px;">&nbsp;</span> Low (0.2-0.4)</p>
    <p style="margin: 5px 0;"><span style="background-color:#388e3c; padding: 2px 8px;">&nbsp;</span> Minimal (0.0-0.2)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    logger.info(f"Map saved to: {output_path}")


def push_geojson_to_firebase(
    geojson_path: str,
    firebase_config: Optional[str] = None,
    db_path: str = 'disaster/zones'
) -> bool:
    """
    Push GeoJSON data to Firebase Realtime Database.
    
    Args:
        geojson_path: Path to GeoJSON file
        firebase_config: Path to Firebase credentials JSON file
        db_path: Database path (default: 'disaster/zones')
        
    Returns:
        True if successful, False otherwise
    """
    if not HAS_FIREBASE:
        logger.error("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
        return False
    
    if firebase_config is None:
        logger.error("Firebase config not provided. Please provide path to service account JSON file.")
        return False
    
    try:
        # Initialize Firebase (if not already initialized)
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://your-project.firebaseio.com'  # Replace with actual URL
            })
            logger.info("Firebase initialized")
        
        # Load GeoJSON
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        # Push to database
        ref = db.reference(db_path)
        ref.set(geojson_data)
        
        logger.info(f"GeoJSON pushed to Firebase at: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to push to Firebase: {e}")
        return False


# ============================================================================
# SYNTHETIC DATA GENERATION (for testing)
# ============================================================================

def generate_synthetic_pair(output_dir: str = 'test_data') -> Tuple[str, str]:
    """
    Generate synthetic before/after image pair for testing.
    
    Creates two images where the 'after' image has a bright square in the center
    to simulate damage.
    
    Args:
        output_dir: Directory to save images
        
    Returns:
        Tuple of (before_path, after_path)
    """
    logger.info("Generating synthetic image pair")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create base image (256x256)
    size = 256
    
    # Before image: random texture
    np.random.seed(42)
    before = np.random.randint(80, 120, (size, size), dtype=np.uint8)
    
    # Add some texture
    for _ in range(10):
        x, y = np.random.randint(0, size, 2)
        cv2.circle(before, (x, y), np.random.randint(10, 30), 
                  int(np.random.randint(60, 140)), -1)
    
    # After image: copy of before with damage in center
    after = before.copy()
    
    # Add bright square in center (simulating destroyed building/debris)
    center_size = 64
    y_start = (size - center_size) // 2
    x_start = (size - center_size) // 2
    after[y_start:y_start+center_size, x_start:x_start+center_size] = 240
    
    # Add some random damage spots
    for _ in range(5):
        x, y = np.random.randint(50, size-50, 2)
        cv2.circle(after, (x, y), np.random.randint(5, 15), 220, -1)
    
    # Save images
    before_path = output_path / 'before.png'
    after_path = output_path / 'after.png'
    
    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    
    logger.info(f"Synthetic images saved to: {output_path}")
    
    return str(before_path), str(after_path)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Disaster Mapping Comparison - Patch-based Change Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GeoTIFF images
  python mapping.py --before before.tif --after after.tif --out damage.geojson
  
  # Process regular images with bounding box
  python mapping.py --before before.png --after after.png --out damage.geojson \\
                    --bbox "-86.8,36.1,-86.7,36.2"
  
  # Custom grid size and weights
  python mapping.py --before before.tif --after after.tif --out damage.geojson \\
                    --grid 64 --weights '{"edge":0.6,"mean":0.3,"var":0.1}'
  
  # Generate test data and process
  python mapping.py --generate-test
        """
    )
    
    parser.add_argument('--before', type=str, help='Path to before image')
    parser.add_argument('--after', type=str, help='Path to after image')
    parser.add_argument('--out', type=str, help='Output GeoJSON path')
    parser.add_argument('--grid', type=int, default=32, help='Grid patch size (default: 32)')
    parser.add_argument('--bbox', type=str, help='Bounding box as "min_lon,min_lat,max_lon,max_lat"')
    parser.add_argument('--cloud-thresh', type=int, default=230, help='Cloud detection threshold (default: 230)')
    parser.add_argument('--weights', type=str, help='JSON string of weights: {"edge":0.5,"mean":0.3,"var":0.2}')
    parser.add_argument('--preset', type=str, default='default', choices=PRESETS.keys(),
                        help=f"Damage analysis preset. Choices: {list(PRESETS.keys())}")
    parser.add_argument('--visualize', action='store_true', help='Generate HTML visualization')
    parser.add_argument('--html-out', type=str, help='Output HTML path (default: same as GeoJSON)')
    parser.add_argument('--firebase-config', type=str, help='Path to Firebase service account JSON')
    parser.add_argument('--firebase-path', type=str, default='disaster/zones', help='Firebase database path')
    parser.add_argument('--generate-test', action='store_true', help='Generate and process synthetic test data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle test data generation
    if args.generate_test:
        logger.info("Running test with synthetic data")
        before_path, after_path = generate_synthetic_pair('test_data')
        args.before = before_path
        args.after = after_path
        args.out = 'test_data/damage.geojson'
        args.bbox = "-86.8,36.1,-86.7,36.2"
        args.visualize = True
    
    # Validate required arguments
    if not args.before or not args.after or not args.out:
        parser.error("--before, --after, and --out are required (or use --generate-test)")
    
    # Parse bbox
    bbox = None
    if args.bbox:
        try:
            bbox = [float(x.strip()) for x in args.bbox.split(',')]
            if len(bbox) != 4:
                raise ValueError
            logger.info(f"Using bounding box: {bbox}")
        except ValueError:
            parser.error("--bbox must be 4 comma-separated numbers: min_lon,min_lat,max_lon,max_lat")
    
    # Parse weights
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
            logger.info(f"Using custom weights: {weights}")
        except json.JSONDecodeError:
            parser.error("--weights must be valid JSON")
    
    # Process images
    try:
        geojson_data = process_image_pair(
            before_path=args.before,
            after_path=args.after,
            output_geojson_path=args.out,
            grid_size=args.grid,
            weights=weights,
            bbox=bbox,
            cloud_thresh=args.cloud_thresh,
            preset=args.preset
        )
        
        logger.info("✓ Processing completed successfully")
        
        # Visualize
        if args.visualize:
            html_out = args.html_out or args.out.replace('.geojson', '.html')
            visualize_geojson_on_map(args.out, html_out)
            logger.info(f"✓ Visualization created: {html_out}")
        
        # Push to Firebase
        if args.firebase_config:
            success = push_geojson_to_firebase(
                args.out,
                firebase_config=args.firebase_config,
                db_path=args.firebase_path
            )
            if success:
                logger.info("✓ Pushed to Firebase")
        
        logger.info("=" * 60)
        logger.info("All tasks completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

