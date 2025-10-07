"""
Unit tests for mapping.py module

Run with: pytest test_mapping.py -v
"""

import pytest
import numpy as np
import cv2
import json
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mapping import (
    process_image_pair,
    visualize_geojson_on_map,
    generate_synthetic_pair,
    ImageProcessor,
    ChangeDetector,
    GeoConverter,
    ProcessingConfig,
    PatchFeatures
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.fixture
def identical_images(temp_dir):
    """Generate two identical images"""
    img = np.random.randint(80, 120, (128, 128), dtype=np.uint8)
    
    before_path = Path(temp_dir) / 'before.png'
    after_path = Path(temp_dir) / 'after.png'
    
    cv2.imwrite(str(before_path), img)
    cv2.imwrite(str(after_path), img)
    
    return str(before_path), str(after_path)


@pytest.fixture
def damaged_images(temp_dir):
    """Generate before/after images with damage in center"""
    size = 128
    
    # Before: uniform-ish texture
    np.random.seed(42)
    before = np.random.randint(80, 120, (size, size), dtype=np.uint8)
    
    # After: same but with bright square in center
    after = before.copy()
    damage_size = 32
    y_start = (size - damage_size) // 2
    x_start = (size - damage_size) // 2
    after[y_start:y_start+damage_size, x_start:x_start+damage_size] = 240
    
    before_path = Path(temp_dir) / 'before.png'
    after_path = Path(temp_dir) / 'after.png'
    
    cv2.imwrite(str(before_path), before)
    cv2.imwrite(str(after_path), after)
    
    return str(before_path), str(after_path)


class TestImageProcessor:
    """Test ImageProcessor class"""
    
    def test_load_image(self, temp_dir):
        """Test image loading"""
        # Create test image
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img_path = Path(temp_dir) / 'test.png'
        cv2.imwrite(str(img_path), img)
        
        config = ProcessingConfig()
        processor = ImageProcessor(config)
        
        loaded, transform, crs = processor.load_image(str(img_path))
        
        assert loaded is not None
        assert loaded.shape == (64, 64)
    
    def test_align_images(self):
        """Test image alignment"""
        config = ProcessingConfig()
        processor = ImageProcessor(config)
        
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((150, 150), dtype=np.uint8)
        
        aligned1, aligned2 = processor.align_images(img1, img2)
        
        assert aligned1.shape == aligned2.shape
        assert aligned1.shape == (100, 100)
    
    def test_preprocess(self):
        """Test image preprocessing"""
        config = ProcessingConfig()
        processor = ImageProcessor(config)
        
        # Color image
        img_color = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gray = processor.preprocess(img_color)
        
        assert len(gray.shape) == 2
        assert gray.dtype == np.uint8


class TestChangeDetector:
    """Test ChangeDetector class"""
    
    def test_compute_edge_strength(self):
        """Test edge detection"""
        config = ProcessingConfig()
        detector = ChangeDetector(config)
        
        # Create patch with edges
        patch = np.zeros((32, 32), dtype=np.uint8)
        patch[10:20, 10:20] = 255  # White square
        
        edge_strength = detector.compute_edge_strength(patch)
        
        assert edge_strength > 0
    
    def test_detect_cloud(self):
        """Test cloud detection"""
        config = ProcessingConfig(cloud_thresh=230)
        detector = ChangeDetector(config)
        
        # Bright patch (cloud)
        bright_patch = np.ones((32, 32), dtype=np.uint8) * 250
        assert detector.detect_cloud(bright_patch) == True
        
        # Dark patch (no cloud)
        dark_patch = np.ones((32, 32), dtype=np.uint8) * 100
        assert detector.detect_cloud(dark_patch) == False
    
    def test_extract_patch_features(self):
        """Test feature extraction"""
        config = ProcessingConfig()
        detector = ChangeDetector(config)
        
        before_patch = np.ones((32, 32), dtype=np.uint8) * 100
        after_patch = np.ones((32, 32), dtype=np.uint8) * 150
        
        features = detector.extract_patch_features(before_patch, after_patch)
        
        assert isinstance(features, PatchFeatures)
        assert features.mean_before < features.mean_after
        assert features.mean_diff > 0
    
    def test_compute_damage_score(self):
        """Test damage score computation"""
        config = ProcessingConfig()
        detector = ChangeDetector(config)
        
        # High difference features
        features = PatchFeatures(
            mean_before=100,
            mean_after=200,
            var_before=10,
            var_after=50,
            edge_before=5,
            edge_after=15,
            edge_diff=10,
            mean_diff=100,
            var_diff=40,
            cloud_mask=False
        )
        
        score = detector.compute_damage_score(features)
        assert score > 0
        
        # Clouded patch should return 0
        features.cloud_mask = True
        score = detector.compute_damage_score(features)
        assert score == 0


class TestGeoConverter:
    """Test GeoConverter class"""
    
    def test_bbox_conversion(self):
        """Test pixel to geo conversion using bbox"""
        bbox = [-86.8, 36.1, -86.7, 36.2]
        image_shape = (100, 100)
        
        converter = GeoConverter(bbox=bbox, image_shape=image_shape)
        
        # Test corners
        lon, lat = converter.pixel_to_geo(0, 0)
        assert abs(lon - bbox[0]) < 0.01
        assert abs(lat - bbox[3]) < 0.01
        
        lon, lat = converter.pixel_to_geo(100, 100)
        assert abs(lon - bbox[2]) < 0.01
        assert abs(lat - bbox[1]) < 0.01
    
    def test_pixel_bbox_to_polygon(self):
        """Test polygon creation"""
        bbox = [-86.8, 36.1, -86.7, 36.2]
        image_shape = (100, 100)
        
        converter = GeoConverter(bbox=bbox, image_shape=image_shape)
        polygon = converter.pixel_bbox_to_polygon(0, 0, 10, 10)
        
        assert polygon.is_valid
        assert len(polygon.exterior.coords) == 5  # 4 corners + closing point


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_identical_images_low_scores(self, identical_images, temp_dir):
        """Test 1: Identical images should produce low damage scores"""
        before_path, after_path = identical_images
        output_path = Path(temp_dir) / 'output.geojson'
        
        geojson = process_image_pair(
            before_path,
            after_path,
            str(output_path),
            grid_size=32,
            bbox=[-86.8, 36.1, -86.7, 36.2]
        )
        
        # Check that file was created
        assert output_path.exists()
        
        # Check GeoJSON structure
        assert geojson['type'] == 'FeatureCollection'
        assert len(geojson['features']) > 0
        
        # All scores should be very low (close to 0 after normalization)
        scores = [f['properties']['damage_score'] for f in geojson['features']]
        max_score = max(scores)
        
        # With identical images, even the max should be low
        # (allowing some tolerance due to numerical precision)
        assert max_score < 0.3, f"Expected low scores for identical images, got max {max_score}"
    
    def test_damaged_center_high_scores(self, damaged_images, temp_dir):
        """Test 2: Image with damage in center should have high scores there"""
        before_path, after_path = damaged_images
        output_path = Path(temp_dir) / 'output.geojson'
        
        geojson = process_image_pair(
            before_path,
            after_path,
            str(output_path),
            grid_size=32,
            bbox=[-86.8, 36.1, -86.7, 36.2]
        )
        
        assert output_path.exists()
        
        # Find center patches (grid is 128/32 = 4x4, center should be grids 1_1, 1_2, 2_1, 2_2)
        center_grids = ['1_1', '1_2', '2_1', '2_2']
        center_scores = []
        edge_scores = []
        
        for feature in geojson['features']:
            grid_id = feature['properties']['grid_id']
            score = feature['properties']['damage_score']
            
            if grid_id in center_grids:
                center_scores.append(score)
            else:
                edge_scores.append(score)
        
        # Center patches should have higher average score than edge patches
        if center_scores and edge_scores:
            avg_center = np.mean(center_scores)
            avg_edge = np.mean(edge_scores)
            
            assert avg_center > avg_edge, \
                f"Center avg ({avg_center:.3f}) should be > edge avg ({avg_edge:.3f})"
    
    def test_bbox_coordinates_in_range(self, damaged_images, temp_dir):
        """Test 3: With bbox, all coordinates should be within bbox range"""
        before_path, after_path = damaged_images
        output_path = Path(temp_dir) / 'output.geojson'
        
        bbox = [-86.8, 36.1, -86.7, 36.2]
        
        geojson = process_image_pair(
            before_path,
            after_path,
            str(output_path),
            grid_size=32,
            bbox=bbox
        )
        
        # Check all coordinates are within bbox
        for feature in geojson['features']:
            lon = feature['properties']['lon']
            lat = feature['properties']['lat']
            
            assert bbox[0] <= lon <= bbox[2], f"Longitude {lon} outside bbox"
            assert bbox[1] <= lat <= bbox[3], f"Latitude {lat} outside bbox"
    
    def test_visualization(self, damaged_images, temp_dir):
        """Test visualization generation"""
        before_path, after_path = damaged_images
        geojson_path = Path(temp_dir) / 'output.geojson'
        html_path = Path(temp_dir) / 'map.html'
        
        # First process images
        process_image_pair(
            before_path,
            after_path,
            str(geojson_path),
            grid_size=32,
            bbox=[-86.8, 36.1, -86.7, 36.2]
        )
        
        # Then visualize
        visualize_geojson_on_map(str(geojson_path), str(html_path))
        
        assert html_path.exists()
        
        # Check HTML content
        with open(html_path, 'r') as f:
            html_content = f.read()
            assert 'leaflet' in html_content.lower() or 'folium' in html_content.lower()


def test_synthetic_generation(temp_dir):
    """Test synthetic image pair generation"""
    before_path, after_path = generate_synthetic_pair(temp_dir)
    
    assert Path(before_path).exists()
    assert Path(after_path).exists()
    
    # Load and verify images are different
    before = cv2.imread(before_path, cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(after_path, cv2.IMREAD_GRAYSCALE)
    
    assert before.shape == after.shape
    assert not np.array_equal(before, after), "Synthetic images should be different"


def example_usage():
    """
    Example usage function demonstrating the complete pipeline.
    Can be run independently without pytest.
    """
    print("=" * 60)
    print("MAPPING MODULE - EXAMPLE USAGE")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nWorking directory: {temp_dir}")
    
    try:
        # Generate synthetic test data
        print("\n1. Generating synthetic image pair...")
        before_path, after_path = generate_synthetic_pair(temp_dir)
        print(f"   Before: {before_path}")
        print(f"   After: {after_path}")
        
        # Process images
        print("\n2. Processing image pair...")
        output_geojson = Path(temp_dir) / 'damage.geojson'
        
        geojson = process_image_pair(
            before_path,
            after_path,
            str(output_geojson),
            grid_size=32,
            bbox=[-86.8, 36.1, -86.7, 36.2],
            cloud_thresh=230,
            normalize=True
        )
        
        print(f"   Generated {len(geojson['features'])} patches")
        
        # Print statistics
        scores = [f['properties']['damage_score'] for f in geojson['features'] 
                 if not f['properties']['cloud_mask']]
        
        if scores:
            print(f"\n3. Damage Score Statistics:")
            print(f"   Min:  {min(scores):.3f}")
            print(f"   Max:  {max(scores):.3f}")
            print(f"   Mean: {np.mean(scores):.3f}")
            print(f"   Std:  {np.std(scores):.3f}")
        
        # Create visualization
        print("\n4. Creating visualization...")
        output_html = Path(temp_dir) / 'damage_map.html'
        visualize_geojson_on_map(str(output_geojson), str(output_html))
        print(f"   Map saved to: {output_html}")
        
        print("\n" + "=" * 60)
        print("SUCCESS! All steps completed.")
        print("=" * 60)
        print(f"\nOutput files:")
        print(f"  - GeoJSON: {output_geojson}")
        print(f"  - HTML Map: {output_html}")
        print(f"\nOpen {output_html} in your browser to view the map!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Note: In real usage you might want to keep the temp dir
        # For now we'll just print it
        print(f"\nTemp directory: {temp_dir}")
        print("(Not cleaned up automatically - delete manually if needed)")


if __name__ == '__main__':
    # Run example usage if called directly
    example_usage()

