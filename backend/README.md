# Disaster Digital Twin - Mapping Comparison Module

This module implements patch-based change detection for before/after disaster satellite or drone imagery. It computes damage scores for grid patches and outputs GeoJSON format results with optional visualization and Firebase integration.

## Features

- 🛰️ **Image Processing**: Supports GeoTIFF (with geo-referencing) and standard image formats (PNG, JPG)
- 📊 **Patch-based Analysis**: Divides images into grid patches and computes change metrics
- 🌥️ **Cloud Detection**: Automatically detects and masks cloudy/overexposed regions
- 🎨 **Histogram Matching**: Reduces global brightness differences between images
- 🔍 **Edge Detection**: Uses Sobel operator to detect structural changes
- 🗺️ **GeoJSON Output**: Standard format for damage scores with geographic coordinates
- 📍 **Interactive Maps**: Folium-based visualization with color-coded damage levels
- 🔥 **Firebase Integration**: Optional push to Firebase Realtime Database

## Installation

```bash
# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Core**: numpy, opencv-python, scikit-image, shapely, folium
- **Optional**: rasterio (for GeoTIFF support), firebase-admin (for Firebase integration)

## Usage

### Basic Command Line Usage

```bash
# Process GeoTIFF images (with embedded geo-coordinates)
python mapping.py --before data/before.tif --after data/after.tif --out results/damage.geojson

# Process regular images with manual bounding box
python mapping.py --before data/before.png --after data/after.png \
                  --out results/damage.geojson \
                  --bbox "-86.8,36.1,-86.7,36.2"

# With visualization
python mapping.py --before data/before.tif --after data/after.tif \
                  --out results/damage.geojson \
                  --visualize

# Custom grid size and weights
python mapping.py --before data/before.tif --after data/after.tif \
                  --out results/damage.geojson \
                  --grid 64 \
                  --weights '{"edge":0.6,"mean":0.3,"var":0.1}'

# Generate and process test data
python mapping.py --generate-test
```

### Python API Usage

```python
from mapping import process_image_pair, visualize_geojson_on_map

# Process image pair
geojson = process_image_pair(
    before_path='before.tif',
    after_path='after.tif',
    output_geojson_path='damage.geojson',
    grid_size=32,
    bbox=[-86.8, 36.1, -86.7, 36.2],  # Optional for non-GeoTIFF
    cloud_thresh=230,
    normalize=True
)

# Create visualization
visualize_geojson_on_map(
    geojson_path='damage.geojson',
    output_html_path='damage_map.html'
)
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--before` | Path to before image | Required |
| `--after` | Path to after image | Required |
| `--out` | Output GeoJSON path | Required |
| `--grid` | Grid patch size in pixels | 32 |
| `--bbox` | Bounding box "min_lon,min_lat,max_lon,max_lat" | Auto-detect |
| `--cloud-thresh` | Cloud detection threshold (0-255) | 230 |
| `--weights` | JSON weights for damage score | `{"edge":0.5,"mean":0.3,"var":0.2}` |
| `--visualize` | Generate HTML visualization | False |
| `--html-out` | Output HTML path | Same as GeoJSON |
| `--firebase-config` | Firebase credentials JSON path | None |
| `--firebase-path` | Firebase database path | `disaster/zones` |
| `--generate-test` | Generate and process synthetic test data | False |
| `--verbose, -v` | Enable verbose logging | False |

## Algorithm Details

### Processing Pipeline

1. **Image Loading**
   - Load before/after images
   - Extract geo-reference info (if GeoTIFF)
   - Handle various formats (PNG, JPG, TIF)

2. **Image Alignment**
   - Resample to matching dimensions
   - Convert to grayscale
   - Apply histogram matching to reduce lighting differences

3. **Patch Processing**
   - Divide images into NxN grid patches
   - For each patch, compute:
     - Mean brightness (before/after)
     - Variance (before/after)
     - Edge strength using Sobel operator (before/after)

4. **Cloud Detection**
   - Identify overexposed patches (mean > threshold)
   - Mark clouded patches for exclusion

5. **Damage Scoring**
   - Compute differences: `edge_diff`, `mean_diff`, `var_diff`
   - Weighted combination: `score = w_edge*edge_diff + w_mean*mean_diff + w_var*var_diff`
   - Normalize to [0, 1] range

6. **GeoJSON Generation**
   - Convert pixel coordinates to lat/lon
   - Create polygon features for each patch
   - Include all metadata (scores, features, cloud mask)

### Damage Score Calculation

```
damage_score = w_edge * |edge_after - edge_before| + 
               w_mean * |mean_after - mean_before| + 
               w_var * |var_after - var_before|
```

**Default weights**: `edge=0.5`, `mean=0.3`, `var=0.2`

## Output Format

### GeoJSON Structure

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon1,lat1],[lon2,lat2],[lon3,lat3],[lon4,lat4],[lon1,lat1]]]
      },
      "properties": {
        "grid_id": "10_5",
        "lat": 36.1627,
        "lon": -86.7816,
        "damage_score": 0.82,
        "mean_before": 123.4,
        "mean_after": 34.2,
        "var_before": 45.2,
        "var_after": 78.3,
        "edge_diff": 12.3,
        "cloud_mask": false
      }
    }
  ]
}
```

### Visualization

The HTML visualization includes:
- Interactive Folium/Leaflet map
- Color-coded patches (green → yellow → orange → red)
- Clickable patches showing detailed properties
- Legend with damage level ranges

**Color Scheme**:
- 🔴 Red: Critical (0.8-1.0)
- 🟠 Orange: High (0.6-0.8)
- 🟡 Yellow: Medium (0.4-0.6)
- 🟢 Light Green: Low (0.2-0.4)
- 🟢 Green: Minimal (0.0-0.2)

## Testing

```bash
# Run all tests
pytest tests/test_mapping.py -v

# Run with coverage
pytest tests/test_mapping.py --cov=mapping --cov-report=html

# Run example usage
python tests/test_mapping.py
```

### Test Cases

1. **Identical Images**: Verifies low damage scores for unchanged scenes
2. **Center Damage**: Confirms high scores in damaged regions
3. **Bbox Coordinates**: Validates coordinate transformation accuracy
4. **Visualization**: Tests HTML map generation

## Integration with Other Modules

### Calling from fusion.py

```python
from mapping import process_image_pair

# Process mapping data
damage_geojson = process_image_pair(
    before_path=s3_before_url,
    after_path=s3_after_url,
    output_geojson_path='temp/damage.geojson',
    grid_size=32
)

# Extract damage scores for fusion
damage_scores = {
    f['properties']['grid_id']: f['properties']['damage_score']
    for f in damage_geojson['features']
}
```

### Firebase Integration

```bash
# Push to Firebase
python mapping.py --before before.tif --after after.tif \
                  --out damage.geojson \
                  --firebase-config ./firebase-credentials.json \
                  --firebase-path disaster/nashville/zones
```

## Advanced Features

### Custom Weights

Adjust the importance of different change metrics:

```bash
# Prioritize edge changes (structural damage)
python mapping.py --before before.tif --after after.tif --out damage.geojson \
                  --weights '{"edge":0.7,"mean":0.2,"var":0.1}'

# Prioritize brightness changes (debris/smoke)
python mapping.py --before before.tif --after after.tif --out damage.geojson \
                  --weights '{"edge":0.3,"mean":0.6,"var":0.1}'
```

### Cloud Masking

Adjust cloud detection sensitivity:

```bash
# More sensitive (lower threshold)
python mapping.py --before before.tif --after after.tif --out damage.geojson \
                  --cloud-thresh 200

# Less sensitive (higher threshold)
python mapping.py --before before.tif --after after.tif --out damage.geojson \
                  --cloud-thresh 250
```

## Limitations & Future Work

### Current Limitations

1. **No Image Registration**: Assumes images are pre-aligned. For unaligned images, use external tools first.
2. **Simple Cloud Detection**: Basic threshold-based approach. Advanced methods (ML-based) could improve accuracy.
3. **Fixed Grid**: Uses regular grid. Adaptive partitioning could improve efficiency.
4. **No Temporal Filtering**: Single pair comparison. Multi-temporal analysis could reduce false positives.

### TODO: Advanced Change Detection

Replace simple difference metrics with deep learning:

```python
# TODO: Integrate Siamese neural network for change detection
# See: https://arxiv.org/abs/1810.08462
# 
# from siamese_model import SiameseChangeDetector
# 
# model = SiameseChangeDetector.load_pretrained('weights.pth')
# change_map = model.predict(before, after)
```

### TODO: Multi-scale Analysis

Add hierarchical processing for better accuracy:

```python
# TODO: Implement multi-scale pyramid for change detection
# Process at multiple resolutions (16x16, 32x32, 64x64)
# Combine results with weighted voting
```

## Troubleshooting

### Common Issues

**1. "rasterio not available" warning**
- Install rasterio: `pip install rasterio`
- Or use regular images with `--bbox` parameter

**2. "All scores are identical" warning**
- Images may be too similar (no changes)
- Try different weights or lower cloud threshold
- Verify images are actually different

**3. Coordinates in pixel space**
- For non-GeoTIFF images, provide `--bbox` parameter
- Format: `--bbox "min_lon,min_lat,max_lon,max_lat"`

**4. Out of memory errors**
- Reduce grid size: `--grid 64` or higher
- Process smaller image tiles separately

## Examples

### Example 1: Nashville Tornado Damage

```bash
python mapping.py \
    --before data/nashville_before.tif \
    --after data/nashville_after.tif \
    --out results/nashville_damage.geojson \
    --grid 32 \
    --visualize \
    --html-out results/nashville_map.html
```

### Example 2: Drone Imagery with Manual Coordinates

```bash
python mapping.py \
    --before drone_before.jpg \
    --after drone_after.jpg \
    --out damage.geojson \
    --bbox "-86.7850,36.1600,-86.7800,36.1650" \
    --grid 16 \
    --visualize
```

### Example 3: Custom Weights for Urban Areas

```bash
python mapping.py \
    --before urban_before.tif \
    --after urban_after.tif \
    --out urban_damage.geojson \
    --weights '{"edge":0.6,"mean":0.3,"var":0.1}' \
    --cloud-thresh 240
```

## License

MIT License - AWS Hackathon 2025

## Contact

For questions or issues, please contact the AWS Hackathon Team.

