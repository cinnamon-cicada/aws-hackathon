"""
Enhanced visualization with actual satellite image as base layer.
Uses the 'post' (after disaster) image as the map background.
"""

import json
import numpy as np
import folium
from pathlib import Path
import base64
from PIL import Image
import io


def visualize_with_image_background(
    geojson_path: str,
    post_image_path: str,
    output_html_path: str,
    bbox: list = None
):
    """
    Visualize damage map with the actual 'post' satellite image as background.
    
    Args:
        geojson_path: Path to damage GeoJSON file
        post_image_path: Path to post-disaster image (used as base layer)
        output_html_path: Output HTML file path
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
    """
    print(f"Creating visualization with image background...")
    print(f"  GeoJSON: {geojson_path}")
    print(f"  Background image: {post_image_path}")
    
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    if not geojson_data['features']:
        print("Warning: No features in GeoJSON")
        return
    
    # Get bbox from GeoJSON if not provided
    if bbox is None:
        lats = [f['properties']['lat'] for f in geojson_data['features']]
        lons = [f['properties']['lon'] for f in geojson_data['features']]
        bbox = [min(lons), min(lats), max(lons), max(lats)]
    
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Create map centered on the area (no base tiles, only our image)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=None,  # No GPS/street map tiles
        zoom_control=True,
        scrollWheelZoom=True,
        dragging=True
    )
    
    # Load and add the post-disaster image as base layer
    img = Image.open(post_image_path)
    
    # Convert image to base64 for embedding
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Add image overlay
    folium.raster_layers.ImageOverlay(
        image=f'data:image/png;base64,{img_base64}',
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1.0,
        name='Post-Disaster Satellite Image',
        overlay=True,
        control=True
    ).add_to(m)
    
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
    
    # Add damage overlay features
    feature_group = folium.FeatureGroup(name='Damage Assessment', show=True)
    
    for feature in geojson_data['features']:
        props = feature['properties']
        
        # Skip clouds
        if props.get('cloud_mask', False):
            continue
        
        color = get_color(props['damage_score'])
        
        # Create popup with detailed info
        popup_html = f"""
        <div style="font-family: Arial; width: 300px;">
            <h4 style="margin: 5px 0; color: {color};">Grid {props['grid_id']}</h4>
            <div style="padding: 5px; background: #f0f0f0; border-radius: 3px;">
                <b style="font-size: 16px; color: {color};">Damage Score: {props['damage_score']:.3f}</b>
            </div>
            <hr style="margin: 8px 0;">
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td><b>Location:</b></td>
                    <td>{props['lat']:.5f}, {props['lon']:.5f}</td>
                </tr>
                <tr>
                    <td colspan="2"><hr style="margin: 5px 0;"></td>
                </tr>
                <tr>
                    <td><b>Before:</b></td>
                    <td>Mean={props['mean_before']:.1f}, Var={props['var_before']:.1f}</td>
                </tr>
                <tr>
                    <td><b>After:</b></td>
                    <td>Mean={props['mean_after']:.1f}, Var={props['var_after']:.1f}</td>
                </tr>
                <tr>
                    <td colspan="2"><hr style="margin: 5px 0;"></td>
                </tr>
                <tr>
                    <td><b>Edge Change:</b></td>
                    <td>{props['edge_diff']:.4f}</td>
                </tr>
            </table>
        </div>
        """
        
        # Add polygon with semi-transparent fill
        folium.GeoJson(
            feature,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': color,
                'weight': 1,
                'fillOpacity': 0.5  # Semi-transparent to see image below
            },
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(feature_group)
    
    feature_group.add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 15px; border-radius: 5px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <p style="margin: 0 0 10px 0; font-weight: bold; font-size: 16px;">Damage Level</p>
        <p style="margin: 5px 0;">
            <span style="background-color:#d32f2f; padding: 3px 10px; border-radius: 3px; color: white;">
                &nbsp;
            </span> Critical (0.8-1.0)
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color:#f57c00; padding: 3px 10px; border-radius: 3px; color: white;">
                &nbsp;
            </span> High (0.6-0.8)
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color:#fbc02d; padding: 3px 10px; border-radius: 3px;">
                &nbsp;
            </span> Medium (0.4-0.6)
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color:#8bc34a; padding: 3px 10px; border-radius: 3px; color: white;">
                &nbsp;
            </span> Low (0.2-0.4)
        </p>
        <p style="margin: 5px 0;">
            <span style="background-color:#388e3c; padding: 3px 10px; border-radius: 3px; color: white;">
                &nbsp;
            </span> Minimal (0.0-0.2)
        </p>
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 11px; color: #666;">
            🖱️ Click patches for details<br>
            🔍 Zoom in/out to see more
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = f"""
    <div style="position: fixed; 
                top: 10px; left: 50px; 
                background-color: rgba(255,255,255,0.9); 
                border:2px solid #333; z-index:9999; 
                font-size:18px; padding: 10px 20px; 
                border-radius: 5px; font-weight: bold;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        🛰️ Post-Disaster Satellite Analysis
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    
    print(f"✓ Map saved to: {output_path}")
    print(f"✓ Features: {len(geojson_data['features'])}")
    
    # Get statistics
    scores = [f['properties']['damage_score'] for f in geojson_data['features'] 
              if not f['properties']['cloud_mask']]
    if scores:
        print(f"✓ Damage scores: min={min(scores):.3f}, max={max(scores):.3f}, mean={np.mean(scores):.3f}")


def process_and_visualize_all_pairs(examples_dir: str, bbox: list, preset: str = 'default'):
    """
    Process all image pairs in the examples directory and create visualizations.
    
    Args:
        examples_dir: Directory containing image pairs
        bbox: Bounding box for all images
        preset: The analysis preset to use ('default', 'earthquake', 'typhoon', 'flood')
    """
    from mapping import process_image_pair
    
    examples_path = Path(examples_dir)
    
    # Find all pairs (1-5)
    pairs = []
    for i in range(1, 6):
        pre = examples_path / f"{i}pre.png"
        post = examples_path / f"{i}post.png"
        if pre.exists() and post.exists():
            pairs.append((i, str(pre), str(post)))
    
    print(f"Found {len(pairs)} image pairs")
    print("=" * 60)
    
    for idx, pre_path, post_path in pairs:
        print(f"\n🔄 Processing pair {idx}...")
        
        # Output paths
        geojson_path = examples_path / f"{idx}_damage.geojson"
        html_path = examples_path / f"{idx}_damage_map.html"
        
        # Process with finer grid (each original 32x32 block divided into 3x3 = 9 blocks)
        # New grid size: 32 / 3 ≈ 11 pixels per block
        print(f"  Analyzing images with finer grid (11x11 pixels per patch) using '{preset}' preset...")
        geojson = process_image_pair(
            before_path=pre_path,
            after_path=post_path,
            output_geojson_path=str(geojson_path),
            grid_size=11,  # Finer grid: 1024/11 ≈ 93 blocks per side = ~8649 total blocks
            bbox=bbox,
            normalize=True,
            preset=preset
        )
        print(f"  ✓ Generated {len(geojson['features'])} patches")
        
        # Create visualization
        print(f"  Creating map visualization...")
        visualize_with_image_background(
            geojson_path=str(geojson_path),
            post_image_path=post_path,
            output_html_path=str(html_path),
            bbox=bbox
        )
        print(f"  ✓ Map saved to: {html_path}")
        print("  " + "-" * 56)
    
    print("\n" + "=" * 60)
    print("✅ All pairs processed!")
    print(f"📂 Output location: {examples_path}")
    print("\n💡 Open the HTML files in your browser to view the maps!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process and visualize disaster image pairs.")
    parser.add_argument(
        '--preset', 
        type=str, 
        default='earthquake', 
        choices=['default', 'earthquake', 'typhoon', 'flood'],
        help="The analysis preset to use for damage calculation."
    )
    args = parser.parse_args()

    # Process all image pairs
    process_and_visualize_all_pairs(
        examples_dir='test_data/mapping examples',
        bbox=[-86.8, 36.1, -86.7, 36.2],
        preset=args.preset
    )

