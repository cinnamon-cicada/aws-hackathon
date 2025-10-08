"""
fusion.py

This script fuses the outputs of mapping.py (damage assessment) and house.py (building density)
to create a comprehensive disaster analysis map.

Workflow:
1.  Calculates patch-based damage scores from before/after images.
2.  Generates a building density heatmap from the post-disaster image.
3.  Fuses these two metrics into a single 'urgency_score' for each patch.
4.  Generates a single interactive Folium map with three switchable layers:
    - Damage Assessment (colored by damage_score)
    - Building Density (colored by density_score)
    - Fused Urgency (colored by urgency_score)
"""

import cv2
import numpy as np
import folium
from pathlib import Path
import base64
from PIL import Image
import io
import argparse

# Ensure sibling modules can be imported
import sys
sys.path.append(str(Path(__file__).parent))

from mapping import process_image_pair
from house import create_light_gray_heatmap
# A small bug in visualize_with_image, let's redefine get_color here for stability
# from visualize_with_image import get_color as get_damage_color 

# --- COLORMAPS ---

def get_damage_color(damage_score: float) -> str:
    """Map damage score to color"""
    if damage_score >= 0.8: return '#d32f2f'  # Red
    elif damage_score >= 0.6: return '#f57c00'  # Orange
    elif damage_score >= 0.4: return '#fbc02d'  # Yellow
    elif damage_score >= 0.2: return '#8bc34a'  # Light green
    else: return '#388e3c'  # Green

def get_density_color(density_score: float) -> str:
    """Maps density score (0-1) to a color (e.g., plasma colormap)."""
    if density_score >= 0.8: return '#f0f921' # Yellow
    elif density_score >= 0.6: return '#fdca26' # Gold
    elif density_score >= 0.4: return '#f89540' # Orange
    elif density_score >= 0.2: return '#e16462' # Red-Orange
    else: return '#b12a90' # Purple

def get_urgency_color(urgency_score: float) -> str:
    """Maps urgency score (0-1) to a color (e.g., viridis colormap)."""
    if urgency_score >= 0.8: return '#440154' # Dark Purple
    elif urgency_score >= 0.6: return '#414487' # Indigo
    elif urgency_score >= 0.4: return '#2a7886' # Teal
    elif urgency_score >= 0.2: return '#22a884' # Green
    else: return '#7ad151' # Light Green

# --- FUSION LOGIC ---

def sample_density_for_patches(heat_map: np.ndarray, patch_results_with_pixels: list) -> list:
    """Samples the density heatmap for each patch, calculating the average density."""
    for patch in patch_results_with_pixels:
        # Use pixel bounding box from mapping results
        x_start, y_start, x_end, y_end = patch['bbox_pixels']
        
        # Extract the corresponding region from the heatmap
        patch_density_region = heat_map[y_start:y_end, x_start:x_end]
        
        # Calculate the mean density for the patch
        avg_density = np.mean(patch_density_region) if patch_density_region.size > 0 else 0
        
        # Add to the patch's data
        patch['properties']['density_score'] = float(avg_density)
        
    return patch_results_with_pixels

def calculate_urgency(patch_results: list, weights: dict) -> list:
    """Calculates the fused urgency score for each patch."""
    w_damage = weights['damage']
    w_density = weights['density']
    
    for patch in patch_results:
        props = patch['properties']
        damage = props.get('damage_score', 0)
        density = props.get('density_score', 0)
        
        # Weighted sum for urgency
        urgency = w_damage * damage + w_density * density
        props['urgency_score'] = np.clip(urgency, 0, 1)
        
    return patch_results

# --- VISUALIZATION ---

def create_fusion_map(patch_features: list, post_image_path: str, bbox: list, output_html_path: str):
    """Creates a Folium map with three switchable layers."""
    print(f"Creating fused visualization...")
    
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Create map using a transparent tile layer as a base.
    # This ensures the map initializes correctly without showing any unwanted background.
    transparent_tile = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles=transparent_tile,
        attr=" "  # Provide an empty attribution
    )
    
    # --- Create FeatureGroups for all switchable layers ---
    fg_satellite = folium.FeatureGroup(name='Post-Disaster Satellite Image', show=True)
    fg_damage = folium.FeatureGroup(name='Damage Assessment (Mapping)', show=False)
    fg_density = folium.FeatureGroup(name='Building Density (House)', show=False)
    fg_urgency = folium.FeatureGroup(name='Fused Urgency (Fusion)', show=True) # Show by default

    # --- Add content to their respective groups ---

    # Layer 1: Satellite Image (using external file path instead of base64 for performance)
    folium.raster_layers.ImageOverlay(
        image=post_image_path,  # Use file path directly instead of base64 encoding
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=1.0,
        name='Satellite Base Image' # Name here is just for reference, not for LayerControl
    ).add_to(fg_satellite)

    # --- Prepare a single FeatureCollection for all patches ---
    # This is much more efficient than creating one GeoJson object per patch.
    all_features_geojson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'geometry': patch['geometry'],
                'properties': patch['properties']
            }
            for patch in patch_features
        ]
    }

    # --- Add the single FeatureCollection to each analysis layer with different styles ---
    
    # Define a tooltip/popup function that will be shared by all layers
    def create_popup_html(props):
        """Create a lightweight HTML popup for a patch"""
        return f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">Grid {props['grid_id']}</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 3px; font-weight: bold;">Urgency:</td>
                    <td style="padding: 3px; color: {get_urgency_color(props['urgency_score'])};">
                        {props['urgency_score']:.3f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 3px; font-weight: bold;">Damage:</td>
                    <td style="padding: 3px; color: {get_damage_color(props['damage_score'])};">
                        {props['damage_score']:.3f}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 3px; font-weight: bold;">Density:</td>
                    <td style="padding: 3px; color: {get_density_color(props['density_score'])};">
                        {props['density_score']:.3f}
                    </td>
                </tr>
                <tr><td colspan="2"><hr style="margin: 5px 0;"></td></tr>
                <tr>
                    <td style="padding: 3px; font-size: 0.9em;">Location:</td>
                    <td style="padding: 3px; font-size: 0.9em;">{props['lat']:.4f}, {props['lon']:.4f}</td>
                </tr>
            </table>
        </div>
        """
    
    # Create a tooltip function for onEachFeature
    def add_popup(feature, layer):
        popup_html = create_popup_html(feature['properties'])
        layer.bindPopup(popup_html, max_width=300)
    
    # Layer 2: Damage Assessment
    folium.GeoJson(
        all_features_geojson,
        style_function=lambda feature: {
            'fillColor': get_damage_color(feature['properties']['damage_score']),
            'color': get_damage_color(feature['properties']['damage_score']),
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=['grid_id'], aliases=['Grid ID:']),
        popup=folium.GeoJsonPopup(fields=['grid_id', 'urgency_score', 'damage_score', 'density_score']),
        name='Damage Assessment Layer'
    ).add_to(fg_damage)
    
    # Layer 3: Building Density
    folium.GeoJson(
        all_features_geojson,
        style_function=lambda feature: {
            'fillColor': get_density_color(feature['properties']['density_score']),
            'color': get_density_color(feature['properties']['density_score']),
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=['grid_id'], aliases=['Grid ID:']),
        popup=folium.GeoJsonPopup(fields=['grid_id', 'urgency_score', 'damage_score', 'density_score']),
        name='Building Density Layer'
    ).add_to(fg_density)
    
    # Layer 4: Fused Urgency
    folium.GeoJson(
        all_features_geojson,
        style_function=lambda feature: {
            'fillColor': get_urgency_color(feature['properties']['urgency_score']),
            'color': get_urgency_color(feature['properties']['urgency_score']),
            'weight': 1,
            'fillOpacity': 0.6
        },
        tooltip=folium.GeoJsonTooltip(fields=['grid_id'], aliases=['Grid ID:']),
        popup=folium.GeoJsonPopup(fields=['grid_id', 'urgency_score', 'damage_score', 'density_score']),
        name='Fused Urgency Layer'
    ).add_to(fg_urgency)


    # --- Add all FeatureGroups to the map ---
    m.add_child(fg_satellite)
    m.add_child(fg_damage)
    m.add_child(fg_density)
    m.add_child(fg_urgency)
    
    folium.LayerControl(position='topright', collapsed=False).add_to(m)
    
    # Save map
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"✓ Fusion map saved to: {output_path}")

# --- MAIN EXECUTION ---

def main(args):
    """Main pipeline to run the fusion analysis."""
    
    before_path = args.before
    after_path = args.after
    
    print("="*60)
    print("Step 1: Running Damage Assessment (mapping.py)...")
    damage_geojson, patch_results_raw = process_image_pair(
        before_path=before_path,
        after_path=after_path,
        output_geojson_path=args.out_geojson,
        grid_size=args.grid,
        bbox=args.bbox,
        preset=args.preset,
        normalize=True
    )
    # The raw results now contain pixel info, let's combine it with geojson features
    patch_features = damage_geojson['features']
    for i, feature in enumerate(patch_features):
        feature.update(patch_results_raw[i])

        # Remove the non-serializable PatchFeatures object before visualization
        if 'features' in feature:
            del feature['features']

    print(f"✓ Found {len(patch_features)} patches.")

    print("\nStep 2: Running Building Density Analysis (house.py)...")
    post_image = cv2.imread(after_path)
    if post_image is None:
        raise FileNotFoundError(f"Could not read post-disaster image at {after_path}")
    density_heatmap = create_light_gray_heatmap(post_image)
    print("✓ Generated density heatmap.")

    print("\nStep 3: Fusing Damage and Density data...")
    # Add density score to each patch
    patches_with_density = sample_density_for_patches(density_heatmap, patch_features)
    
    # Calculate final urgency score
    fusion_weights = {'damage': 0.6, 'density': 0.4}
    final_patches = calculate_urgency(patches_with_density, fusion_weights)
    print(f"✓ Calculated urgency scores with weights: {fusion_weights}")

    # --- START: DATA SANITIZATION STEP ---
    # Ensure all score values are valid finite floats before sending to Folium
    print("\nStep 3.5: Sanitizing data before visualization...")
    invalid_patches_found = 0
    for patch in final_patches:
        props = patch['properties']
        for key in ['damage_score', 'density_score', 'urgency_score']:
            if key in props:
                # Check for None, NaN, or Infinity and replace with 0.0
                if props[key] is None or not np.isfinite(props[key]):
                    if invalid_patches_found < 5: # Log first few examples
                        print(f"  WARNING: Found invalid value '{props[key]}' for {key} in grid {props['grid_id']}. Replacing with 0.0.")
                    props[key] = 0.0
                    invalid_patches_found += 1
    if invalid_patches_found > 0:
        print(f"✓ Data sanitization complete. Found and fixed {invalid_patches_found} invalid score(s).")
    else:
        print("✓ Data sanitization complete. All scores are valid.")
    # --- END: DATA SANITIZATION STEP ---

    print("\nStep 4: Creating interactive fusion map...")
    create_fusion_map(
        patch_features=final_patches,
        post_image_path=after_path,
        bbox=args.bbox,
        output_html_path=args.out_html
    )
    print("="*60)
    print("✅ Fusion analysis complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fuse damage and building density analysis.")
    
    # Use the first example pair as default
    default_before = 'test_data/mapping examples/2pre.png'
    default_after = 'test_data/mapping examples/2post.png'

    parser.add_argument('--before', type=str, default=default_before, help='Path to before image.')
    parser.add_argument('--after', type=str, default=default_after, help='Path to after image.')
    parser.add_argument('--out_html', type=str, default='fusion_map.html', help='Output path for the fused HTML map.')
    parser.add_argument('--out_geojson', type=str, default='damage_assessment.geojson', help='Output for the intermediate damage GeoJSON.')
    parser.add_argument('--grid', type=int, default=11, help='Grid size for patch analysis.')
    parser.add_argument('--preset', type=str, default='earthquake', choices=['default', 'earthquake', 'typhoon', 'flood'], help='Damage analysis preset.')
    
    # Bbox needs to be parsed from string to list of floats
    parser.add_argument('--bbox', type=lambda s: [float(item) for item in s.split(',')], 
                        default='-86.8,36.1,-86.7,36.2', 
                        help='Bounding box as "min_lon,min_lat,max_lon,max_lat"')
    
    args = parser.parse_args()
    main(args)
