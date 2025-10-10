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
from typing import List, Tuple

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

def create_damage_heatmap(patch_results_with_pixels: list, image_shape: tuple) -> np.ndarray:
    """Create a damage heatmap from patch results."""
    h, w = image_shape[:2]
    damage_heatmap = np.zeros((h, w), dtype=np.float32)
    
    for patch in patch_results_with_pixels:
        x_start, y_start, x_end, y_end = patch['bbox_pixels']
        damage_score = patch['properties'].get('damage_score', 0)
        
        # Fill the patch area with damage score
        damage_heatmap[y_start:y_end, x_start:x_end] = damage_score
    
    # Apply Gaussian blur to create smooth heatmap
    sigma = 20
    ksize = int(max(3, (sigma*6)//1)) | 1  # ensure odd
    damage_heatmap = cv2.GaussianBlur(damage_heatmap, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    
    # Normalize to 0..1
    if damage_heatmap.max() > 0:
        damage_heatmap = damage_heatmap / damage_heatmap.max()
    
    return damage_heatmap

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

def fuse_heatmaps(damage_heatmap: np.ndarray, density_heatmap: np.ndarray, weights: dict) -> np.ndarray:
    """Fuse damage and density heatmaps into a single urgency heatmap."""
    w_damage = weights['damage']
    w_density = weights['density']
    
    # Ensure both heatmaps have the same shape
    if damage_heatmap.shape != density_heatmap.shape:
        # Resize the smaller one to match the larger one
        h, w = max(damage_heatmap.shape[0], density_heatmap.shape[0]), max(damage_heatmap.shape[1], density_heatmap.shape[1])
        damage_heatmap = cv2.resize(damage_heatmap, (w, h))
        density_heatmap = cv2.resize(density_heatmap, (w, h))
    
    # Weighted combination of the two heatmaps
    fused_heatmap = w_damage * damage_heatmap + w_density * density_heatmap
    
    # Normalize to 0..1
    if fused_heatmap.max() > 0:
        fused_heatmap = fused_heatmap / fused_heatmap.max()
    
    return fused_heatmap

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

def save_heatmap_image(heatmap: np.ndarray, output_path: str, colormap=cv2.COLORMAP_JET):
    """Save heatmap as a colored image."""
    # Apply gamma correction to enhance visibility
    gamma = 0.5
    heat_corrected = np.power(heatmap, gamma)
    
    # Scale to 0-255
    heat_8u = np.clip((heat_corrected * 255), 0, 255).astype('uint8')
    heat_color = cv2.applyColorMap(heat_8u, colormap)
    
    # Save the image
    cv2.imwrite(output_path, heat_color)
    print(f"[OK] Heatmap saved to: {output_path}")

# --- RESCUE ANALYSIS (NATURAL LANGUAGE) ---

def _format_coord(lat: float, lon: float) -> str:
    """Format coordinates for display with 4 decimal places."""
    return f"({lat:.4f}, {lon:.4f})"

def _categorize_urgency(urgency: float) -> str:
    """Return severity bucket label for a given urgency score."""
    if urgency >= 0.7:
        return 'Critical'
    if urgency >= 0.5:
        return 'High'
    if urgency >= 0.3:
        return 'Medium'
    return 'Low'

def _quadrant_for_point(lat: float, lon: float, center_lat: float, center_lon: float) -> str:
    ns = 'North' if lat >= center_lat else 'South'
    ew = 'East' if lon >= center_lon else 'West'
    return f"{ns}-{ew}"

def build_rescue_analysis_text(patch_features: List[dict], bbox: List[float], top_k: int = 5) -> str:
    """Build an English rescue-priority narrative from patch features.

    Args:
        patch_features: List of GeoJSON-like features with 'properties' containing
            'urgency_score', 'damage_score', 'density_score', 'lat', 'lon', 'grid_id'.
        bbox: [min_lon, min_lat, max_lon, max_lat] used for rough quadrant labeling.
        top_k: Number of top urgent areas to list explicitly.

    Returns:
        HTML string with a concise analysis paragraph and top areas.
    """
    if not patch_features:
        return (
            "<div id=\"rescue-analysis\" style=\"margin:16px 0;padding:12px;border:1px solid #e5e7eb;"
            "border-radius:8px;background:#fafafa;font-family:Arial,sans-serif;\">"
            "<h3 style=\"margin:0 0 8px 0;color:#111827;\">Rescue Priority Assessment</h3>"
            "<p style=\"margin:0;color:#374151;\">No patch features available for analysis.</p>"
            "</div>"
        )

    # Prepare data
    props_list = [f['properties'] for f in patch_features if 'properties' in f]
    valid_props = [p for p in props_list if isinstance(p.get('urgency_score', None), (int, float))]
    if not valid_props:
        return (
            "<div id=\"rescue-analysis\" style=\"margin:16px 0;padding:12px;border:1px solid #e5e7eb;"
            "border-radius:8px;background:#fafafa;font-family:Arial,sans-serif;\">"
            "<h3 style=\"margin:0 0 8px 0;color:#111827;\">Rescue Priority Assessment</h3>"
            "<p style=\"margin:0;color:#374151;\">No valid urgency scores found.</p>"
            "</div>"
        )

    # Sort by urgency descending
    sorted_props = sorted(valid_props, key=lambda p: p.get('urgency_score', 0.0), reverse=True)

    # Buckets
    total = len(sorted_props)
    counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    for p in sorted_props:
        counts[_categorize_urgency(float(p.get('urgency_score', 0.0)))] += 1

    high_critical = counts['Critical'] + counts['High']
    pct_high_critical = (high_critical / total * 100.0) if total else 0.0

    # Top-k details
    top_items = sorted_props[:max(1, top_k)]

    # Quadrant heuristic from top items
    min_lon, min_lat, max_lon, max_lat = bbox if bbox and len(bbox) == 4 else (-86.8, 36.1, -86.7, 36.2)
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    avg_lat = float(np.mean([p.get('lat', center_lat) for p in top_items]))
    avg_lon = float(np.mean([p.get('lon', center_lon) for p in top_items]))
    quadrant = _quadrant_for_point(avg_lat, avg_lon, center_lat, center_lon)

    # Build HTML content
    summary = (
        f"<p style=\"margin:6px 0;color:#374151;\">"
        f"A total of <b>{total}</b> patches were analyzed. "
        f"<b>{high_critical}</b> ({pct_high_critical:.1f}%) fall into <b>High</b> or <b>Critical</b> priority. "
        f"Highest urgency appears clustered towards the <b>{quadrant}</b> quadrant."  # heuristic
        f"</p>"
    )

    # Top list items
    list_items = []
    for idx, p in enumerate(top_items, start=1):
        grid_id = p.get('grid_id', 'N/A')
        urgency = float(p.get('urgency_score', 0.0))
        damage = float(p.get('damage_score', 0.0))
        density = float(p.get('density_score', 0.0))
        lat = float(p.get('lat', center_lat))
        lon = float(p.get('lon', center_lon))
        list_items.append(
            (
                f"<li style=\"margin:4px 0;\">"
                f"<b>#{idx} Grid {grid_id}</b> at {_format_coord(lat, lon)} — "
                f"urgency <b>{urgency:.3f}</b> (damage {damage:.3f}, density {density:.3f})"
                f"</li>"
            )
        )

    top_list_html = (
        "<div style=\"margin-top:8px;\">"
        "<div style=\"font-weight:bold;color:#111827;margin-bottom:4px;\">Top priority areas</div>"
        "<ol style=\"margin:0 0 0 18px;color:#374151;\">" + "".join(list_items) + "</ol>"
        "</div>"
    )

    # Bucket distribution line
    dist = (
        f"<p style=\"margin:6px 0;color:#6b7280;\">"
        f"Distribution — Critical: {counts['Critical']}, High: {counts['High']}, "
        f"Medium: {counts['Medium']}, Low: {counts['Low']}."
        f"</p>"
    )

    # Recommendation
    reco = (
        f"<p style=\"margin:6px 0;color:#374151;\">"
        f"Recommendation: Prioritize immediate response to <b>Critical</b> and <b>High</b> areas, "
        f"starting with the listed hotspots. Allocate assessment teams to nearby <b>Medium</b> areas to verify needs."
        f"</p>"
    )

    container = (
        "<div id=\"rescue-analysis\" "
        "style=\"margin:16px 0;padding:12px;border:1px solid #e5e7eb;border-radius:8px;"
        "background:#fafafa;font-family:Arial,sans-serif;\">"
        "<h3 style=\"margin:0 0 8px 0;color:#111827;\">Rescue Priority Assessment</h3>"
        + summary + dist + top_list_html + reco + "</div>"
    )

    return container

def inject_analysis_into_html(html_path: str, analysis_html: str) -> None:
    """Insert the analysis HTML before </html> so it appears below the map.

    Folium doesn't generate <body> tags, so we insert before </html> instead.
    """
    try:
        path = Path(html_path)
        if not path.exists():
            print(f"[WARN] HTML not found for analysis injection: {html_path}")
            return
        content = path.read_text(encoding='utf-8')
        
        # Folium doesn't generate <body> tags, so insert before </html> instead
        insert_point = content.rfind('</html>')
        if insert_point == -1:
            # Fallback: append at end
            new_content = content + analysis_html
            print("[WARN] No </html> tag found, appending to end.")
        else:
            # Insert before </html> with proper spacing
            new_content = content[:insert_point] + analysis_html + '\n' + content[insert_point:]
        
        path.write_text(new_content, encoding='utf-8')
        print("[OK] Injected rescue analysis into HTML.")
    except Exception as e:
        print(f"[WARN] Failed to inject analysis HTML: {e}")

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
        attr=" ",  # Provide an empty attribution
        height='80vh',
        width='100%'
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
    print(f"[OK] Fusion map saved to: {output_path}")

    # --- Append English rescue-priority analysis below the map ---
    try:
        analysis_html = build_rescue_analysis_text(patch_features, bbox)
        inject_analysis_into_html(str(output_path), analysis_html)
    except Exception as e:
        print(f"[WARN] Skipped analysis injection due to error: {e}")

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

    print(f"[OK] Found {len(patch_features)} patches.")

    print("\nStep 2: Running Building Density Analysis (house.py)...")
    # Load before image for building density analysis
    before_image = cv2.imread(before_path)
    if before_image is None:
        raise FileNotFoundError(f"Could not read before image at {before_path}")
    density_heatmap = create_light_gray_heatmap(before_image)
    print("[OK] Generated building density heatmap from BEFORE image.")

    print("\nStep 3: Creating Damage Heatmap...")
    # Load after image for damage heatmap dimensions
    post_image = cv2.imread(after_path)
    if post_image is None:
        raise FileNotFoundError(f"Could not read after image at {after_path}")
    # Create damage heatmap from patch results (before vs after comparison)
    damage_heatmap = create_damage_heatmap(patch_features, post_image.shape)
    print("[OK] Generated damage heatmap from BEFORE vs AFTER comparison.")

    print("\nStep 4: Fusing Heatmaps...")
    # Fuse the two heatmaps
    fusion_weights = {'damage': 0.6, 'density': 0.4}
    fused_heatmap = fuse_heatmaps(damage_heatmap, density_heatmap, fusion_weights)
    print("[OK] Fused damage and density heatmaps.")

    print("\nStep 5: Saving Heatmap Images...")
    # Save individual heatmaps
    save_heatmap_image(damage_heatmap, 'damage_heatmap.png', cv2.COLORMAP_HOT)
    save_heatmap_image(density_heatmap, 'density_heatmap.png', cv2.COLORMAP_COOL)
    save_heatmap_image(fused_heatmap, 'fused_heatmap.png', cv2.COLORMAP_JET)

    print("\nStep 6: Fusing Damage and Density data for patches...")
    # Add density score to each patch
    patches_with_density = sample_density_for_patches(density_heatmap, patch_features)
    
    # Calculate final urgency score
    final_patches = calculate_urgency(patches_with_density, fusion_weights)
    print(f"[OK] Calculated urgency scores with weights: {fusion_weights}")

    # --- START: DATA SANITIZATION STEP ---
    # Ensure all score values are valid finite floats before sending to Folium
    print("\nStep 7: Sanitizing data before visualization...")
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
        print(f"[OK] Data sanitization complete. Found and fixed {invalid_patches_found} invalid score(s).")
    else:
        print("[OK] Data sanitization complete. All scores are valid.")
    # --- END: DATA SANITIZATION STEP ---

    print("\nStep 8: Creating interactive fusion map...")
    create_fusion_map(
        patch_features=final_patches,
        post_image_path=after_path,
        bbox=args.bbox,
        output_html_path=args.out_html
    )
    print("="*60)
    print("[SUCCESS] Fusion analysis complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fuse damage and building density analysis.")
    
    # Use the 4th example pair as default
    default_before = 'test_data/mapping examples/4pre.png'
    default_after = 'test_data/mapping examples/4post.png'

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
