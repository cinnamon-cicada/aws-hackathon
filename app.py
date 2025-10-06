import streamlit as st
import streamlit.components.v1 as components
import json
import random

# Set page config
st.set_page_config(page_title="EMS Urgency Map - Nashville", layout="wide")

def get_urgency(lat, lon, population_density=None):
    """
    Calculate urgency level based on population density.
    Returns urgency score (0-100) and color code.
    
    In a real implementation, this would query actual population data.
    For demo purposes, we simulate varying density across Nashville.
    """
    if population_density is None:
        # Simulate population density based on proximity to downtown Nashville (36.1627, -86.7816)
        downtown_lat, downtown_lon = 36.1627, -86.7816
        distance = ((lat - downtown_lat)**2 + (lon - downtown_lon)**2)**0.5
        
        # Higher density closer to downtown
        population_density = max(0, 5000 - (distance * 10000)) + random.uniform(-500, 500)
    
    # Calculate urgency score (0-100)
    if population_density > 4000:
        urgency = 90 + random.uniform(0, 10)
        color = "#d32f2f"  # Red - Critical
    elif population_density > 2500:
        urgency = 70 + random.uniform(0, 15)
        color = "#f57c00"  # Orange - High
    elif population_density > 1000:
        urgency = 50 + random.uniform(0, 15)
        color = "#fbc02d"  # Yellow - Medium
    else:
        urgency = 20 + random.uniform(0, 20)
        color = "#388e3c"  # Green - Low
    
    return round(urgency, 1), color

# Generate sample building data for Nashville
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
        urgency, color = get_urgency(lat, lon)
        
        buildings.append({
            "lat": lat,
            "lon": lon,
            "urgency": urgency,
            "color": color,
            "name": f"Building {i+1}"
        })
    
    return buildings

# Streamlit UI
st.title("🚑 EMS Urgency Map - Nashville, TN")
st.markdown("Building-level urgency visualization based on population density")

# Sidebar controls
st.sidebar.header("Map Controls")
mapbox_token = st.sidebar.text_input(
    "Mapbox Access Token",
    type="password",
    help="Enter your Mapbox GL JS token. Get one free at mapbox.com"
)

if not mapbox_token:
    st.sidebar.warning("⚠️ Please enter a Mapbox token to view the map")
    st.info("📝 To use this app, you need a free Mapbox access token. Get yours at https://account.mapbox.com/")
    st.stop()

refresh = st.sidebar.button("🔄 Refresh Building Data")

# Legend
st.sidebar.markdown("### Urgency Levels")
st.sidebar.markdown("🔴 **Critical** (90-100): >4000 people/sq mi")
st.sidebar.markdown("🟠 **High** (70-85): 2500-4000 people/sq mi")
st.sidebar.markdown("🟡 **Medium** (50-65): 1000-2500 people/sq mi")
st.sidebar.markdown("🟢 **Low** (20-40): <1000 people/sq mi")

# Generate building data
buildings = generate_nashville_buildings()

# Convert buildings to GeoJSON
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [b["lon"], b["lat"]]
            },
            "properties": {
                "name": b["name"],
                "urgency": b["urgency"],
                "color": b["color"]
            }
        }
        for b in buildings
    ]
}

# Create Mapbox map
map_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
    <script src='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js'></script>
    <link href='https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css' rel='stylesheet' />
    <style>
        body {{ margin: 0; padding: 0; }}
        #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
        .mapboxgl-popup-content {{
            padding: 15px;
            font-family: sans-serif;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        mapboxgl.accessToken = '{mapbox_token}';
        
        const map = new mapboxgl.Map({{
            container: 'map',
            style: 'mapbox://styles/mapbox/dark-v11',
            center: [-86.7816, 36.1627],
            zoom: 12
        }});
        
        const geojsonData = {json.dumps(geojson_data)};
        
        map.on('load', () => {{
            map.addSource('buildings', {{
                type: 'geojson',
                data: geojsonData
            }});
            
            map.addLayer({{
                id: 'building-circles',
                type: 'circle',
                source: 'buildings',
                paint: {{
                    'circle-radius': [
                        'interpolate',
                        ['linear'],
                        ['zoom'],
                        10, 6,
                        15, 12
                    ],
                    'circle-color': ['get', 'color'],
                    'circle-opacity': 0.8,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#ffffff'
                }}
            }});
            
            // Add click popup
            map.on('click', 'building-circles', (e) => {{
                const coordinates = e.features[0].geometry.coordinates.slice();
                const props = e.features[0].properties;
                
                const html = `
                    <strong>${{props.name}}</strong><br>
                    <span style="color: ${{props.color}}; font-weight: bold;">
                        Urgency: ${{props.urgency}}
                    </span>
                `;
                
                new mapboxgl.Popup()
                    .setLngLat(coordinates)
                    .setHTML(html)
                    .addTo(map);
            }});
            
            map.on('mouseenter', 'building-circles', () => {{
                map.getCanvas().style.cursor = 'pointer';
            }});
            
            map.on('mouseleave', 'building-circles', () => {{
                map.getCanvas().style.cursor = '';
            }});
        }});
    </script>
</body>
</html>
"""

# Display map
components.html(map_html, height=600)

# Display statistics
st.markdown("### Current Statistics")
col1, col2, col3, col4 = st.columns(4)

critical = len([b for b in buildings if b["urgency"] >= 90])
high = len([b for b in buildings if 70 <= b["urgency"] < 90])
medium = len([b for b in buildings if 50 <= b["urgency"] < 70])
low = len([b for b in buildings if b["urgency"] < 50])

col1.metric("🔴 Critical", critical)
col2.metric("🟠 High", high)
col3.metric("🟡 Medium", medium)
col4.metric("🟢 Low", low)

# Show top urgency buildings
st.markdown("### Top 10 Highest Urgency Locations")
sorted_buildings = sorted(buildings, key=lambda x: x["urgency"], reverse=True)[:10]
for i, b in enumerate(sorted_buildings, 1):
    st.markdown(f"{i}. **{b['name']}** - Urgency: {b['urgency']} (Lat: {b['lat']:.4f}, Lon: {b['lon']:.4f})")