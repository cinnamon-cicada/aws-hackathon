import streamlit as st
import streamlit.components.v1 as components
import json, random, time, base64
from density import load_population_density, high_density_coordinates
from utils import generate_nashville_buildings
from alert_system import alert_system, trigger_100_level_alert
from human_detection import detect_human_heatmap_points

# ───────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ───────────────────────────────────────────────
st.set_page_config(page_title="EMS Urgency Map - Nashville", layout="wide")

# ───────────────────────────────────────────────
# URGENCY FUNCTION
# ───────────────────────────────────────────────
def get_urgency(lat, lon, population_density=None, alert_severity=None):
    """Calculate urgency level based on population density and alert severity."""
    if population_density is None:
        downtown_lat, downtown_lon = 36.0331, -86.7828
        distance = ((lat - downtown_lat)**2 + (lon - downtown_lon)**2)**0.5
        population_density = max(0, 5000 - (distance * 10000)) + random.uniform(-500, 500)

    # Use alert severity if available, otherwise calculate from population density
    if alert_severity is not None:
        urgency = alert_severity
        if urgency >= 100:
            color = "#9c27b0"  # 🟣 Purple for human-detected alerts
        elif urgency >= 90:
            color = "#d32f2f"  # 🔴 Red
        elif urgency >= 70:
            color = "#f57c00"  # 🟠 Orange
        elif urgency >= 50:
            color = "#fbc02d"  # 🟡 Yellow
        else:
            color = "#388e3c"  # 🟢 Green
    elif population_density > 4000:
        urgency = 90 + random.uniform(0, 10)
        color = "#d32f2f"
    elif population_density > 2500:
        urgency = 70 + random.uniform(0, 15)
        color = "#f57c00"
    elif population_density > 1000:
        urgency = 50 + random.uniform(0, 15)
        color = "#fbc02d"
    else:
        urgency = 20 + random.uniform(0, 20)
        color = "#388e3c"
    return round(urgency, 1), color


# ───────────────────────────────────────────────
# ALERT SIMULATION
# ───────────────────────────────────────────────
def simulate_alert_updates():
    """Run real detection once and update coordinates/alerts accordingly."""
    try:
        points = detect_human_heatmap_points()
        if points:
            st.toast("New Alert: Human detected", icon="🚨")
            heat_key = 'human_heat_points'
            if heat_key not in st.session_state:
                st.session_state[heat_key] = []
            st.session_state[heat_key].extend(points)
            if len(st.session_state[heat_key]) > 500:
                st.session_state[heat_key] = st.session_state[heat_key][-500:]
            
            from alert_system import trigger_100_level_alert
            import time
            for i, point in enumerate(points):
                alert_system.update_drone_coordinates(point['lat'], point['lon'], altitude=100.0)
                time.sleep(0.001)
                trigger_100_level_alert(human_detected=True, population_density=4000)
            
            st.sidebar.info(f"Human heat points: {len(st.session_state[heat_key])} (added {len(points)})")
        else:
            st.sidebar.info(f"Human heat points: {len(st.session_state.get('human_heat_points', []))}")
    except Exception:
        st.sidebar.info(f"Human heat points: {len(st.session_state.get('human_heat_points', []))}")

# ───────────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────────
st.title("🚑 EMS Urgency Map - Brentwood, TN")
st.markdown("Building-level urgency visualization based on population density and live human detection")

# Sidebar
st.sidebar.header("Map Controls")
mapbox_token = st.sidebar.text_input(
    "Mapbox Access Token",
    type="password",
    help="Enter your Mapbox GL JS token. Get one free at mapbox.com"
)
if not mapbox_token:
    st.sidebar.warning("⚠️ Please enter a Mapbox token to view the map")
    st.stop()

refresh = st.sidebar.button("🔄 Refresh Alert Data")
if refresh:
    simulate_alert_updates()
    st.rerun()

# Urgency legend
st.sidebar.markdown("### Urgency Levels")
st.sidebar.markdown("🔴 **Critical** (90-100): High density or alerts")
st.sidebar.markdown("🟠 **High** (70-85): 2500-4000 people/sq mi")
st.sidebar.markdown("🟡 **Medium** (50-65): 1000-2500 people/sq mi")
st.sidebar.markdown("🟢 **Low** (20-40): <1000 people/sq mi")
st.sidebar.markdown("🟣 **Survivor:** Potential survivor detected.")

# ───────────────────────────────────────────────
# GENERATE ALERT LOCATIONS (buildings + alerts)
# ───────────────────────────────────────────────
# alert_locations = generate_nashville_buildings()
alert_locations = []

# Get active alerts and add them to locations
active_alerts = alert_system.get_active_alerts()
for alert in active_alerts:
    survivor = any("Human detected" in cond for cond in alert.get('conditions', []))
    alert_locations.append({
        "lat": alert['coordinates']['lat'],
        "lon": alert['coordinates']['lon'],
        "name": f"Alert {alert['id']}",
        "alert_id": alert['id'],
        "alert_type": alert['type'],
        "alert_severity": alert['severity'],
        "survivor": survivor
    })

# Assign urgency levels to all locations
for location in alert_locations:
    if 'alert_severity' in location:
        urgency, color = get_urgency(location["lat"], location["lon"], alert_severity=location['alert_severity'])
    else:
        urgency, color = get_urgency(location["lat"], location["lon"])

    # Survivor detected: force purple color and urgency 100
    if location.get('survivor'):
        urgency = 100
        color = "#9c27b0"

    location["urgency"] = urgency
    location["color"] = color

# ───────────────────────────────────────────────
# GEOJSON CONVERSION
# ───────────────────────────────────────────────
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [loc["lon"], loc["lat"]]},
            "properties": {
                "name": loc["name"], 
                "urgency": loc["urgency"], 
                "color": loc["color"],
                "is_alert": "alert_id" in loc,
                "alert_id": loc.get("alert_id", ""),
                "alert_type": loc.get("alert_type", ""),
                "survivor": loc.get("survivor", False)
            }
        }
        for loc in alert_locations
    ]
}

# ───────────────────────────────────────────────
# MAPBOX MAP
# ───────────────────────────────────────────────
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
        .mapboxgl-popup-content {{ padding: 15px; font-family: sans-serif; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <script>
        mapboxgl.accessToken = '{mapbox_token}';
        const map = new mapboxgl.Map({{
            container: 'map',
            style: 'mapbox://styles/mapbox/dark-v11',
            center: [-86.7828, 36.0331],
            zoom: 12
        }});
        const geojsonData = {json.dumps(geojson_data)};
        const humanHeatPoints = {json.dumps([
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
                "properties": {"weight": p["weight"]}
            } for p in st.session_state.get('human_heat_points', [])
        ])};
        const humanHeatData = {{"type": "FeatureCollection", "features": humanHeatPoints}};
        map.on('load', () => {{
            map.addSource('background-heatmap', {{
                type: 'image',
                url: 'data:image/png;base64,{base64.b64encode(open("assets/background.png", "rb").read()).decode()}',
                coordinates: [
                    [-86.85, 36.10],
                    [-86.70, 36.10],
                    [-86.70, 35.95],
                    [-86.85, 35.95]
                ]
            }});
            
            map.addLayer({{
                id: 'background-heatmap-layer',
                type: 'raster',
                source: 'background-heatmap',
                paint: {{
                    'raster-opacity': 0.6
                }}
            }});
            
            map.addSource('buildings', {{ type: 'geojson', data: geojsonData }});
            // Human detection heatmap source
            map.addSource('human-heat', {{ type: 'geojson', data: humanHeatData }});
            // Heatmap layer under points
            map.addLayer({{
                id: 'human-detection-heatmap',
                type: 'heatmap',
                source: 'human-heat',
                maxzoom: 16,
                paint: {{
                    'heatmap-weight': ['coalesce', ['get', 'weight'], 0.4],
                    'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 10, 2.0, 15, 5.0],
                    'heatmap-color': [
                        'interpolate', ['linear'], ['heatmap-density'],
                        0.0, 'rgba(156,39,176,0.0)',
                        0.1, 'rgba(156,39,176,0.4)',
                        0.2, 'rgba(156,39,176,0.6)',
                        0.4, 'rgba(211,47,47,0.7)',
                        0.6, 'rgba(245,124,0,0.8)',
                        0.8, 'rgba(251,192,45,0.9)',
                        1.0, 'rgba(255,255,255,1.0)'
                    ],
                    'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 10, 20, 15, 40],
                    'heatmap-opacity': 1.0
                }}
            }});
            
            // Add purple circles for individual human detections
            map.addLayer({{
                id: 'human-detection-points',
                type: 'circle',
                source: 'human-heat',
                paint: {{
                    'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 2, 15, 4],
                    'circle-color': '#9c27b0',
                    'circle-opacity': 0.8,
                    'circle-stroke-width': 1,
                    'circle-stroke-color': '#ffffff'
                }}
            }});
            map.addLayer({{
                id: 'building-circles',
                type: 'circle',
                source: 'buildings',
                paint: {{
                    'circle-radius': ['interpolate',['linear'],['zoom'],10,3,15,6],
                    'circle-color': ['get', 'color'],
                    'circle-opacity': 0.8,
                    'circle-stroke-width': 1,
                    'circle-stroke-color': '#ffffff'
                }}
            }});
            // Click handler for human detection points
            map.on('click', 'human-detection-points', (e) => {{
                const c = e.features[0].geometry.coordinates.slice();
                const p = e.features[0].properties;
                const html = `
                    <div style="font-family: Arial; width: 200px;">
                        <h4 style="color: #9c27b0; margin: 5px 0;">🟣 Human Detected</h4>
                        <div style="padding: 5px; background: #f0f0f0; border-radius: 3px;">
                            <b>Confidence: ${{(p.weight * 100).toFixed(1)}}%</b>
                        </div>
                        <hr style="margin: 8px 0;">
                        <div style="font-size: 12px; color: #666;">
                            Location: ${{c[1].toFixed(4)}}, ${{c[0].toFixed(4)}}
                        </div>
                    </div>
                `;
                new mapboxgl.Popup().setLngLat(c).setHTML(html).addTo(map);
            }});
            
            map.on('click', 'building-circles', (e) => {{
                const c = e.features[0].geometry.coordinates.slice();
                const p = e.features[0].properties;
                let html = `<strong>${{p.name}}</strong><br><span style="color:${{p.color}};font-weight:bold;">Urgency: ${{p.urgency}}</span>`;
                if (p.is_alert) {{
                    if (p.survivor) {{
                        html += `<br><span style="color:#9c27b0;font-weight:bold;">Survivor detected</span>`;
                    }} else {{
                        html += `<br><span style="color:#d32f2f;font-weight:bold;">ALERT: ${{p.alert_type}}</span>`;
                    }}
                }}
                new mapboxgl.Popup().setLngLat(c).setHTML(html).addTo(map);
            }});
        }});
    </script>
</body>
</html>
"""
components.html(map_html, height=600)

# ───────────────────────────────────────────────
# ALERT STATUS
# ───────────────────────────────────────────────
st.markdown("### 🚨 Alert Status")
alert_summary = alert_system.get_alert_summary()
col1, col2, col3 = st.columns(3)
survivor_alerts = len([alert for alert in active_alerts if any("Human detected" in cond for cond in alert.get('conditions', []))])
col1.metric("🟣 Active Survivor Alerts", len(st.session_state.get('human_heat_points', [])))
col2.metric("📊 Total Alerts Today", alert_summary['total_alerts'])
if alert_summary['drone_coordinates']:
    col3.metric("📍 Drone Location", f"{alert_summary['drone_coordinates']['lat']:.4f}, {alert_summary['drone_coordinates']['lon']:.4f}")

# Display active alerts
if active_alerts:
    st.markdown("#### Active Alerts:")
    for alert in active_alerts:
        with st.expander(f"🚨 Alert {alert['id']} - {alert['timestamp']}"):
            st.write(f"**Type:** {alert['type']}")
            st.write(f"**Severity:** {alert['severity']}")
            st.write(f"**Location:** {alert['coordinates']['lat']:.4f}, {alert['coordinates']['lon']:.4f}")
            st.write(f"**Conditions:** {', '.join(alert['conditions'])}")
            if st.button(f"Resolve Alert {alert['id']}", key=f"resolve_{alert['id']}"):
                alert_system.resolve_alert(alert['id'])
                st.rerun()

# ───────────────────────────────────────────────
# STATISTICS
# ───────────────────────────────────────────────
st.markdown("### Current Statistics")
col1, col2, col3, col4 = st.columns(4)
critical = len([loc for loc in alert_locations if loc["urgency"] >= 90])
high = len([loc for loc in alert_locations if 70 <= loc["urgency"] < 90])
medium = len([loc for loc in alert_locations if 50 <= loc["urgency"] < 70])
low = len([loc for loc in alert_locations if loc["urgency"] < 50])
col1.metric("🔴 Critical", critical)
col2.metric("🟠 High", high)
col3.metric("🟡 Medium", medium)
col4.metric("🟢 Low", low)

# ───────────────────────────────────────────────
# TOP URGENCY LOCATIONS
# ───────────────────────────────────────────────
st.markdown("### Top 10 Highest Urgency Locations")
sorted_locations = sorted(alert_locations, key=lambda x: x["urgency"], reverse=True)[:10]
for i, loc in enumerate(sorted_locations, 1):
    alert_indicator = " 🚨" if "alert_id" in loc else ""
    st.markdown(f"{i}. **{loc['name']}**{alert_indicator} - Urgency: {loc['urgency']} (Lat: {loc['lat']:.4f}, Lon: {loc['lon']:.4f})")

# ───────────────────────────────────────────────
# MANUAL REFRESH ONLY
# ───────────────────────────────────────────────
# Alert updates are handled manually via the refresh button
