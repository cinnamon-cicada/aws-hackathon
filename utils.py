# Generate sample building data for Nashville
import random

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
