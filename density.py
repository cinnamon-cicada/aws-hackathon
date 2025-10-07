# density.py
import numpy as np
# Optionally: use worldpop, rasterio, geopandas, or other population density dataset

def load_population_density(dataset_path: str):
    """
    Load population density data from a raster or vector dataset.
    Returns a function that maps a coordinate (lat, lon) -> density.
    """
    # Pseudo code: load raster/grid
    # For simplicity, let's mock a function
    density_map = {}  # {(lat, lon): density_value}

    def get_density(lat: float, lon: float) -> float:
        # In practice, interpolate the density grid here
        return density_map.get((lat, lon), 0)

    return get_density


def high_density_coordinates(coords_list, density_func, threshold: float):
    """
    Input: list of coordinates
    Output: subset with density >= threshold, sorted by density descending
    """
    urgent = [(coord, density_func(*coord)) for coord in coords_list]
    urgent = [coord for coord, d in sorted(urgent, key=lambda x: x[1], reverse=True) if d >= threshold]
    return urgent
