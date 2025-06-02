import os
import json
import numpy as np
import rasterio
from skimage import io, measure
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# --- Config ---
ORTHO_TIF_PATH = "ortho/ortho_red_ball_cut.tif"
CHUNKS_DIR = "color_segmentation\red_ball\intermediate_results"
OUTPUT = "coordinate_extraction/output"
CHUNK_POSITIONS_FILE = "chunks/chunk_positions.json"
MASK_NAME = "mask_hsv.png"

def pixel_to_gps(transform, row, col):
    """Convert pixel row, col to GPS coordinates using rasterio transform."""
    lon, lat = rasterio.transform.xy(transform, row, col)
    return lon, lat

def find_centroids_in_mask(mask):
    """Return pixel coordinates (row, col) of centroids in a binary mask."""
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    return [(int(p.centroid[0]), int(p.centroid[1])) for p in props]

def coordinate_extraction(ortho_path=ORTHO_TIF_PATH, cv_out_dir=CHUNKS_DIR, chunk_pos_path=CHUNK_POSITIONS_FILE, mask_name=MASK_NAME):
    os.makedirs(OUTPUT, exist_ok=True)

    # Load chunk positions
    with open(chunk_pos_path, "r") as f:
        chunk_positions = json.load(f)

    with rasterio.open(ortho_path) as src:
        transform = src.transform

    gps_points = []

    chunk_folders = sorted(f for f in os.listdir(cv_out_dir) if f.startswith("chunk_"))

    for chunk_id in tqdm(chunk_folders, desc="Processing chunks"):
        chunk_path = os.path.join(cv_out_dir, chunk_id, mask_name)
        if not os.path.exists(chunk_path):
            continue

        if chunk_id not in chunk_positions:
            print(f"Warning: {chunk_id} missing in chunk_positions.json. Skipping.")
            continue

        y_offset, x_offset = chunk_positions[chunk_id]

        mask = io.imread(chunk_path)
        if mask.ndim == 3:
            mask = mask[..., 0]  # Convert RGB mask to grayscale if needed
        
        centroids = find_centroids_in_mask(mask)

        for row, col in centroids:
            global_row = y_offset + row
            global_col = x_offset + col
            lon, lat = pixel_to_gps(transform, global_row, global_col)
            gps_points.append(Point(lon, lat))

    # --- Create GeoDataFrame and save as GeoJSON ---
    gdf = gpd.GeoDataFrame(geometry=gps_points, crs="EPSG:4326")
    gdf.to_file(os.path.join(OUTPUT, "centroids.geojson"), driver="GeoJSON")
    print(f"Saved {len(gdf)} centroids to {os.path.join(OUTPUT, 'centroids.geojson')}")


if __name__ == "__main__":
    coordinate_extraction()
