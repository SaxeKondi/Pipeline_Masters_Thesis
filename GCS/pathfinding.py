import os
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString
from geopy.distance import geodesic
from python_tsp.heuristics import solve_tsp_local_search

INPUT_PATH = "coordinate_extraction/output/centroids.geojson"
OUTPUT_PATH = "pathfinding/output"

def optimize_coordinate_order(input_geojson_path=INPUT_PATH, output_dir=OUTPUT_PATH):
    os.makedirs(output_dir, exist_ok=True)

    gdf = gpd.read_file(input_geojson_path)
    if gdf.empty:
        print("No points found in GeoJSON.")
        return

    coords = [(pt.y, pt.x) for pt in gdf.geometry]
    n = len(coords)

    # Build distance matrix
    distance_matrix = np.array([
        [geodesic(coords[i], coords[j]).meters if i != j else 0 for j in range(n)]
        for i in range(n)
    ])

    # Solve TSP using local search
    order, _ = solve_tsp_local_search(distance_matrix)
    order.append(order[0])  # close the loop

    # Create route geometry
    ordered_points = [gdf.geometry[i] for i in order]
    route_line = LineString(ordered_points)

    # Save as GeoJSON
    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    route_path = os.path.join(output_dir, "shortest_route.geojson")
    route_gdf.to_file(route_path, driver="GeoJSON")
    print(f"Route saved to {route_path}")

if __name__ == "__main__":
    optimize_coordinate_order()