import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from common.split_ortho_to_chunks import split_tif_into_png_chunks

from color_segmentation.run_color_segmentation_batch import run_color_pipeline
from gradient_analysis.run_gradient_magnitude_on_batch import run_gradient_pipeline
from texture_analysis.run_texture_analysis_on_batch import run_texture_pipeline
from combination.run_combination_batch import run_combination_pipeline

from color_segmentation.red_ball.red_ball_color_segmentation import run_red_ball_color_pipeline

from coordinate_extraction.extract_centroids_and_map import coordinate_extraction
from pathfinding.pathfinding import optimize_coordinate_order
from pathfinding.forward_wpts import upload_mission_from_geojson

def main():
    # --- CONFIGURATION ---
    INPUT_TIF = "ortho/ortho_red_ball_cut.tif"
    CHUNKS_DIR = "chunks"
    CHUNK_SIZE = 512

    RED_BALL_COLOR_OUTPUT = "color_segmentation\\red_ball\\intermediate_results"
    COLOR_OUTPUT = "color_segmentation\\intermediate_results"
    GRADIENT_OUTPUT = "gradient_analysis\\intermediate_results"
    TEXTURE_OUTPUT = "texture_analysis\\intermediate_results"
    COMBINATION_OUTPUT = "combination\\intermediate_results"

    CHUNK_POSITIONS_FILE = "chunks/chunk_positions.json"

    RED_BALL_MASK_NAME = "mask_hsv.png"
    COLOR_MASK_NAME = "mask_hsv.png"
    GRADIENT_MASK_NAME = "binary_mask.png"
    TEXTURE_MASK_NAME = "texture_mask.png"
    COMBINATION_MASK_NAME = "combined_mask.png"

    OPTIMAL_PATH = "pathfinding/output/"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, INPUT_TIF)
    chunks_dir = os.path.join(current_dir, CHUNKS_DIR)

    # --- Split input into chunks ---
    split_tif_into_png_chunks(input_path, chunks_dir, CHUNK_SIZE)

    # --- Run CV pipeline on extracted chunks ---
    #run_red_ball_color_pipeline(chunks_dir, os.path.join(current_dir, RED_BALL_COLOR_OUTPUT))
    #run_color_pipeline(chunks_dir, os.path.join(current_dir, COLOR_OUTPUT)) # Using only color segmentation.
    #run_gradient_pipeline(chunks_dir, os.path.join(current_dir, GRADIENT_OUTPUT)) # Using only gradient analysis.
    # run_texture_pipeline(chunks_dir, os.path.join(current_dir, TEXTURE_OUTPUT)) # Using only texture analysis.
    run_combination_pipeline(chunks_dir, os.path.join(current_dir, COMBINATION_OUTPUT)) # Combining texture, gradient and color.


    # --- Extracting centroids from binary masks and converting to GPS coordinates ---
    coordinate_extraction(INPUT_TIF, RED_BALL_COLOR_OUTPUT,CHUNK_POSITIONS_FILE, RED_BALL_MASK_NAME)

    # --- Optimizing for shortes route between GPS coordinates ---
    optimize_coordinate_order()

    # --- Sending coordinates to UGV ---
    upload_mission_from_geojson(chunks_dir, filename="shortest_route.geojson", set_auto_mode=False)

if __name__ == "__main__":
    main()
