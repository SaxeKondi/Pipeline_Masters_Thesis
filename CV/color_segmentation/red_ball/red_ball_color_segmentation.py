import os
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import cv2
from color_segmentation.color_utils import mahalanobis_distance_map

# --- Config ---
COLOR_SPACE = "hsv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /color_segmentation/red_ball
COLOR_STATS_DIR = os.path.join(BASE_DIR, "color_stats", COLOR_SPACE)
INTERMEDIATE_RESULTS_DIR = os.path.join(BASE_DIR, "intermediate_results")
THRESHOLD = 0.5

def load_color_stats():
    mean_path = os.path.join(COLOR_STATS_DIR, "mean.npy")
    cov_path = os.path.join(COLOR_STATS_DIR, "cov.npy")

    if not os.path.exists(mean_path) or not os.path.exists(cov_path):
        raise FileNotFoundError(f"Missing HSV color stats in {COLOR_STATS_DIR}")
    
    mean = np.load(mean_path)
    cov = np.load(cov_path)
    return mean, cov

def process_color_chunk(image, mean, cov, threshold):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    distance_map = mahalanobis_distance_map(converted, mean, cov)
    binary_mask = (distance_map < threshold).astype(np.uint8) * 255

    # --- Morphological cleanup ---
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.dilate(binary_mask, kernel, iterations=2)

    return cleaned_mask, distance_map

def save_color_results(output_dir, binary_mask, distance_map):
    os.makedirs(output_dir, exist_ok=True)
    imsave(os.path.join(output_dir, "mask_hsv.png"), binary_mask)
    np.save(os.path.join(output_dir, "distance_hsv.npy"), distance_map)

def run_red_ball_color_pipeline(chunks_dir, output_root=INTERMEDIATE_RESULTS_DIR):
    os.makedirs(output_root, exist_ok=True)
    mean, cov = load_color_stats()

    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".png")])

    for chunk_file in tqdm(chunk_files, desc="Processing HSV color chunks"):
        chunk_id = os.path.splitext(chunk_file)[0]
        chunk_path = os.path.join(chunks_dir, chunk_file)
        image = imread(chunk_path)
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        binary_mask, distance_map = process_color_chunk(image, mean, cov, THRESHOLD)
        save_color_results(os.path.join(output_root, chunk_id), binary_mask, distance_map)

if __name__ == "__main__":
    run_red_ball_color_pipeline("chunks")
