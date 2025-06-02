import os
import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm
import cv2
from color_segmentation.color_utils import mahalanobis_distance_map

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
COLOR_STATS_DIR = "color_segmentation/color_stats"
COLOR_SPACES = ["rgb", "hsv", "lab"]
THRESHOLDS = {
    "rgb": 2.0,
    "hsv": 2.0,
    "lab": 2.0
}
AI_MASK_DIR = "ai_test_masks"

def load_color_stats(cs):
    mean_path = os.path.join(COLOR_STATS_DIR, cs, "mean.npy")
    cov_path = os.path.join(COLOR_STATS_DIR, cs, "cov.npy")

    if not os.path.exists(mean_path) or not os.path.exists(cov_path):
        raise FileNotFoundError(f"Missing color stats for {cs}")
    
    mean = np.load(mean_path)
    cov = np.load(cov_path)
    return mean, cov

def process_color_chunk(image, color_space, mean, cov, threshold):
    if color_space == "rgb":
        converted = image
    elif color_space == "hsv":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == "lab":
        converted = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    distance_map = mahalanobis_distance_map(converted, mean, cov)
    binary_mask = (distance_map < threshold).astype(np.uint8) * 255

    overlay = image.astype(np.float32)
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 255
    alpha_mask = (binary_mask == 255)[..., np.newaxis]
    overlay = np.where(alpha_mask, 0.7 * overlay + 0.3 * red_layer, overlay).astype(np.uint8)

    return binary_mask, overlay, distance_map

def save_color_results(output_dir, chunk_id, color_space, binary_mask, overlay, distance_map):
    os.makedirs(output_dir, exist_ok=True)
    imsave(os.path.join(output_dir, f"mask_{color_space}.png"), binary_mask)
    imsave(os.path.join(output_dir, f"overlay_{color_space}.png"), overlay)
    np.save(os.path.join(output_dir, f"distance_{color_space}.npy"), distance_map)

def run_color_pipeline(chunks_dir, output_root):
    os.makedirs(output_root, exist_ok=True)

    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".png")])
    
    # Filter using AI mask set
    ai_mask_files = sorted([f for f in os.listdir(AI_MASK_DIR) if f.endswith(".png")])
    valid_chunk_ids = {os.path.splitext(f)[0] for f in ai_mask_files}
    chunk_files = [f for f in chunk_files if os.path.splitext(f)[0] in valid_chunk_ids]

    print(f"Filtered to {len(chunk_files)} chunks based on AI test set.")

    for chunk_file in tqdm(chunk_files, desc="Processing color chunks"):
        chunk_id = os.path.splitext(chunk_file)[0]
        chunk_path = os.path.join(chunks_dir, chunk_file)
        image = imread(chunk_path)
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        for cs in COLOR_SPACES:
            try:
                mean, cov = load_color_stats(cs)
            except FileNotFoundError:
                print(f"Skipping color space {cs} due to missing stats.")
                continue

            threshold = THRESHOLDS[cs]
            binary_mask, overlay, distance_map = process_color_chunk(image, cs, mean, cov, threshold)
            save_color_results(os.path.join(output_root, chunk_id), chunk_id, cs, binary_mask, overlay, distance_map)

if __name__ == "__main__":
    run_color_pipeline("chunks", "color_segmentation/intermediate_results")
