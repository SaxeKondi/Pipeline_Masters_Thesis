import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import sobel
from tqdm import tqdm
import joblib
import cv2

from texture_analysis.texture_utils import quantize_grayscale, extract_glcm_features
from color_segmentation.color_utils import mahalanobis_distance_map

# --- Config ---
PATCH_SIZE = 50
STRIDE = 20
COLOR_SPACE = "lab"
COLOR_THRESHOLD = 4.2 # 3.6
GRAD_OFFSET = -0.035 # -0.025
PROB_THRESHOLD = 0.9

GRAY_LEVELS = 128
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]

MODEL_PATH = "texture_analysis/models/rf_texture_model.joblib"
FEATURE_PATH = "texture_analysis/models/rf_texture_features.txt"
COLOR_MEAN_PATH = f"combination/color_stats/{COLOR_SPACE}/mean.npy"
COLOR_COV_PATH = f"combination/color_stats/{COLOR_SPACE}/cov.npy"
GRADIENT_MEAN_PATH = "gradient_analysis/gradient_stats/gradient_means.npy"
AI_MASK_DIR = "ai_test_masks"

def load_models_and_stats():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH) as f:
        texture_features = [line.strip() for line in f]

    color_mean = np.load(COLOR_MEAN_PATH)
    color_cov = np.load(COLOR_COV_PATH)

    raw_grad_mean = np.mean(np.load(GRADIENT_MEAN_PATH))
    grad_threshold = raw_grad_mean + GRAD_OFFSET

    return model, texture_features, color_mean, color_cov, grad_threshold

def process_chunk(image, model, texture_features, color_mean, color_cov, grad_threshold):
    height, width = image.shape[:2]
    gray = color.rgb2gray(image)
    quantized = quantize_grayscale(gray)

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    prob_map = np.zeros((height, width), dtype=np.float32)
    vote_map = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch_rgb = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_gray = gray[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_quant = quantized[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            if patch_gray.mean() < 0.01:
                continue

            # Color check
            if COLOR_SPACE == "lab":
                converted = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2Lab)
            elif COLOR_SPACE == "hsv":
                converted = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
            else:
                converted = patch_rgb

            color_dist = mahalanobis_distance_map(converted, color_mean, color_cov)
            if np.mean(color_dist) > COLOR_THRESHOLD:
                continue

            # Gradient check
            grad = sobel(patch_gray)
            if grad.mean() < grad_threshold:
                continue

            # GLCM + prediction
            feats = extract_glcm_features(patch_quant, gray_levels=GRAY_LEVELS, distances=DISTANCES, angles=ANGLES)
            vec = np.array([feats[k] for k in texture_features]).reshape(1, -1)
            prob = model.predict_proba(vec)[0][1]

            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
            vote_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

            if prob >= PROB_THRESHOLD:
                binary_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255

    valid = vote_map > 0
    prob_map[valid] /= vote_map[valid]

    return binary_mask, prob_map

def save_outputs(chunk_id, image, binary_mask, prob_map, output_root):
    out_dir = os.path.join(output_root, chunk_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save binary mask
    plt.imsave(os.path.join(out_dir, "texture_mask.png"), binary_mask, cmap="gray")

    # Save probability heatmap
    plt.imsave(os.path.join(out_dir, "texture_prob_map.png"), prob_map, cmap="hot")

    # Save overlay image
    overlay = image.copy()
    red_overlay = np.zeros_like(image)
    red_overlay[..., 0] = 255
    mask = binary_mask == 255
    overlay = np.where(mask[..., None], 0.7 * overlay + 0.3 * red_overlay, overlay).astype(np.uint8)
    plt.imsave(os.path.join(out_dir, "overlay_texture.png"), overlay)

def run_combination_pipeline(chunks_dir="chunks", output_root="combination/intermediate_results"):
    os.makedirs(output_root, exist_ok=True)

    model, texture_features, color_mean, color_cov, grad_threshold = load_models_and_stats()

    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".png")])

    # Filter based on available AI test masks
    ai_mask_files = sorted([f for f in os.listdir(AI_MASK_DIR) if f.endswith(".png")])
    valid_ids = {os.path.splitext(f)[0] for f in ai_mask_files}
    chunk_files = [f for f in chunk_files if os.path.splitext(f)[0] in valid_ids]

    print(f"Filtered to {len(chunk_files)} chunks based on AI test set.")

    for chunk_file in tqdm(chunk_files, desc="Running combination batch"):
        chunk_id = os.path.splitext(chunk_file)[0]
        chunk_path = os.path.join(chunks_dir, chunk_file)

        image = io.imread(chunk_path)
        if image.shape[-1] == 4:
            image = image[..., :3]

        binary_mask, prob_map = process_chunk(image, model, texture_features, color_mean, color_cov, grad_threshold)
        save_outputs(chunk_id, image, binary_mask, prob_map, output_root)

    print(f"Processed {len(chunk_files)} chunk images.")

if __name__ == "__main__":
    run_combination_pipeline()
