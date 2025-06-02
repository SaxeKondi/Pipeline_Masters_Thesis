import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from tqdm import tqdm
import joblib
from texture_analysis.texture_utils import quantize_grayscale, extract_glcm_features

# --- Config ---
PATCH_SIZE = 50
STRIDE = 20
GRAY_LEVELS = 128
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
PROB_THRESHOLD = 0.9
BLACK_PIXEL_THRESHOLD = 0.02

MODEL_PATH = "texture_analysis/models/rf_texture_model.joblib"
FEATURE_PATH = "texture_analysis/models/rf_texture_features.txt"

def load_texture_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r") as f:
        features = [line.strip() for line in f]
    return model, features

def analyze_patch_texture(patch, float_patch, model, used_features):
    valid_mask = float_patch > 0.0001
    valid_ratio = np.count_nonzero(valid_mask) / patch.size
    if valid_ratio < (1 - BLACK_PIXEL_THRESHOLD):
        return None

    feats = extract_glcm_features(patch, gray_levels=GRAY_LEVELS, distances=DISTANCES, angles=ANGLES)
    vector = np.array([feats[k] for k in used_features]).reshape(1, -1)
    prob = model.predict_proba(vector)[0][1]
    return prob

def process_texture_chunk(image, model, used_features):
    gray = color.rgb2gray(image)
    quantized = quantize_grayscale(gray, gray_levels=GRAY_LEVELS)
    height, width = gray.shape

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    prob_map = np.zeros((height, width), dtype=np.float32)
    vote_map = np.zeros((height, width), dtype=np.uint8)

    for y in range(0, height - PATCH_SIZE + 1, STRIDE):
        for x in range(0, width - PATCH_SIZE + 1, STRIDE):
            patch = quantized[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            float_patch = gray[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            prob = analyze_patch_texture(patch, float_patch, model, used_features)
            if prob is None:
                continue

            if prob >= PROB_THRESHOLD:
                binary_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = 255
            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
            vote_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    valid = vote_map > 0
    prob_map[valid] /= vote_map[valid]

    return binary_mask, prob_map


def save_texture_results(output_dir, chunk_id, image, binary_mask, prob_map):
    os.makedirs(output_dir, exist_ok=True)

    plt.imsave(os.path.join(output_dir, "texture_mask.png"), binary_mask, cmap="gray")
    plt.imsave(os.path.join(output_dir, "texture_prob_map.png"), prob_map, cmap="hot")

    overlay = image.copy()
    red_overlay = np.zeros_like(image)
    red_overlay[..., 0] = 255
    mask = binary_mask == 255
    overlay = np.where(mask[..., None], 0.7 * overlay + 0.3 * red_overlay, overlay).astype(np.uint8)
    plt.imsave(os.path.join(output_dir, "overlay_texture.png"), overlay)

def run_texture_pipeline(chunks_dir, output_root):
    os.makedirs(output_root, exist_ok=True)
    model, used_features = load_texture_model_and_features()

    chunk_files = sorted([f for f in os.listdir(chunks_dir) if f.endswith(".png")])

    # --- Filter only chunks that have matching AI masks ---
    ai_mask_dir = "ai_test_masks"
    ai_mask_files = sorted([f for f in os.listdir(ai_mask_dir) if f.endswith(".png")])
    test_chunk_ids = {os.path.splitext(f)[0] for f in ai_mask_files}
    chunk_files = [f for f in chunk_files if os.path.splitext(f)[0] in test_chunk_ids]

    print(f"Filtered to {len(chunk_files)} chunks based on AI test set.")

    # --- Process each chunk ---
    for chunk_file in tqdm(chunk_files, desc="Processing texture chunks"):
        chunk_id = os.path.splitext(chunk_file)[0]
        chunk_path = os.path.join(chunks_dir, chunk_file)

        image = io.imread(chunk_path)
        if image.shape[-1] == 4:
            image = image[..., :3]

        binary_mask, prob_map = process_texture_chunk(image, model, used_features)
        save_texture_results(os.path.join(output_root, chunk_id), chunk_id, image, binary_mask, prob_map)

    print(f"Processed {len(chunk_files)} image chunks. Results saved to: {output_root}")


# Optional entry point for standalone testing
if __name__ == "__main__":
    run_texture_pipeline("chunks", "texture_analysis/intermediate_results")
