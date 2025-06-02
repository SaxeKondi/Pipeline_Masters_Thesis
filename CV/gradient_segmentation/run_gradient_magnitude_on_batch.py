import os
import numpy as np
from skimage import io, color, filters, img_as_float
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Config ---
GRAD_STATS_PATH = "gradient_analysis/gradient_stats/gradient_means.npy"
THRESHOLD_OFFSET = 0.12
AI_MASK_DIR = "ai_test_masks"

def load_gradient_threshold():
    dockweed_grads = np.load(GRAD_STATS_PATH)
    grad_mean = np.mean(dockweed_grads)
    return grad_mean + THRESHOLD_OFFSET

def process_gradient_chunk(image, threshold):
    gray = color.rgb2gray(img_as_float(image))
    mask_valid = gray > 0.0001
    mean_val = np.mean(gray[mask_valid])
    gray_fixed = np.where(mask_valid, gray, mean_val)

    # Compute gradient magnitude
    gradient_mag = filters.sobel(gray_fixed)
    binary_mask = (gradient_mag > threshold).astype(np.uint8) * 255

    # Morphology post-processing
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    eroded = cv2.erode(closed, np.ones((1, 2), np.uint8), iterations=2)
    eroded = cv2.erode(eroded, np.ones((2, 1), np.uint8), iterations=2)
    dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=1)

    return gradient_mag, dilated

def save_gradient_results(output_dir, chunk_id, image, gradient_mag, binary_mask):
    out_dir = os.path.join(output_dir, chunk_id)
    os.makedirs(out_dir, exist_ok=True)

    plt.imsave(os.path.join(out_dir, "gradient_magnitude.png"), gradient_mag, cmap="gray")
    plt.imsave(os.path.join(out_dir, "binary_mask.png"), binary_mask, cmap="gray")

    red_overlay = np.zeros_like(image)
    red_overlay[..., 0] = 255
    overlay_mask = binary_mask == 255
    overlay = np.where(overlay_mask[..., None], 0.7 * image + 0.3 * red_overlay, image).astype(np.uint8)
    plt.imsave(os.path.join(out_dir, "overlay.png"), overlay)

def run_gradient_pipeline(chunks_dir, output_root):
    os.makedirs(output_root, exist_ok=True)
    threshold = load_gradient_threshold()

    chunk_files = sorted(f for f in os.listdir(chunks_dir) if f.endswith(".png"))

    # --- Filter based on AI test mask availability ---
    ai_mask_files = sorted(f for f in os.listdir(AI_MASK_DIR) if f.endswith(".png"))
    valid_ids = {os.path.splitext(f)[0] for f in ai_mask_files}
    chunk_files = [f for f in chunk_files if os.path.splitext(f)[0] in valid_ids]

    print(f"Filtered to {len(chunk_files)} chunks based on AI test set.")

    for chunk_file in tqdm(chunk_files, desc="Processing gradient chunks"):
        chunk_id = os.path.splitext(chunk_file)[0]
        chunk_path = os.path.join(chunks_dir, chunk_file)
        image = io.imread(chunk_path)
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        gradient_mag, binary_mask = process_gradient_chunk(image, threshold)
        save_gradient_results(output_root, chunk_id, image, gradient_mag, binary_mask)

if __name__ == "__main__":
    run_gradient_pipeline("chunks", "gradient_analysis/intermediate_results")
