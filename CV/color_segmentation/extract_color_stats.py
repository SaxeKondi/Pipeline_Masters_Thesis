import os
import numpy as np
from skimage.io import imread, imsave
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
CHUNK_IDS = [2, 47, 453, 498]  # Labeled chunk IDs
CHUNK_DIR = "chunks"
MASK_DIR = "color_segmentation/mask_imgs"
OUTPUT_DIR = "color_segmentation/color_stats"
COLOR_SPACES = ["rgb", "hsv", "lab"]
DOCKWEED_COLOR_RGB = (0, 0, 255)
TOL = 5

# --- Setup ---
pixels_by_space = {}
for cs in COLOR_SPACES:
    cs_dir = os.path.join(OUTPUT_DIR, cs)
    os.makedirs(cs_dir, exist_ok=True)
    os.makedirs(os.path.join(cs_dir, "overlays"), exist_ok=True)
    pixels_by_space[cs] = []

# --- Process Chunks ---
for chunk_id in tqdm(CHUNK_IDS, desc="Processing labeled chunks"):
    img_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_id:04d}.png")
    mask_path = os.path.join(MASK_DIR, f"chunk_{chunk_id:04d}.png")

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"Skipping chunk {chunk_id} - missing image or mask.")
        continue

    img = imread(img_path)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    mask = imread(mask_path)
    if img.shape[:2] != mask.shape[:2]:
        print(f"Skipping chunk {chunk_id} - image and mask size mismatch.")
        continue

    # Binary mask: where mask is pure blue (RGB)
    diff = np.abs(mask[:, :, :3].astype(int) - DOCKWEED_COLOR_RGB)
    mask_blue = np.all(diff <= TOL, axis=-1)

    if not np.any(mask_blue):
        print(f"Warning: No blue pixels found in mask for chunk {chunk_id}")
        continue

    # --- Save visual overlay for verification ---
    overlay = img.copy().astype(np.float32)
    red_layer = np.zeros_like(overlay)
    red_layer[..., 0] = 255  # Red
    alpha = mask_blue[..., np.newaxis]
    overlay = np.where(alpha, 0.7 * overlay + 0.3 * red_layer, overlay).astype(np.uint8)

    for cs in COLOR_SPACES:
        cs_dir = os.path.join(OUTPUT_DIR, cs)
        overlay_path = os.path.join(cs_dir, "overlays", f"chunk_{chunk_id:04d}_mask_overlay.png")
        imsave(overlay_path, overlay)

        if cs == "rgb":
            converted = img
        elif cs == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cs == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        else:
            continue

        pixels = converted[mask_blue]
        if len(pixels) > 0:
            pixels_by_space[cs].append(pixels)

# --- Compute and Save Stats ---
for cs, pixel_lists in pixels_by_space.items():
    if not pixel_lists:
        print(f"No pixels collected for color space: {cs}")
        continue

    all_pixels = np.vstack(pixel_lists)
    mean = np.mean(all_pixels, axis=0)
    cov = np.cov(all_pixels.T)

    save_dir = os.path.join(OUTPUT_DIR, cs)
    np.save(os.path.join(save_dir, "mean.npy"), mean)
    np.save(os.path.join(save_dir, "cov.npy"), cov)
    print(f"[{cs.upper()}] Saved mean and covariance. Mean shape: {mean.shape}, Cov shape: {cov.shape}")

    # --- Plot histograms for each channel ---
    num_channels = all_pixels.shape[1]
    predefined_names = {
        "rgb": ["R", "G", "B"],
        "hsv": ["H", "S", "V"],
        "lab": ["L", "A", "B"]
    }
    channel_names = predefined_names.get(cs, [])
    if len(channel_names) < num_channels:
        channel_names += [f"C{i}" for i in range(len(channel_names), num_channels)]

    for i in range(num_channels):
        plt.figure(figsize=(6, 4))
        plt.hist(all_pixels[:, i], bins=50, color='gray', edgecolor='black')
        plt.title(f"{cs.upper()} - Channel {channel_names[i]}")
        plt.xlabel(f"{channel_names[i]} Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{channel_names[i].lower()}_hist.png"))
        plt.close()
