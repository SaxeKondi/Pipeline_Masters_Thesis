import os
import numpy as np
from skimage.io import imread
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
CHUNK_IDS = [375, 603, 720, 779]
CHUNK_DIR = "chunks"
MASK_DIR = "color_segmentation/red_ball/mask_imgs"
OUTPUT_DIR = "color_segmentation/red_ball/color_stats/hsv"
LABEL_COLOR_RGB = (0, 0, 255)
TOL = 5

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
collected_pixels = []

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

    # Create binary mask for pure red areas in the label
    diff = np.abs(mask[:, :, :3].astype(int) - LABEL_COLOR_RGB)
    mask_red = np.all(diff <= TOL, axis=-1)

    if not np.any(mask_red):
        print(f"Warning: No red pixels found in mask for chunk {chunk_id}")
        continue

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels = hsv_img[mask_red]
    if len(pixels) > 0:
        collected_pixels.append(pixels)

# --- Compute and Save Stats ---
if not collected_pixels:
    print("No HSV pixels collected - check masks or labeling color.")
else:
    all_pixels = np.vstack(collected_pixels)
    mean = np.mean(all_pixels, axis=0)
    cov = np.cov(all_pixels.T)

    np.save(os.path.join(OUTPUT_DIR, "mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "cov.npy"), cov)
    print(f"[HSV] Saved mean and covariance. Mean shape: {mean.shape}, Cov shape: {cov.shape}")

    # --- Plot histograms for H, S, V channels ---
    channel_names = ["H", "S", "V"]
    for i in range(3):
        plt.figure(figsize=(6, 4))
        plt.hist(all_pixels[:, i], bins=50, color='gray', edgecolor='black')
        plt.title(f"HSV - Channel {channel_names[i]}")
        plt.xlabel(f"{channel_names[i]} Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{channel_names[i].lower()}_hist.png"))
        plt.close()
