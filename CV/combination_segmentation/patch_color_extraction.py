import os
import numpy as np
import cv2
from skimage import io
from tqdm import tqdm

# --- Config ---
PATCH_DIR = "gt_patches/dockweed"
COLOR_SPACES = {
    "rgb": lambda img: img,  # RGB as loaded
    "lab": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2Lab),
    "hsv": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
}

# --- Process each color space ---
for color_space, convert_fn in COLOR_SPACES.items():
    print(f"\nProcessing color space: {color_space.upper()}")
    all_pixels = []

    for fname in tqdm(os.listdir(PATCH_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(PATCH_DIR, fname)
        img = io.imread(img_path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Drop alpha channel if present

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR
        converted = convert_fn(img)  # Convert to current color space
        pixels = converted.reshape(-1, 3)
        all_pixels.append(pixels)

    if not all_pixels:
        print(f"No valid images found in {PATCH_DIR} for {color_space.upper()}")
        continue

    # --- Stack and compute statistics ---
    all_pixels = np.vstack(all_pixels)
    mean = np.mean(all_pixels, axis=0)
    cov = np.cov(all_pixels, rowvar=False)

    # --- Save ---
    out_dir = f"combination/color_stats/{color_space}"
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "mean.npy"), mean)
    np.save(os.path.join(out_dir, "cov.npy"), cov)

    print(f"Saved mean and covariance to: {out_dir}")
