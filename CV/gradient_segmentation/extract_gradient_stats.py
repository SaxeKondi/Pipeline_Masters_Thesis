import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from tqdm import tqdm
from common.extract_patches_from_csv import patch_extraction_from_csv

# --- Configuration ---
PATCH_DIR = "gt_patches/dockweed"
OUTPUT_DIR = "gradient_analysis/gradient_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

gradient_means = []

# --- Extracting patches ---
patch_extraction_from_csv()

# --- Process Each Patch ---
for fname in tqdm(os.listdir(PATCH_DIR), desc="Processing dockweed patches"):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(PATCH_DIR, fname)
    img = io.imread(path)

    if img.ndim == 3:
        if img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha channel
        gray = color.rgb2gray(img)
    else:
        gray = img  # Already grayscale

    gradient_mag = filters.sobel(gray)
    gradient_means.append(gradient_mag.mean())

# --- Save Data ---
gradient_means = np.array(gradient_means)
np.save(os.path.join(OUTPUT_DIR, "gradient_means.npy"), gradient_means)

# Save summary
mean_val = gradient_means.mean()
std_val = gradient_means.std()
with open(os.path.join(OUTPUT_DIR, "mean_std.txt"), "w") as f:
    f.write(f"Mean: {mean_val:.4f}\n")
    f.write(f"Std:  {std_val:.4f}\n")

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(gradient_means, bins=30, color='teal', edgecolor='black', alpha=0.8)
plt.title("Gradient Magnitude Mean per Patch (Dockweed)")
plt.xlabel("Mean Gradient Magnitude")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "histogram.png"))
plt.close()

print(f"Saved gradient stats to: {OUTPUT_DIR}")
