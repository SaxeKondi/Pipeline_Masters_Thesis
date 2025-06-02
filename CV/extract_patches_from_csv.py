import os
import pandas as pd
import numpy as np
from skimage.io import imread, imsave

# --- Configuration ---
CHUNKS_DIR = "chunks"
COORDS_DIR = "gt_patches/patch_coords"
OUTPUT_DIR = "gt_patches"
PATCH_SIZE = 50

def patch_extraction_from_csv():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each CSV coordinate file
    for fname in os.listdir(COORDS_DIR):
        if not fname.endswith(".csv"):
            continue

        chunk_id = fname.replace(".csv", "")
        chunk_path = os.path.join(CHUNKS_DIR, f"{chunk_id}.png")
        coord_path = os.path.join(COORDS_DIR, fname)

        if not os.path.exists(chunk_path):
            print(f"Missing image for {chunk_id}, skipping.")
            continue

        image = imread(chunk_path)
        df = pd.read_csv(coord_path)

        for idx, row in df.iterrows():
            x, y, label = int(row["x"]), int(row["y"]), str(row["label"])
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            # Skip if patch is incomplete
            if patch.shape[0] < PATCH_SIZE or patch.shape[1] < PATCH_SIZE:
                continue

            # Create label folder
            label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            save_name = f"{x}_{y}_{chunk_id}.png"
            save_path = os.path.join(label_dir, save_name)
            imsave(save_path, patch)

    print("Finished extracting labeled patches.")
