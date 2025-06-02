import rasterio
import numpy as np
import os
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm
import json

def split_tif_into_png_chunks(input_path, output_dir, chunk_size=512):
    """
    Split a TIF file into 512x512 PNG chunks and save non-black chunks,
    also recording their pixel coordinates in the original orthomosaic.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_positions = {}

    with rasterio.open(input_path) as src:
        height, width = src.shape

        num_chunks_x = int(np.ceil(width / chunk_size))
        num_chunks_y = int(np.ceil(height / chunk_size))

        chunk_counter = 0

        for i in tqdm(range(num_chunks_y), desc="Processing rows"):
            for j in range(num_chunks_x):
                y_offset = i * chunk_size
                x_offset = j * chunk_size
                win_height = min(chunk_size, height - y_offset)
                win_width = min(chunk_size, width - x_offset)

                window = Window(x_offset, y_offset, win_width, win_height)
                chunk = src.read(window=window)

                if np.all(chunk == 0):
                    continue

                if win_height < chunk_size or win_width < chunk_size:
                    padded_chunk = np.zeros((src.count, chunk_size, chunk_size), dtype=chunk.dtype)
                    padded_chunk[:, :win_height, :win_width] = chunk
                    chunk = padded_chunk

                if len(chunk.shape) == 3:
                    chunk = np.transpose(chunk, (1, 2, 0))

                if chunk.dtype != np.uint8:
                    min_val, max_val = chunk.min(), chunk.max()
                    if max_val > min_val:
                        chunk = ((chunk - min_val) * (255 / (max_val - min_val))).astype(np.uint8)
                    else:
                        chunk = np.zeros_like(chunk, dtype=np.uint8)

                chunk_id = f'chunk_{chunk_counter:04d}'
                output_path = os.path.join(output_dir, f'{chunk_id}.png')
                Image.fromarray(chunk).save(output_path)

                chunk_positions[chunk_id] = [int(y_offset), int(x_offset)]

                chunk_counter += 1

    with open(os.path.join(output_dir, "chunk_positions.json"), "w") as f:
        json.dump(chunk_positions, f, indent=2)

    print(f"Saved {chunk_counter} non-black PNG chunks to {output_dir}")
    print(f"Saved chunk positions to {os.path.join(output_dir, 'chunk_positions.json')}")

if __name__ == "__main__":
    # === CONFIGURATION ===
    INPUT_TIF = "../ortho/ortho_cut.tif"
    OUTPUT_DIR = "../chunks"
    CHUNK_SIZE = 512

    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, INPUT_TIF)
    output_dir = os.path.join(current_dir, OUTPUT_DIR)

    split_tif_into_png_chunks(input_path, output_dir, CHUNK_SIZE)
