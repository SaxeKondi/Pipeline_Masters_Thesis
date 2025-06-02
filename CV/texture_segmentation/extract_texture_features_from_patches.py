import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from texture_analysis.texture_utils import extract_glcm_features, quantize_grayscale
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from common.extract_patches_from_csv import patch_extraction_from_csv

# --- Config ---
PATCHES_ROOT = "gt_patches"
FEATURE_NAMES = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
GRAY_LEVELS = 128
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]
OUTPUT_DIR = "texture_analysis/feature_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Extracting patches ---
patch_extraction_from_csv()

# --- Collect Features by Class ---
features_by_class = {}

for class_name in tqdm(os.listdir(PATCHES_ROOT), desc="Extracting features"):
    class_dir = os.path.join(PATCHES_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue

    class_features = []
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith((".jpg", ".png")):
            continue
        img = plt.imread(os.path.join(class_dir, fname))
        if img.shape[-1] == 4:
            img = img[..., :3]
        gray = color.rgb2gray(img)
        gray_q = quantize_grayscale(gray, gray_levels=GRAY_LEVELS)
        feats_dict = extract_glcm_features(gray_q, gray_levels=GRAY_LEVELS, distances=DISTANCES, angles=ANGLES)
        feats_vector = [feats_dict[name] for name in FEATURE_NAMES]
        class_features.append(feats_vector)

    if class_features:
        features_by_class[class_name] = np.vstack(class_features)

# --- Save raw feature vectors for future classification ---
np.savez(os.path.join(OUTPUT_DIR, "feature_vectors.npz"), **features_by_class)
print(f"Saved raw feature vectors to {os.path.join(OUTPUT_DIR, 'feature_vectors.npz')}")

# --- Define consistent color map (dockweed = red) ---
class_names = sorted(features_by_class.keys())
cmap_colors = plt.get_cmap("tab10").colors
color_map = {}
color_index = 0
for cls in class_names:
    if cls.lower() == "dockweed":
        color_map[cls] = 'red'
    else:
        color_map[cls] = cmap_colors[color_index % len(cmap_colors)]
        color_index += 1

# --- Boxplots ---
for i, feat_name in enumerate(FEATURE_NAMES):
    plt.figure(figsize=(10, 5))
    data = [features_by_class[cls][:, i] for cls in class_names]
    box = plt.boxplot(data, patch_artist=True)

    for patch, cls in zip(box['boxes'], class_names):
        patch.set_facecolor(color_map[cls])

    plt.xticks(ticks=range(1, len(class_names) + 1), labels=class_names)
    plt.title(f"Boxplot of {feat_name}")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feat_name}.png"))
    plt.close()

# --- Scatter Plots for Feature Pairs ---
for i, j in combinations(range(len(FEATURE_NAMES)), 2):
    plt.figure(figsize=(8, 6))
    for cls in class_names:
        feats = features_by_class[cls]
        plt.scatter(feats[:, i], feats[:, j], label=cls, alpha=1, s=20, color=color_map[cls])
    plt.xlabel(FEATURE_NAMES[i])
    plt.ylabel(FEATURE_NAMES[j])
    plt.title(f"{FEATURE_NAMES[i]} vs {FEATURE_NAMES[j]}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"scatter_{FEATURE_NAMES[i]}_vs_{FEATURE_NAMES[j]}.png"))
    plt.close()

print(f"Saved visualizations to {OUTPUT_DIR}")

# --- Concatenate all feature vectors ---
all_features = []
all_labels = []
for cls in class_names:
    feats = features_by_class[cls]
    all_features.append(feats)
    all_labels.extend([cls] * len(feats))

all_features = np.vstack(all_features)
df = pd.DataFrame(all_features, columns=FEATURE_NAMES)
df["label"] = all_labels

# --- Save full features and labels ---
np.save(os.path.join(OUTPUT_DIR, "features_all_classes.npy"), all_features)
np.save(os.path.join(OUTPUT_DIR, "labels_all_classes.npy"), np.array(all_labels))

# --- Correlation heatmap ---
corr = df[FEATURE_NAMES].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("GLCM Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "glcm_correlation_matrix.png"))
plt.close()