import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

# --- Parameters ---
GRAY_LEVELS = 128
DISTANCES = [1, 2, 3, 4, 5]
ANGLES = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8]

def quantize_grayscale(image, gray_levels=GRAY_LEVELS):
    """
    Convert float grayscale [0,1] to quantized uint8 image.
    """
    image_uint8 = img_as_ubyte(image)
    quantized = (image_uint8 / (256 / gray_levels)).astype(np.uint8)
    return quantized

def extract_glcm_features(patch, gray_levels=GRAY_LEVELS, distances=DISTANCES, angles=ANGLES):
    """
    Extract full set of GLCM features from a quantized grayscale patch.
    Returns a dictionary of feature name -> mean value across distances/angles.
    """
    glcm = graycomatrix(
        patch,
        distances=distances,
        angles=angles,
        levels=gray_levels,
        symmetric=True,
        normed=True
    )

    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    features = {prop: graycoprops(glcm, prop).mean() for prop in props}
    return features

