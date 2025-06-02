import numpy as np
import cv2
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import os

def convert_rgb_to_colorspace(image, colorspace):
    """
    Converts an RGB image to a given color space.
    
    Args:
        image: RGB image as NumPy array (H, W, 3)
        colorspace: One of ['rgb', 'hsv', 'lab']
    
    Returns:
        Converted image
    """
    if colorspace == "rgb":
        return image
    elif colorspace == "hsv":
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif colorspace == "lab":
        return cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    else:
        raise ValueError(f"Unsupported color space: {colorspace}")

def mahalanobis_distance_map(img, mean, cov):
    """
    Computes the Mahalanobis distance for every pixel in an image.

    Args:
        img: Image in shape (H, W, C)
        mean: Mean vector (C,)
        cov: Covariance matrix (C, C)

    Returns:
        Mahalanobis distance map of shape (H, W)
    """
    h, w, c = img.shape
    flat = img.reshape(-1, c).astype(np.float32)
    diff = flat - mean

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not invertible.")

    dists = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    return dists.reshape(h, w)

def generate_binary_mask(distance_map, threshold):
    """
    Thresholds the Mahalanobis distance map to produce a binary mask.

    Args:
        distance_map: (H, W) array of Mahalanobis distances
        threshold: distance threshold

    Returns:
        Binary mask: (H, W), uint8 {0, 255}
    """
    return (distance_map < threshold).astype(np.uint8) * 255

