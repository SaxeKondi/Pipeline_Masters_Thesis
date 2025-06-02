import numpy as np
from skimage import color, filters
from skimage.io import imsave
import matplotlib.pyplot as plt

def compute_gradient_magnitude(image, method='sobel'):
    """
    Compute the gradient magnitude of a grayscale image.

    Args:
        image (ndarray): Input RGB or grayscale image.
        method (str): 'sobel' (default) or 'scharr'.

    Returns:
        ndarray: Gradient magnitude map (float32).
    """
    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    if method == 'sobel':
        grad = filters.sobel(gray)
    elif method == 'scharr':
        grad = filters.scharr(gray)
    else:
        raise ValueError("Unsupported method. Use 'sobel' or 'scharr'.")

    return grad.astype(np.float32)

def apply_threshold(grad_map, threshold):
    """
    Apply binary thresholding to a gradient map.

    Args:
        grad_map (ndarray): Gradient magnitude map.
        threshold (float): Threshold value.

    Returns:
        ndarray: Binary mask (uint8), values in {0, 255}.
    """
    return ((grad_map >= threshold) * 255).astype(np.uint8)

def create_overlay(image, mask, color=(255, 0, 0), alpha=0.3):
    """
    Create an overlay by blending a colored mask onto the image.

    Args:
        image (ndarray): Original RGB image.
        mask (ndarray): Binary mask (values 0 or 255).
        color (tuple): Overlay color (R, G, B).
        alpha (float): Opacity of the overlay.

    Returns:
        ndarray: RGB image with overlay.
    """
    overlay = image.copy()
    if image.ndim == 2 or image.shape[2] != 3:
        raise ValueError("Overlay requires a 3-channel RGB image.")

    mask_bool = mask.astype(bool)
    for c in range(3):
        overlay[..., c] = np.where(mask_bool,
                                   (1 - alpha) * overlay[..., c] + alpha * color[c],
                                   overlay[..., c])
    return overlay.astype(np.uint8)

def save_all_outputs(output_dir, base_name, image, grad_map, mask, overlay):
    """
    Save intermediate results to disk.

    Args:
        output_dir (str): Folder to save images.
        base_name (str): Base filename (without extension).
        image (ndarray): Original image.
        grad_map (ndarray): Gradient magnitude map.
        mask (ndarray): Binary mask.
        overlay (ndarray): Overlay image.
    """
    os.makedirs(output_dir, exist_ok=True)
    imsave(os.path.join(output_dir, f"{base_name}_grad_map.png"), grad_map, cmap='gray')
    imsave(os.path.join(output_dir, f"{base_name}_mask.png"), mask)
    imsave(os.path.join(output_dir, f"{base_name}_overlay.png"), overlay)
