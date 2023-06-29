import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

def create_depth_image(buffer: np.ndarray, filter: bool = False):
    """Converts a depth buffer to a 8-bit depth image, with min and max depth as range. 
    Optionally applies a suite of heuristic filters.""" 
    depth_buffer = buffer.copy()
    bits_per_pixel = 8
    res_per_pixel = 2**bits_per_pixel - 1
    min_depth = np.min(depth_buffer[depth_buffer < 1.0])
    max_depth = np.max(depth_buffer[depth_buffer < 1.0])
    depth_step = (max_depth - min_depth) / res_per_pixel
    if filter:
        fill_zero_pixels(depth_buffer)
        depth_buffer = median_filter(depth_buffer, size=3)

    depth_buffer[depth_buffer == 1.0] = max_depth
    depth_buffer = np.round((depth_buffer - min_depth) / depth_step, 0).astype(np.uint8)
    depth_buffer = res_per_pixel - depth_buffer
    depth_image = Image.fromarray(depth_buffer, mode='L')
    return depth_image

def fill_zero_pixels(depth_buffer: np.ndarray):
    """Fills zero pixels with the average of their non zero neighbors."""
    rows, cols = depth_buffer.shape
    kernel_size = 3
    kernel_radius = kernel_size // 2
    for i in range(rows):
        for j in range(cols):
            if depth_buffer[i, j] == 1.0:
                neighborhood = depth_buffer[max(i - kernel_radius, 0):min(i + kernel_radius + 1, rows), max(j - kernel_radius, 0):min(j + kernel_radius + 1, cols)]
                neighborhood = neighborhood[neighborhood < 1.0]
                if neighborhood.size > 0:
                    depth_buffer[i, j] = np.mean(neighborhood)
