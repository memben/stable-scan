import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter

from pointcloud import PointCloud


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
        depth_buffer = gaussian_filter(depth_buffer, sigma=1)

    depth_buffer[depth_buffer == 1.0] = max_depth
    depth_buffer = np.round((depth_buffer - min_depth) / depth_step, 0).astype(np.uint8)
    depth_buffer = res_per_pixel - depth_buffer
    depth_image = Image.fromarray(depth_buffer, mode="L")
    return depth_image


def fill_zero_pixels(depth_buffer: np.ndarray):
    """Fills zero pixels with the average of their non zero neighbors."""
    rows, cols = depth_buffer.shape
    kernel_size = 3
    kernel_radius = kernel_size // 2
    for i in range(rows):
        for j in range(cols):
            if depth_buffer[i, j] == 1.0:
                neighborhood = depth_buffer[
                    max(i - kernel_radius, 0) : min(i + kernel_radius + 1, rows),
                    max(j - kernel_radius, 0) : min(j + kernel_radius + 1, cols),
                ]
                neighborhood = neighborhood[neighborhood < 1.0]
                if neighborhood.size > 0:
                    depth_buffer[i, j] = np.mean(neighborhood)


def filter_ids(
    ids: np.ndarray,
    depth_filtered: Image,
    depth_unfiltered: Image,
    deviation: float = 0.1,
    debug=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a 2D array of ids, a depth image, and a depth image without filtering,
    filter out ids that deviate too much from the filtered depth image."""
    ids_filtered = ids.copy()
    ids_removed = ids.copy()
    if debug:
        depth_filtered.show(title="Filtered Depth Image")
        depth_unfiltered.show(title="Unfiltered Depth Image")
    depth_filtered = np.array(depth_filtered)
    depth_unfiltered = np.array(depth_unfiltered)
    filter_mask = np.zeros_like(depth_filtered, dtype=np.uint8)
    changed = 0
    for x in range(ids.shape[1]):
        for y in range(ids.shape[0]):
            id = ids[y, x]
            if id == PointCloud.EMPTY:
                continue
            upper = depth_filtered[y, x] * (1 + deviation)
            lower = depth_filtered[y, x] * (1 - deviation)
            if depth_unfiltered[y, x] > upper or depth_unfiltered[y, x] < lower:
                changed += 1
                filter_mask[y, x] = 255
                ids_filtered[y, x] = PointCloud.EMPTY
            else:
                ids_removed[y, x] = PointCloud.EMPTY
    # convert filter mask to image
    filter_mask = Image.fromarray(filter_mask, mode="L")
    if debug:
        filter_mask.show(title="Filter Mask")
        print(f"Filtered {changed} ids.")
    return ids_filtered, ids_removed
