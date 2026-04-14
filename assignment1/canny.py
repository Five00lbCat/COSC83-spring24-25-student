import numpy as np
from filtering import gaussian_filter, sobel_filter

#10%bonus
def canny_edge_detector(image: np.ndarray, low_threshold: float = 0.05,
                        high_threshold: float = 0.15,
                        sigma: float = 1.0) -> np.ndarray:
    """
    Canny edge detection algorithm (four canonical stages):
      1. Gaussian smoothing  — suppress noise
      2. Sobel gradients     — compute magnitude + direction
      3. Non-maximum suppression (NMS) — thin edges to 1-pixel width
      4. Double-threshold hysteresis   — retain strong edges, extend via weak edges

    Thresholds are given as fractions of the maximum gradient magnitude,
    matching the calling convention in test_filters.py (e.g. 0.05 → 5 %).

    Args:
        image: Grayscale input (2D uint8 or float, H×W)
        low_threshold:  Weak-edge fraction threshold  (0–1)
        high_threshold: Strong-edge fraction threshold (0–1)
        sigma: Standard deviation for the Gaussian pre-blur

    Returns:
        Binary edge map (uint8, 0 or 255), same H×W as input
    """
    # ── 1. Gaussian smoothing ────────────────────────────────────────────────
    # Kernel size: round sigma to an odd integer covering ±3σ
    ksize = max(3, 2 * int(3 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    smoothed = gaussian_filter(image.astype(np.float64), kernel_size=ksize,
                                sigma=sigma, padding_mode='reflect')

    # ── 2. Gradient magnitude + direction ───────────────────────────────────
    magnitude, direction = sobel_filter(smoothed, direction='both',
                                        kernel_size=3, padding_mode='reflect')

    # ── 3. Non-maximum suppression ───────────────────────────────────────────
    # Quantise gradient direction to one of four orientations (0°,45°,90°,135°),
    # then keep a pixel only if its magnitude exceeds both neighbours along
    # the gradient direction.
    h, w = magnitude.shape
    nms = np.zeros_like(magnitude)

    # Convert radians to degrees and wrap to [0, 180)
    angle = np.degrees(direction) % 180

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            m = magnitude[i, j]

            # Select the two neighbours along the gradient direction
            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = magnitude[i, j-1], magnitude[i, j+1]       # 0°
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i-1, j+1], magnitude[i+1, j-1]   # 45°
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i-1, j], magnitude[i+1, j]       # 90°
            else:
                n1, n2 = magnitude[i-1, j-1], magnitude[i+1, j+1]   # 135°

            if m >= n1 and m >= n2:
                nms[i, j] = m

    # ── 4. Double-threshold hysteresis ──────────────────────────────────────
    # Thresholds are fractions of the peak NMS magnitude.
    max_mag = nms.max()
    low  = low_threshold  * max_mag
    high = high_threshold * max_mag

    strong = (nms >= high)
    weak   = (nms >= low) & ~strong

    # Propagate: a weak pixel becomes an edge if it is 8-connected to a strong one.
    # Iterative flood-fill until convergence (simple but correct for typical images).
    edges = strong.copy()
    prev_count = -1
    while edges.sum() != prev_count:
        prev_count = edges.sum()
        # Dilate the current edge set by 1 pixel (8-connected)
        dilated = (
            np.roll(edges,  1, axis=0) | np.roll(edges, -1, axis=0) |
            np.roll(edges,  1, axis=1) | np.roll(edges, -1, axis=1) |
            np.roll(np.roll(edges,  1, axis=0),  1, axis=1) |
            np.roll(np.roll(edges,  1, axis=0), -1, axis=1) |
            np.roll(np.roll(edges, -1, axis=0),  1, axis=1) |
            np.roll(np.roll(edges, -1, axis=0), -1, axis=1)
        )
        edges = edges | (weak & dilated)

    return (edges * 255).astype(np.uint8)
