import numpy as np
from typing import Tuple, Union

# 10%
def convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply 2D correlation (not convolution) on an image with a given kernel.
    Uses the correlation convention to match OpenCV's filter2D — kernel is NOT flipped.

    Args:
        image: Input image (2D grayscale H×W or 3D color H×W×C)
        kernel: Filter kernel (2D numpy array, must have odd dimensions)
        padding_mode: Border strategy — 'constant' (zero pad), 'reflect', or 'replicate'

    Returns:
        Filtered image, same shape and dtype-range as input (float64)
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be 2D")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    pad_h, pad_w = kh // 2, kw // 2

    # np.pad mode names match our interface exactly for 'reflect' and 'wrap',
    # but 'replicate' is called 'edge' in numpy.
    np_mode_map = {
        'constant': 'constant',
        'reflect':  'reflect',
        'replicate': 'edge',
    }
    if padding_mode not in np_mode_map:
        raise ValueError(f"Unknown padding_mode '{padding_mode}'. "
                         "Use 'constant', 'reflect', or 'replicate'.")
    np_mode = np_mode_map[padding_mode]

    is_color = image.ndim == 3
    # Work on float to avoid uint8 overflow/truncation during accumulation
    img = image.astype(np.float64)

    if is_color:
        h, w, c = img.shape
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                        mode=np_mode)
        out = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i:i+h, j:j+w, :]
    else:
        h, w = img.shape
        padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=np_mode)
        out = np.zeros_like(img)
        for i in range(kh):
            for j in range(kw):
                out += kernel[i, j] * padded[i:i+h, j:j+w]

    return out


#5%
def mean_filter(image: np.ndarray, kernel_size: int = 3, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply mean (box) filtering to an image.

    Args:
        image: Input image (2D or 3D)
        kernel_size: Side length of the square kernel (must be odd)
        padding_mode: Border strategy — 'constant', 'reflect', or 'replicate'

    Returns:
        Filtered image (float64, same spatial shape as input)
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size ** 2)
    return convolve2d(image, kernel, padding_mode)


#5%
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel using the separable Gaussian formula.

    G(x,y) = exp(-(x²+y²) / (2σ²))  then normalised to sum to 1.

    Args:
        size: Kernel side length (must be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalised 2D Gaussian kernel (float64, shape size×size)
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    # Grid of integer coordinates centred at 0, e.g. for size=5: [-2,-1,0,1,2]
    half = size // 2
    ax = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)          # both shape (size, size)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()                # normalise so weights sum to 1
    return kernel


#5%
def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0,
                   padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Gaussian smoothing to an image.

    Args:
        image: Input image (2D or 3D)
        kernel_size: Side length of the Gaussian kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        padding_mode: Border strategy — 'constant', 'reflect', or 'replicate'

    Returns:
        Smoothed image (float64, same spatial shape as input)
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel, padding_mode)


#5%
def laplacian_filter(image: np.ndarray, kernel_type: str = 'standard',
                    padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Laplacian filtering for edge / second-derivative detection.

    'standard'  — 4-connected kernel (no diagonal neighbours):
                  [ 0  1  0]
                  [ 1 -4  1]
                  [ 0  1  0]

    'diagonal'  — 8-connected kernel (includes diagonals):
                  [ 1  1  1]
                  [ 1 -8  1]
                  [ 1  1  1]

    Args:
        image: Input image (2D or 3D)
        kernel_type: 'standard' or 'diagonal'
        padding_mode: Border strategy — 'constant', 'reflect', or 'replicate'

    Returns:
        Filtered image (float64) — values can be negative; use normalize_image for display
    """
    if kernel_type == 'standard':
        kernel = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float64)
    elif kernel_type == 'diagonal':
        kernel = np.array([[1,  1, 1],
                           [1, -8, 1],
                           [1,  1, 1]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown kernel_type '{kernel_type}'. Use 'standard' or 'diagonal'.")

    return convolve2d(image, kernel, padding_mode)


#10%
def sobel_filter(image: np.ndarray, direction: str = 'both', kernel_size: int = 3,
                padding_mode: str = 'constant') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply Sobel filtering for edge / first-derivative detection.

    For kernel_size=3 the standard kernels are used:
        Kx = [-1  0  1]      Ky = [-1 -2 -1]
             [-2  0  2]           [ 0  0  0]
             [-1  0  1]           [ 1  2  1]

    For kernel_size=5 the kernels are constructed as the outer product of the
    1D smoothing vector [1,4,6,4,1] and the 1D differencing vector [-1,-2,0,2,1]
    (the same approach OpenCV uses internally).

    Args:
        image: Input image (2D or 3D)
        direction: 'x' (horizontal edges), 'y' (vertical edges), or 'both'
        kernel_size: 3 or 5
        padding_mode: Border strategy — 'constant', 'reflect', or 'replicate'

    Returns:
        direction='x' or 'y': filtered image (float64)
        direction='both': tuple (gradient_magnitude, gradient_direction)
                          magnitude = sqrt(Gx²+Gy²), direction = arctan2(Gy, Gx) in radians [-π, π]
    """
    if kernel_size == 3:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)
    elif kernel_size == 5:
        # 5×5 Sobel: outer product of smooth=[1,4,6,4,1] and deriv=[-1,-2,0,2,1]
        smooth = np.array([1, 4, 6, 4, 1], dtype=np.float64)
        deriv  = np.array([-1, -2, 0, 2, 1], dtype=np.float64)
        kx = np.outer(smooth, deriv)   # detects horizontal gradient
        ky = np.outer(deriv, smooth)   # detects vertical gradient
    else:
        raise ValueError(f"kernel_size must be 3 or 5, got {kernel_size}")

    if direction == 'x':
        return convolve2d(image, kx, padding_mode)
    elif direction == 'y':
        return convolve2d(image, ky, padding_mode)
    elif direction == 'both':
        gx = convolve2d(image, kx, padding_mode)
        gy = convolve2d(image, ky, padding_mode)
        magnitude  = np.sqrt(gx ** 2 + gy ** 2)
        direction_ = np.arctan2(gy, gx)   # radians in [-π, π]
        return magnitude, direction_
    else:
        raise ValueError(f"direction must be 'x', 'y', or 'both', got '{direction}'")


# These helper functions are provided for you

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to range [0, 255] and convert to uint8.
    """
    min_val = np.min(image)
    max_val = np.max(image)

    # Check to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize to [0, 255]
    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)


def add_noise(image: np.ndarray, noise_type: str = 'gaussian', var: float = 0.01) -> np.ndarray:
    """
    Add noise to an image.

    Args:
        image: Input image
        noise_type: Type of noise ('gaussian' or 'salt_pepper')
        var: Variance (for Gaussian) or density (for salt and pepper)

    Returns:
        Noisy image
    """
    image_copy = image.copy().astype(np.float32)

    if noise_type == 'gaussian':
        # Add Gaussian noise
        noise = np.random.normal(0, var**0.5, image.shape)
        noisy = image_copy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    elif noise_type == 'salt_pepper':
        # Add salt and pepper noise
        salt_mask = np.random.random(image.shape) < var/2
        pepper_mask = np.random.random(image.shape) < var/2

        noisy = image_copy.copy()
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy.astype(np.uint8)

    else:
        raise ValueError("Unknown noise type. Use 'gaussian' or 'salt_pepper'")
