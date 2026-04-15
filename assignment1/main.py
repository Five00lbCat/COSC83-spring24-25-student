"""
Demo: run all five Part 1 filters on one image and save a summary figure.
Usage: python main.py
Output: output/main_demo.png
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from filtering import (
    mean_filter,
    gaussian_filter,
    laplacian_filter,
    sobel_filter,
    normalize_image,
)

_HERE = Path(__file__).parent
IMG_PATH = _HERE / 'example_images' / 'test.jpg'
OUT_PATH = _HERE.parent / 'output' / 'main_demo.png'


def load_gray(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert('L')).astype(np.float64) / 255.0
    return img


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    img = load_gray(IMG_PATH)

    mean_out      = mean_filter(img, kernel_size=5)
    gaussian_out  = gaussian_filter(img, kernel_size=7, sigma=1.5)
    laplacian_out = laplacian_filter(img, kernel_type='standard')
    magnitude, _  = sobel_filter(img, direction='both', kernel_size=3)

    # canny is bonus — import conditionally so main.py works without it
    try:
        from canny import canny_edge_detector
        canny_out = canny_edge_detector(img, low_thresh=0.05, high_thresh=0.15, sigma=1.0)
        show_canny = True
    except Exception:
        canny_out = np.zeros_like(img)
        show_canny = False

    panels = [
        (img,           'Original'),
        (mean_out,      'Mean (k=5)'),
        (gaussian_out,  'Gaussian (σ=1.5, k=7)'),
        (normalize_image(laplacian_out), 'Laplacian (standard)'),
        (normalize_image(magnitude),     'Sobel magnitude'),
        (canny_out.astype(float),        'Canny' if show_canny else 'Canny (unavailable)'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Part 1 Filter Demo', fontsize=14)

    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
