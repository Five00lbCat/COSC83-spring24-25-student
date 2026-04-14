"""
generate_figures.py — Part 1 filter visualisations for COSC83 Assignment 1.

Tests all filters on three inputs:
  1. Programmatically generated checkerboard (128×128)
  2. Noisy version of test.jpg (Gaussian noise, var=0.02)
  3. Natural image: photo_01.jpg (resized to 512px long edge)

Output: output/figures/*.png (one PNG per filter × input combination)
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from filtering import (
    mean_filter, gaussian_filter, gaussian_kernel,
    laplacian_filter, sobel_filter, normalize_image, add_noise,
)

OUT_DIR = _HERE.parent / 'output' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Input images
# ─────────────────────────────────────────────────────────────────────────────

def make_checkerboard(size=128, tile=16):
    """8×8 black/white checkerboard as uint8 [0,255]."""
    board = np.zeros((size, size), dtype=np.uint8)
    for r in range(size // tile):
        for c in range(size // tile):
            if (r + c) % 2 == 0:
                board[r*tile:(r+1)*tile, c*tile:(c+1)*tile] = 255
    return board


def load_natural(path, max_size=512):
    """Load a colour image, resize so long edge ≤ max_size, return RGB uint8."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


checkerboard  = make_checkerboard(128, 16)
natural_color = load_natural(_HERE / 'example_images' / 'photo_01.jpg')
natural_gray  = to_gray(natural_color)

# Noisy: grayscale test.jpg + Gaussian noise
_test_bgr  = cv2.imread(str(_HERE / 'example_images' / 'test.jpg'))
_test_rgb  = cv2.cvtColor(_test_bgr, cv2.COLOR_BGR2RGB)
_test_gray = to_gray(_test_rgb)
noisy_gray = add_noise(_test_gray, noise_type='gaussian', var=0.02)

INPUTS = [
    ('checkerboard', checkerboard,  'Checkerboard'),
    ('noisy',        noisy_gray,    'Noisy test.jpg (σ_noise=0.02)'),
    ('natural',      natural_gray,  'Natural image (photo_01)'),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name):
    path = OUT_DIR / f'{name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {path.relative_to(_HERE.parent)}')


def show_gray(ax, data, title, cmap='gray'):
    ax.imshow(data, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=8, pad=3)
    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gaussian filter — vary sigma (0.5, 1, 2) × kernel size (3, 7, 15)
# ─────────────────────────────────────────────────────────────────────────────

SIGMAS      = [0.5, 1.0, 2.0]
GAUSS_SIZES = [3, 7, 15]

for tag, img, label in INPUTS:
    n_sigma = len(SIGMAS)
    n_size  = len(GAUSS_SIZES)
    # rows: sigma, cols: kernel_size + 1 (original)
    fig, axes = plt.subplots(n_sigma, n_size + 1,
                             figsize=(3*(n_size+1), 3*n_sigma))
    fig.suptitle(f'Gaussian filter — {label}', fontsize=10, y=1.01)

    for ri, sigma in enumerate(SIGMAS):
        show_gray(axes[ri, 0], img, 'Original' if ri == 0 else '')
        for ci, ks in enumerate(GAUSS_SIZES):
            out = gaussian_filter(img.astype(np.float64), kernel_size=ks, sigma=sigma)
            show_gray(axes[ri, ci+1], normalize_image(out),
                      f'k={ks}, σ={sigma}')

    plt.tight_layout()
    save(fig, f'{tag}_gaussian')


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mean filter — vary kernel size (3, 7, 15)
# ─────────────────────────────────────────────────────────────────────────────

MEAN_SIZES = [3, 7, 15]

for tag, img, label in INPUTS:
    fig, axes = plt.subplots(1, len(MEAN_SIZES) + 1,
                             figsize=(3*(len(MEAN_SIZES)+1), 3.5))
    fig.suptitle(f'Mean filter — {label}', fontsize=10)

    show_gray(axes[0], img, 'Original')
    for ci, ks in enumerate(MEAN_SIZES):
        out = mean_filter(img.astype(np.float64), kernel_size=ks)
        show_gray(axes[ci+1], normalize_image(out), f'k={ks}')

    plt.tight_layout()
    save(fig, f'{tag}_mean')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sobel filter — x, y, magnitude, direction
# ─────────────────────────────────────────────────────────────────────────────

for tag, img, label in INPUTS:
    img_f = img.astype(np.float64)
    gx          = sobel_filter(img_f, direction='x')
    gy          = sobel_filter(img_f, direction='y')
    mag, dirn   = sobel_filter(img_f, direction='both')

    panels = [
        (img,              'Original',           'gray'),
        (normalize_image(gx),   'Sobel X (horiz. edges)', 'gray'),
        (normalize_image(gy),   'Sobel Y (vert. edges)',  'gray'),
        (normalize_image(mag),  'Gradient magnitude',     'gray'),
        (dirn,             'Gradient direction',  'hsv'),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.5))
    fig.suptitle(f'Sobel filter — {label}', fontsize=10)

    for ax, (data, title, cmap) in zip(axes, panels):
        show_gray(ax, data, title, cmap=cmap)

    plt.tight_layout()
    save(fig, f'{tag}_sobel')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Laplacian filter — standard and diagonal kernels
# ─────────────────────────────────────────────────────────────────────────────

for tag, img, label in INPUTS:
    img_f = img.astype(np.float64)
    lap_std  = laplacian_filter(img_f, kernel_type='standard')
    lap_diag = laplacian_filter(img_f, kernel_type='diagonal')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(f'Laplacian filter — {label}', fontsize=10)

    show_gray(axes[0], img,                     'Original')
    show_gray(axes[1], normalize_image(lap_std), 'Standard (4-connected)')
    show_gray(axes[2], normalize_image(lap_diag),'Diagonal (8-connected)')

    plt.tight_layout()
    save(fig, f'{tag}_laplacian')


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gaussian kernel visualisation (for report illustration)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(len(SIGMAS), len(GAUSS_SIZES),
                         figsize=(3*len(GAUSS_SIZES), 3*len(SIGMAS)))
fig.suptitle('Gaussian kernels (normalised weight maps)', fontsize=10, y=1.01)

for ri, sigma in enumerate(SIGMAS):
    for ci, ks in enumerate(GAUSS_SIZES):
        k = gaussian_kernel(ks, sigma)
        im = axes[ri, ci].imshow(k, cmap='hot', interpolation='nearest')
        axes[ri, ci].set_title(f'k={ks}, σ={sigma}', fontsize=8)
        axes[ri, ci].axis('off')
        plt.colorbar(im, ax=axes[ri, ci], fraction=0.046, pad=0.04)

plt.tight_layout()
save(fig, 'gaussian_kernels')


# ─────────────────────────────────────────────────────────────────────────────
# 6. Noise reduction comparison (checkerboard and noisy image only)
# ─────────────────────────────────────────────────────────────────────────────

DENOISE_INPUTS = [
    ('noisy', noisy_gray, 'Noisy test.jpg'),
    ('checkerboard', checkerboard, 'Checkerboard'),
]

for tag, img, label in DENOISE_INPUTS:
    img_f  = img.astype(np.float64)
    panels = [
        (img, 'Noisy input'),
        (normalize_image(mean_filter(img_f, kernel_size=3)),       'Mean k=3'),
        (normalize_image(mean_filter(img_f, kernel_size=7)),       'Mean k=7'),
        (normalize_image(gaussian_filter(img_f, 7, sigma=1.0)),    'Gauss k=7 σ=1'),
        (normalize_image(gaussian_filter(img_f, 7, sigma=2.0)),    'Gauss k=7 σ=2'),
        (normalize_image(gaussian_filter(img_f, 15, sigma=2.0)),   'Gauss k=15 σ=2'),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.5))
    fig.suptitle(f'Noise reduction comparison — {label}', fontsize=10)
    for ax, (data, title) in zip(axes, panels):
        show_gray(ax, data, title)
    plt.tight_layout()
    save(fig, f'{tag}_denoise')


# ─────────────────────────────────────────────────────────────────────────────
# 7. Salt-and-pepper noise series
# ─────────────────────────────────────────────────────────────────────────────

saltpepper_gray = add_noise(_test_gray, noise_type='salt_pepper', var=0.05)

SP_INPUTS = [
    ('saltpepper', saltpepper_gray, 'Salt-and-pepper test.jpg (density=0.05)'),
]

# Run all four filter types on the salt-and-pepper image
for tag, img, label in SP_INPUTS:
    # Gaussian
    fig, axes = plt.subplots(1, len(GAUSS_SIZES) + 1, figsize=(3*(len(GAUSS_SIZES)+1), 3.5))
    fig.suptitle(f'Gaussian filter — {label}', fontsize=10)
    show_gray(axes[0], img, 'Original')
    for ci, ks in enumerate(GAUSS_SIZES):
        out = gaussian_filter(img.astype(np.float64), kernel_size=ks, sigma=1.0)
        show_gray(axes[ci+1], normalize_image(out), f'k={ks}, σ=1.0')
    plt.tight_layout()
    save(fig, f'{tag}_gaussian')

    # Mean
    fig, axes = plt.subplots(1, len(MEAN_SIZES) + 1, figsize=(3*(len(MEAN_SIZES)+1), 3.5))
    fig.suptitle(f'Mean filter — {label}', fontsize=10)
    show_gray(axes[0], img, 'Original')
    for ci, ks in enumerate(MEAN_SIZES):
        out = mean_filter(img.astype(np.float64), kernel_size=ks)
        show_gray(axes[ci+1], normalize_image(out), f'k={ks}')
    plt.tight_layout()
    save(fig, f'{tag}_mean')

    # Sobel
    img_f = img.astype(np.float64)
    gx, gy = sobel_filter(img_f, 'x'), sobel_filter(img_f, 'y')
    mag, dirn = sobel_filter(img_f, 'both')
    panels = [
        (img,                  'Original',           'gray'),
        (normalize_image(gx),  'Sobel X',            'gray'),
        (normalize_image(gy),  'Sobel Y',            'gray'),
        (normalize_image(mag), 'Magnitude',          'gray'),
        (dirn,                 'Direction',          'hsv'),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.5))
    fig.suptitle(f'Sobel filter — {label}', fontsize=10)
    for ax, (data, title, cmap) in zip(axes, panels):
        show_gray(ax, data, title, cmap=cmap)
    plt.tight_layout()
    save(fig, f'{tag}_sobel')

    # Laplacian
    lap_std  = laplacian_filter(img.astype(np.float64), kernel_type='standard')
    lap_diag = laplacian_filter(img.astype(np.float64), kernel_type='diagonal')
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(f'Laplacian filter — {label}', fontsize=10)
    show_gray(axes[0], img,                      'Original')
    show_gray(axes[1], normalize_image(lap_std), 'Standard (4-connected)')
    show_gray(axes[2], normalize_image(lap_diag),'Diagonal (8-connected)')
    plt.tight_layout()
    save(fig, f'{tag}_laplacian')

    # Denoising comparison (salt-and-pepper is best removed by median, but
    # we show mean and gaussian as implemented here)
    img_f = img.astype(np.float64)
    panels = [
        (img, 'S&P input'),
        (normalize_image(mean_filter(img_f, kernel_size=3)),       'Mean k=3'),
        (normalize_image(mean_filter(img_f, kernel_size=7)),       'Mean k=7'),
        (normalize_image(gaussian_filter(img_f, 7, sigma=1.0)),    'Gauss k=7 σ=1'),
        (normalize_image(gaussian_filter(img_f, 7, sigma=2.0)),    'Gauss k=7 σ=2'),
        (normalize_image(gaussian_filter(img_f, 15, sigma=2.0)),   'Gauss k=15 σ=2'),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.5))
    fig.suptitle(f'Noise reduction — {label}', fontsize=10)
    for ax, (data, title) in zip(axes, panels):
        show_gray(ax, data, title)
    plt.tight_layout()
    save(fig, f'{tag}_denoise')


# ─────────────────────────────────────────────────────────────────────────────
# 8. Padding mode comparison (mean + gaussian, same crop, zero/reflect/replicate)
# ─────────────────────────────────────────────────────────────────────────────

# Use a 64×64 corner crop of the checkerboard so border effects are prominent
pad_img = checkerboard[:64, :64].astype(np.float64)

PADDING_MODES = ['constant', 'reflect', 'replicate']
PAD_LABELS    = ['Zero (constant)', 'Reflect', 'Replicate (edge)']

fig, axes = plt.subplots(2, len(PADDING_MODES) + 1, figsize=(3*(len(PADDING_MODES)+1), 7))
fig.suptitle('Padding mode comparison — 64×64 checkerboard corner crop, k=7', fontsize=10, y=1.01)

for row, (filter_fn, filter_label, kwargs) in enumerate([
    (mean_filter,     'Mean filter k=7',           {'kernel_size': 7}),
    (gaussian_filter, 'Gaussian filter k=7, σ=2',  {'kernel_size': 7, 'sigma': 2.0}),
]):
    show_gray(axes[row, 0], pad_img.astype(np.uint8),
              'Original' if row == 0 else '')
    for ci, (mode, label) in enumerate(zip(PADDING_MODES, PAD_LABELS)):
        out = filter_fn(pad_img, padding_mode=mode, **kwargs)
        show_gray(axes[row, ci+1], normalize_image(out),
                  f'{filter_label}\n{label}' if row == 0 else label)

plt.tight_layout()
save(fig, 'padding_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# 9. Portrait/personal photo — all filters on photo_01.jpg
# ─────────────────────────────────────────────────────────────────────────────

portrait_color = load_natural(_HERE / 'example_images' / 'photo_01.jpg', max_size=512)
portrait_gray  = to_gray(portrait_color)

PORTRAIT_TAG   = 'portrait'
PORTRAIT_LABEL = 'Personal photo (photo_01)'

# Gaussian
fig, axes = plt.subplots(len(SIGMAS), len(GAUSS_SIZES) + 1,
                         figsize=(3*(len(GAUSS_SIZES)+1), 3*len(SIGMAS)))
fig.suptitle(f'Gaussian filter — {PORTRAIT_LABEL}', fontsize=10, y=1.01)
for ri, sigma in enumerate(SIGMAS):
    show_gray(axes[ri, 0], portrait_gray, 'Original' if ri == 0 else '')
    for ci, ks in enumerate(GAUSS_SIZES):
        out = gaussian_filter(portrait_gray.astype(np.float64), kernel_size=ks, sigma=sigma)
        show_gray(axes[ri, ci+1], normalize_image(out), f'k={ks}, σ={sigma}')
plt.tight_layout()
save(fig, f'{PORTRAIT_TAG}_gaussian')

# Mean
fig, axes = plt.subplots(1, len(MEAN_SIZES) + 1, figsize=(3*(len(MEAN_SIZES)+1), 3.5))
fig.suptitle(f'Mean filter — {PORTRAIT_LABEL}', fontsize=10)
show_gray(axes[0], portrait_gray, 'Original')
for ci, ks in enumerate(MEAN_SIZES):
    out = mean_filter(portrait_gray.astype(np.float64), kernel_size=ks)
    show_gray(axes[ci+1], normalize_image(out), f'k={ks}')
plt.tight_layout()
save(fig, f'{PORTRAIT_TAG}_mean')

# Sobel
img_f = portrait_gray.astype(np.float64)
gx, gy = sobel_filter(img_f, 'x'), sobel_filter(img_f, 'y')
mag, dirn = sobel_filter(img_f, 'both')
panels = [
    (portrait_gray,        'Original',  'gray'),
    (normalize_image(gx),  'Sobel X',   'gray'),
    (normalize_image(gy),  'Sobel Y',   'gray'),
    (normalize_image(mag), 'Magnitude', 'gray'),
    (dirn,                 'Direction', 'hsv'),
]
fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 3.5))
fig.suptitle(f'Sobel filter — {PORTRAIT_LABEL}', fontsize=10)
for ax, (data, title, cmap) in zip(axes, panels):
    show_gray(ax, data, title, cmap=cmap)
plt.tight_layout()
save(fig, f'{PORTRAIT_TAG}_sobel')

# Laplacian
lap_std  = laplacian_filter(portrait_gray.astype(np.float64), kernel_type='standard')
lap_diag = laplacian_filter(portrait_gray.astype(np.float64), kernel_type='diagonal')
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
fig.suptitle(f'Laplacian filter — {PORTRAIT_LABEL}', fontsize=10)
show_gray(axes[0], portrait_gray,               'Original')
show_gray(axes[1], normalize_image(lap_std),    'Standard (4-connected)')
show_gray(axes[2], normalize_image(lap_diag),   'Diagonal (8-connected)')
plt.tight_layout()
save(fig, f'{PORTRAIT_TAG}_laplacian')


print(f'\nAll figures written to {OUT_DIR}')
