# CLAUDE.md — COSC83 Assignment 1

## Project overview
Computer vision assignment for COSC83 (Deep Learning for CV) at Dartmouth.
Two parts: traditional image filtering from scratch (Part 1), CNN-based super-resolution (Part 2).

## Environment
- Python 3.7+, virtual environment at `../venv/`
- Activate: `source ../venv/bin/activate`
- Key deps: numpy, torch, torchvision, opencv-python, matplotlib, pillow

## Repo structure
```
assignment1/
  filtering.py      # Part 1: implement all filter functions here
  canny.py          # Bonus: Canny edge detector
  srcnn.py          # Part 2: CNN model definition
  dataloader.py     # Part 2: dataset class and augmentation
  train.py          # Part 2: training loop
  metrics.py        # PSNR and SSIM implementations
  test.py           # integration tests
  test_filters.py   # unit tests for Part 1
  example_images/   # provided test images
```

## Part 1: filters (filtering.py)
Implement from scratch using numpy only — no cv2 filter calls:
- `convolve2d(image, kernel, padding)` — core convolution, handle border conditions
- `mean_filter(image, kernel_size)`
- `gaussian_filter(image, sigma, kernel_size)` — generate kernel from formula
- `laplacian_filter(image)`
- `sobel_filter(image, kernel_size)` — both axes, gradient magnitude + direction

Border padding strategies: zero, reflect, replicate.

## Part 2: super-resolution (srcnn.py, dataloader.py, train.py)
CNN architecture:
- Input: low-res RGB (3 channels)
- Feature extraction: 9x9 conv → 16 residual blocks (conv→BN→ReLU, skip connection)
- Global skip connection across all residual blocks
- Upscaling: PixelShuffle sub-pixel convolution (2x upscale blocks for 4x total)
- Reconstruction: 9x9 conv → HR RGB output
- Support 2x, 3x, 4x scale factors

Training:
- Adam optimizer, L1 loss
- Random crop, flip, rotation augmentation
- Track PSNR and SSIM per epoch
- Save checkpoints

Bonus option B: add perceptual loss (L1 + VGG feature loss), compare results.

## Coding conventions
- Numpy for Part 1, PyTorch for Part 2 — no mixing
- No cv2 filter/blur/sobel calls in Part 1 (defeats the point)
- Type hints preferred
- Each function should have a docstring with param descriptions
- Keep filter implementations readable — math should be clear from the code

## What I own
This is my assignment. You can implement, debug, refactor, and explain — but flag anything conceptually tricky so I understand it, not just run it. Don't silently skip edge cases or use shortcuts that would break on different image sizes.

## Running tests
```bash
python test_filters.py   # Part 1 unit tests
python test.py           # integration
python train.py          # Part 2 training
```