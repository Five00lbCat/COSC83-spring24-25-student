import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from typing import List, Tuple

# Import the filtering functions
from filtering import (
    mean_filter, gaussian_filter, laplacian_filter, sobel_filter, 
    normalize_image, add_noise
)
from canny import canny_edge_detector


def test_basic_filters(image_path: str, save_path: str = None):
    """
    Test all implemented filters on the given image.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply filters
    print("Applying mean filter...")
    mean3x3 = mean_filter(image_rgb, kernel_size=3)
    mean5x5 = mean_filter(image_rgb, kernel_size=5)
    
    print("Applying Gaussian filter...")
    gauss3x3 = gaussian_filter(image_rgb, kernel_size=3, sigma=1.0)
    gauss5x5 = gaussian_filter(image_rgb, kernel_size=5, sigma=1.5)
    
    print("Applying Laplacian filter...")
    laplacian_std = laplacian_filter(gray, kernel_type='standard')
    laplacian_diag = laplacian_filter(gray, kernel_type='diagonal')
    
    print("Applying Sobel filter...")
    sobel_x = sobel_filter(gray, direction='x')
    sobel_y = sobel_filter(gray, direction='y')
    sobel_mag, sobel_dir = sobel_filter(gray, direction='both')
    
    # Normalize edge detection results for display
    laplacian_std_norm = normalize_image(laplacian_std)
    laplacian_diag_norm = normalize_image(laplacian_diag)
    sobel_x_norm = normalize_image(sobel_x)
    sobel_y_norm = normalize_image(sobel_y)
    sobel_mag_norm = normalize_image(sobel_mag)
    
    # Add noise and test denoising
    print("Testing noise reduction...")
    noisy_gray = add_noise(gray, noise_type='gaussian', var=0.02)
    denoised_mean = mean_filter(noisy_gray, kernel_size=5)
    denoised_gauss = gaussian_filter(noisy_gray, kernel_size=5, sigma=1.5)
    
    # Display or save results
    fig = plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 4, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(mean3x3.astype(np.uint8))
    plt.title('Mean Filter 3x3')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(mean5x5.astype(np.uint8))
    plt.title('Mean Filter 5x5')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(gauss3x3.astype(np.uint8))
    plt.title('Gaussian Filter 3x3')
    plt.axis('off')
    
    plt.subplot(3, 4, 5)
    plt.imshow(gauss5x5.astype(np.uint8))
    plt.title('Gaussian Filter 5x5')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(laplacian_std_norm, cmap='gray')
    plt.title('Laplacian (Standard)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(laplacian_diag_norm, cmap='gray')
    plt.title('Laplacian (Diagonal)')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(sobel_x_norm, cmap='gray')
    plt.title('Sobel (X direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 9)
    plt.imshow(sobel_y_norm, cmap='gray')
    plt.title('Sobel (Y direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(sobel_mag_norm, cmap='gray')
    plt.title('Sobel (Magnitude)')
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(sobel_dir, cmap='hsv')
    plt.title('Sobel (Direction)')
    plt.axis('off')
    
    plt.subplot(3, 4, 12)
    plt.imshow(np.hstack([noisy_gray, denoised_gauss]), cmap='gray')
    plt.title('Noisy vs Denoised')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Results saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def compare_with_opencv(image_path: str, save_path: str = None):
    """
    Compare implemented filters with OpenCV implementations.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply our filters
    print("Applying our filters...")
    our_mean = mean_filter(image_rgb, kernel_size=5)
    our_gauss = gaussian_filter(image_rgb, kernel_size=5, sigma=1.5)
    our_laplacian = laplacian_filter(gray, kernel_type='standard')
    our_sobel_x = sobel_filter(gray, direction='x', kernel_size=3)
    our_sobel_y = sobel_filter(gray, direction='y', kernel_size=3)
    
    # Apply OpenCV filters
    print("Applying OpenCV filters...")
    cv_mean = cv2.blur(image, (5, 5))
    cv_gauss = cv2.GaussianBlur(image, (5, 5), 1.5)
    cv_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    cv_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    cv_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Normalize for display
    our_laplacian_norm = normalize_image(our_laplacian)
    our_sobel_x_norm = normalize_image(our_sobel_x)
    our_sobel_y_norm = normalize_image(our_sobel_y)
    
    cv_laplacian_norm = normalize_image(cv_laplacian)
    cv_sobel_x_norm = normalize_image(cv_sobel_x)
    cv_sobel_y_norm = normalize_image(cv_sobel_y)
    
    # Display or save comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 2)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Mean filter comparison
    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(cv_mean, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Mean Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(our_mean.astype(np.uint8))
    plt.title('Our Mean Filter')
    plt.axis('off')
    
    # Gaussian filter comparison
    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(cv_gauss, cv2.COLOR_BGR2RGB))
    plt.title('OpenCV Gaussian Filter')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(our_gauss.astype(np.uint8))
    plt.title('Our Gaussian Filter')
    plt.axis('off')
    
    # Create a separate figure for edge detection comparisons
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_smoothing.png'), dpi=300)
        print(f"Smoothing comparison saved to {save_path.replace('.png', '_smoothing.png')}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Edge detection comparison
    fig = plt.figure(figsize=(15, 12))
    
    # Laplacian comparison
    plt.subplot(3, 3, 1)
    plt.imshow(cv_laplacian_norm, cmap='gray')
    plt.title('OpenCV Laplacian')
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(our_laplacian_norm, cmap='gray')
    plt.title('Our Laplacian')
    plt.axis('off')
    
    # Sobel X comparison
    plt.subplot(3, 3, 4)
    plt.imshow(cv_sobel_x_norm, cmap='gray')
    plt.title('OpenCV Sobel X')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(our_sobel_x_norm, cmap='gray')
    plt.title('Our Sobel X')
    plt.axis('off')
    
    # Sobel Y comparison
    plt.subplot(3, 3, 7)
    plt.imshow(cv_sobel_y_norm, cmap='gray')
    plt.title('OpenCV Sobel Y')
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(our_sobel_y_norm, cmap='gray')
    plt.title('Our Sobel Y')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_edges.png'), dpi=300)
        print(f"Edge detection comparison saved to {save_path.replace('.png', '_edges.png')}")
    else:
        plt.show()
    
    plt.close(fig)


def test_canny(image_path: str, save_path: str = None):
    """
    Test Canny edge detector implementation and compare with OpenCV.
    
    Args:
        image_path: Path to the test image
        save_path: Path to save the result image (if None, display instead)
    """
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Test different parameter combinations
    low_thresholds = [0.05, 0.1, 0.15]
    high_thresholds = [0.15, 0.2, 0.3]
    sigmas = [1.0, 1.5, 2.0]
    
    # Create a figure to display results
    rows = len(low_thresholds)
    cols = len(sigmas) + 1  # +1 for original image in first column
    
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    
    # Display original image
    plt.subplot(rows, cols, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Test each parameter combination
    for i, low_thresh in enumerate(low_thresholds):
        high_thresh = high_thresholds[i]
        for j, sigma in enumerate(sigmas):
            print(f"Testing Canny with low_thresh={low_thresh}, high_thresh={high_thresh}, sigma={sigma}")
            
            # Apply our Canny edge detector
            start_time = time.time()
            our_canny = canny_edge_detector(
                image, 
                low_threshold=low_thresh,
                high_threshold=high_thresh,
                sigma=sigma
            )
            our_time = time.time() - start_time
            
            # Apply OpenCV's Canny for comparison
            start_time = time.time()
            # Convert thresholds to absolute values for OpenCV
            low_thresh_abs = int(low_thresh * 255)
            high_thresh_abs = int(high_thresh * 255)
            # Blur first as OpenCV's Canny does
            blurred = cv2.GaussianBlur(image, (5, 5), sigma)
            cv_canny = cv2.Canny(blurred, low_thresh_abs, high_thresh_abs)
            cv_time = time.time() - start_time
            
            # Plot the results
            plt.subplot(rows, cols, i * cols + j + 2)
            
            # Create a comparison image: left half our implementation, right half OpenCV
            height, width = image.shape
            comparison = np.zeros((height, width), dtype=np.uint8)
            comparison[:, :width//2] = our_canny[:, :width//2]
            comparison[:, width//2:] = cv_canny[:, width//2:]
            
            plt.imshow(comparison, cmap='gray')
            plt.title(f'L={low_thresh}, H={high_thresh}, σ={sigma}\n'
                     f'Ours: {our_time:.3f}s | CV: {cv_time:.3f}s')
            plt.axis('off')
            
            # Draw a vertical line to separate the two implementations
            plt.axvline(x=width//2, color='r', linestyle='-', linewidth=1)
            
            # Add labels to indicate which side is which
            plt.text(width//4, height - 20, 'Our Implementation', 
                    horizontalalignment='center', color='white', fontsize=10)
            plt.text(3*width//4, height - 20, 'OpenCV', 
                    horizontalalignment='center', color='white', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Canny comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def test_all(image_path: str, output_dir: str = "filter_results"):
    """
    Run all tests on the given image.
    
    Args:
        image_path: Path to the test image
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Run basic filter tests
    test_basic_filters(
        image_path, 
        save_path=os.path.join(output_dir, f"{base_name}_basic_filters.png")
    )
    
    # Run comparison with OpenCV
    compare_with_opencv(
        image_path,
        save_path=os.path.join(output_dir, f"{base_name}_opencv_comparison.png")
    )
    
    # Run Canny edge detector tests
    test_canny(
        image_path,
        save_path=os.path.join(output_dir, f"{base_name}_canny_comparison.png")
    )
    
    print(f"All tests completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # Path relative to this script's location
    test_image_path = os.path.join(os.path.dirname(__file__), "example_images", "test.jpg")
    
    # Run all tests
    test_all(test_image_path)