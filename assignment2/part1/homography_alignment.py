"""
Bonus: Homography-based image alignment (panorama stitching demo).

Usage:
    python homography_alignment.py --img1 data/image_pairs/img1.jpg \
                                   --img2 data/image_pairs/img2.jpg
"""
import argparse
import os
import cv2
import numpy as np

from src.harris import HarrisDetector
from src.descriptors import FeatureDescriptor, HarrisKeypointExtractor
from src.matching import FeatureMatcher, RANSAC
from utils.image_utils import load_image, resize_image, extract_matched_points


def align_images(img1_path, img2_path, out_dir="results/alignment"):
    os.makedirs(out_dir, exist_ok=True)

    img1 = resize_image(load_image(img1_path), max_size=800)
    img2 = resize_image(load_image(img2_path), max_size=800)

    # ── Feature detection & description ──────────────────────────────
    harris = HarrisDetector(k=0.04, window_size=5, threshold=0.01)
    extractor = HarrisKeypointExtractor(harris)

    kp1 = extractor.detect(img1)
    kp2 = extractor.detect(img2)

    desc = FeatureDescriptor(descriptor_type='SIFT')
    kp1, des1 = desc.compute_for_keypoints(img1, kp1)
    kp2, des2 = desc.compute_for_keypoints(img2, kp2)

    # ── Matching + RANSAC ─────────────────────────────────────────────
    matcher = FeatureMatcher(ratio_threshold=0.75)
    matches = matcher.match_descriptors(des1, des2)

    if len(matches) < 4:
        print(f"Not enough matches ({len(matches)}) for homography.")
        return

    pts1, pts2 = extract_matched_points(kp1, kp2, matches)

    ransac = RANSAC(n_iterations=2000, inlier_threshold=4.0, min_inliers=10)
    H, inliers = ransac.estimate_homography(pts1, pts2)

    if H is None:
        print("RANSAC failed to find a valid homography.")
        return

    print(f"Inliers: {inliers.sum()}/{len(matches)}  "
          f"Quality: {ransac.compute_match_quality(H, pts1, pts2, inliers):.3f}")

    # ── Warp img2 onto img1's plane ───────────────────────────────────
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find bounding box of both images after applying H to img2's corners
    corners2 = np.array([[0, 0, 1], [w2, 0, 1], [w2, h2, 1], [0, h2, 1]], dtype=np.float64).T
    corners2_proj = H @ corners2
    corners2_proj /= corners2_proj[2:3, :]
    all_x = np.concatenate([[0, w1], corners2_proj[0]])
    all_y = np.concatenate([[0, h1], corners2_proj[1]])

    x_min, x_max = int(np.floor(all_x.min())), int(np.ceil(all_x.max()))
    y_min, y_max = int(np.floor(all_y.min())), int(np.ceil(all_y.max()))

    # Translation to keep all pixels in positive coordinates
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)

    out_w = x_max - x_min
    out_h = y_max - y_min

    # Warp img2 into canvas
    warped = cv2.warpPerspective(img2, T @ H, (out_w, out_h))

    # Place img1 in canvas
    canvas = warped.copy()
    roi = canvas[-y_min: -y_min + h1, -x_min: -x_min + w1]
    # Simple copy — img1 takes priority
    roi[:, :] = img1[:roi.shape[0], :roi.shape[1]]

    # ── Save outputs ──────────────────────────────────────────────────
    cv2.imwrite(os.path.join(out_dir, "warped_img2.jpg"), warped)
    cv2.imwrite(os.path.join(out_dir, "aligned_panorama.jpg"), canvas)

    # Side-by-side comparison
    max_h = max(h1, h2)
    side1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT)
    side2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT)
    comparison = np.hstack([side1, side2])
    cv2.imwrite(os.path.join(out_dir, "original_pair.jpg"), comparison)

    print(f"Saved alignment results to {out_dir}/")
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--out",  default="results/alignment")
    args = parser.parse_args()
    align_images(args.img1, args.img2, args.out)