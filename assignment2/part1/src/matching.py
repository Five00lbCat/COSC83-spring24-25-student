import numpy as np
import cv2
from scipy.spatial.distance import cdist


class FeatureMatcher:
    def __init__(self, ratio_threshold=0.75, distance_metric='euclidean'):
        self.ratio_threshold = ratio_threshold
        self.distance_metric = distance_metric

    def match_descriptors(self, desc1, desc2):
        """
        Match descriptors using Lowe's ratio test.

        Returns a list of cv2.DMatch objects sorted by distance.
        """
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) == 0 or len(desc2) == 0:
            return []

        # Float32 required for distance computation
        desc1 = desc1.astype(np.float32)
        desc2 = desc2.astype(np.float32)

        # Full pairwise distance matrix  shape: (N1, N2)
        distances = cdist(desc1, desc2, metric=self.distance_metric)

        matches = []
        for i, row in enumerate(distances):
            # Get indices of the two nearest neighbours
            sorted_idx = np.argsort(row)
            if len(sorted_idx) < 2:
                # Only one descriptor in desc2 — accept it unconditionally
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=sorted_idx[0],
                                          _distance=float(row[sorted_idx[0]])))
                continue

            best_idx   = sorted_idx[0]
            second_idx = sorted_idx[1]
            best_dist   = row[best_idx]
            second_dist = row[second_idx]

            # Lowe's ratio test: accept match only if clearly better than runner-up
            if second_dist == 0:
                continue  # degenerate case
            if best_dist / second_dist < self.ratio_threshold:
                matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_idx,
                                          _distance=float(best_dist)))

        # Sort by ascending distance so the best matches come first
        matches.sort(key=lambda m: m.distance)
        return matches


class RANSAC:
    def __init__(self, n_iterations=1000, inlier_threshold=3.0, min_inliers=10):
        self.n_iterations = n_iterations
        self.inlier_threshold = inlier_threshold
        self.min_inliers = min_inliers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_homography_dlt(src, dst):
        """
        Compute a 3×3 homography from exactly 4 point correspondences
        using the Direct Linear Transform (DLT).

        src, dst: (4, 2) float arrays
        Returns H (3, 3) or None on failure.
        """
        A = []
        for (x, y), (xp, yp) in zip(src, dst):
            A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
            A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
        A = np.array(A, dtype=np.float64)

        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]          # last row of Vt = solution to Ah = 0
        H = h.reshape(3, 3)
        if abs(H[2, 2]) < 1e-10:
            return None
        H = H / H[2, 2]    # normalise so h33 = 1
        return H

    def _transfer_error(self, H, src, dst):
        """
        Symmetric transfer error: reprojection distance in both directions.
        Returns per-point error array of shape (N,).
        """
        N = src.shape[0]
        # Homogeneous coordinates
        src_h = np.hstack([src, np.ones((N, 1))])   # (N,3)
        dst_h = np.hstack([dst, np.ones((N, 1))])   # (N,3)

        # Forward: src -> dst
        proj_fwd = (H @ src_h.T).T               # (N,3)
        proj_fwd /= proj_fwd[:, 2:3]
        err_fwd = np.linalg.norm(proj_fwd[:, :2] - dst, axis=1)

        # Backward: dst -> src  (use pseudo-inverse for numerical stability)
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return err_fwd

        proj_bwd = (H_inv @ dst_h.T).T
        proj_bwd /= proj_bwd[:, 2:3]
        err_bwd = np.linalg.norm(proj_bwd[:, :2] - src, axis=1)

        return (err_fwd + err_bwd) / 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_homography(self, src_points, dst_points):
        """
        Estimate homography matrix using RANSAC.

        Args:
            src_points (numpy.ndarray): Source points (N, 2)
            dst_points (numpy.ndarray): Destination points (N, 2)

        Returns:
            tuple: (H, inliers) — H is the 3×3 homography (or None),
                   inliers is a boolean mask of shape (N,).
        """
        assert src_points.shape[0] == dst_points.shape[0], "Point count mismatch"
        assert src_points.shape[0] >= 4, "Need at least 4 correspondences"

        src = src_points.astype(np.float64)
        dst = dst_points.astype(np.float64)
        N = src.shape[0]

        best_H       = None
        best_inliers = np.zeros(N, dtype=bool)
        best_count   = 0

        rng = np.random.default_rng(42)

        for _ in range(self.n_iterations):
            # 1. Random minimal sample (4 correspondences)
            idx = rng.choice(N, 4, replace=False)
            H = self._compute_homography_dlt(src[idx], dst[idx])
            if H is None:
                continue

            # 2. Compute reprojection error for all points
            errors = self._transfer_error(H, src, dst)

            # 3. Identify inliers
            inliers = errors < self.inlier_threshold
            count   = inliers.sum()

            # 4. Keep the best model
            if count > best_count:
                best_count   = count
                best_H       = H
                best_inliers = inliers

        # 5. Re-estimate H using all inliers (if enough)
        if best_count >= self.min_inliers and best_H is not None:
            H_refined = self._compute_homography_dlt(
                src[best_inliers][:4], dst[best_inliers][:4]
            ) if best_count == 4 else self._refit(src[best_inliers], dst[best_inliers])
            if H_refined is not None:
                best_H = H_refined

        return best_H, best_inliers

    def _refit(self, src, dst):
        """Least-squares homography refit on all inliers via cv2.findHomography."""
        if len(src) < 4:
            return None
        H, _ = cv2.findHomography(src, dst, method=0)
        return H

    def compute_match_quality(self, H, src_points, dst_points, inliers):
        """
        Compute a match quality score in [0, 1].

        Score = inlier_ratio * mean_accuracy
        where mean_accuracy rewards small reprojection errors among inliers.
        """
        if H is None or inliers is None or inliers.sum() == 0:
            return 0.0

        n_total   = len(inliers)
        n_inliers = int(inliers.sum())
        inlier_ratio = n_inliers / n_total if n_total > 0 else 0.0

        # Mean reprojection error of inliers (capped at threshold)
        errors = self._transfer_error(H,
                                      src_points[inliers].astype(np.float64),
                                      dst_points[inliers].astype(np.float64))
        mean_err   = float(np.mean(errors))
        # Map mean_err in [0, threshold] → accuracy in [1, 0]
        accuracy   = max(0.0, 1.0 - mean_err / self.inlier_threshold)

        quality_score = inlier_ratio * accuracy
        return float(quality_score)