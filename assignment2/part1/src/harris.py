import numpy as np
import cv2
 
 
class HarrisDetector:
    def __init__(self, k=0.04, window_size=3, threshold=0.01):
        self.k = k
        self.window_size = window_size
        self.threshold = threshold
 
    def compute_gradients(self, image):
        image = image.astype(np.float32)
        dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return dx, dy
 
    def compute_structure_tensor(self, dx, dy):
        ksize = self.window_size if self.window_size % 2 == 1 else self.window_size + 1
        Ixx = cv2.GaussianBlur(dx * dx, (ksize, ksize), 0)
        Ixy = cv2.GaussianBlur(dx * dy, (ksize, ksize), 0)
        Iyy = cv2.GaussianBlur(dy * dy, (ksize, ksize), 0)
        return Ixx, Ixy, Iyy
 
    def compute_corner_response(self, Ixx, Ixy, Iyy):
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        response = det - self.k * (trace ** 2)
        return response
 
    def non_max_suppression(self, response, neighborhood_size=3):
        if response.max() > 0:
            response_normalized = response / response.max()
        else:
            response_normalized = response
 
        above_threshold = response_normalized > self.threshold
 
        kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.uint8)
        local_max = cv2.dilate(response_normalized, kernel)
        is_local_max = (response_normalized == local_max)
 
        result = above_threshold & is_local_max
        return result
 
    def detect_corners(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dx, dy = self.compute_gradients(image)
        Ixx, Ixy, Iyy = self.compute_structure_tensor(dx, dy)
        response = self.compute_corner_response(Ixx, Ixy, Iyy)
        corners = self.non_max_suppression(response)
        return corners, response
 
    def get_corner_coordinates(self, corners):
        y_coords, x_coords = np.where(corners)
        return [(x, y) for x, y in zip(x_coords, y_coords)]