import numpy as np
import cv2


class FeatureDescriptor:
    def __init__(self, descriptor_type='SIFT', params=None):
        self.descriptor_type = descriptor_type
        self.params = params if params is not None else {}
        self._init_descriptor()

    def _init_descriptor(self):
        if self.descriptor_type == 'SIFT':
            nfeatures      = self.params.get('nfeatures', 0)
            nOctaveLayers  = self.params.get('nOctaveLayers', 3)
            contrastThresh = self.params.get('contrastThreshold', 0.04)
            edgeThreshold  = self.params.get('edgeThreshold', 10)
            sigma          = self.params.get('sigma', 1.6)
            self.descriptor = cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThresh,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )

        elif self.descriptor_type == 'SURF':
            # SURF lives in opencv-contrib; fall back to SIFT if unavailable
            hessianThreshold = self.params.get('hessianThreshold', 400)
            nOctaves         = self.params.get('nOctaves', 4)
            nOctaveLayers    = self.params.get('nOctaveLayers', 3)
            extended         = self.params.get('extended', False)
            upright          = self.params.get('upright', False)
            try:
                self.descriptor = cv2.xfeatures2d.SURF_create(
                    hessianThreshold=hessianThreshold,
                    nOctaves=nOctaves,
                    nOctaveLayers=nOctaveLayers,
                    extended=extended,
                    upright=upright
                )
            except AttributeError:
                print("SURF not available (patent restriction). Falling back to SIFT.")
                self.descriptor = cv2.SIFT_create()
                self.descriptor_type = 'SIFT'
        else:
            raise ValueError(f"Unsupported descriptor type: {self.descriptor_type}")

    def detect_and_compute(self, image, mask=None):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.descriptor.detectAndCompute(image, mask)
        return keypoints, descriptors

    def compute_for_keypoints(self, image, keypoints):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors


class HarrisKeypointExtractor:
    def __init__(self, harris_detector):
        self.harris_detector = harris_detector

    def detect(self, image, mask=None):
        corners, _ = self.harris_detector.detect_corners(image)
        coords = self.harris_detector.get_corner_coordinates(corners)

        # Convert to cv2.KeyPoint — size 10 is a reasonable default scale for SIFT
        keypoints = [cv2.KeyPoint(float(x), float(y), 10) for (x, y) in coords]

        # Respect optional mask
        if mask is not None:
            keypoints = [kp for kp in keypoints
                         if mask[int(kp.pt[1]), int(kp.pt[0])] > 0]

        return keypoints