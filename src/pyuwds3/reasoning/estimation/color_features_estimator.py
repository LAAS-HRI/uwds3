import cv2
import numpy as np
from pyuwds3.types.features import Features


class ColorFeaturesEstimator(object):
    """Represents a color features estimator"""
    def __init__(self):
        self.name = "color"
        self.dimensions = (180, 1)

    def estimate(self, rgb_image, detections=None):
        """Extracts hsv histogram as a features vector"""
        for det in detections:
            xmin = int(det.bbox.xmin)
            ymin = int(det.bbox.ymin)
            w = int(det.bbox.width())
            h = int(det.bbox.height())
            cropped_image = rgb_image[ymin:ymin+h, xmin:xmin+w]
            hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
            mask = det.mask
            hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
            hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)
            det.features[self.name] = Features(self.name, self.dimensions, hist, h/float(rgb_image.shape[0]))
