import cv2
import numpy as np
from ..types import Features


class AppearanceFeaturesEstimator(object):
    def __init__(self, weights, model, output_dim):
        self.model = cv2.dnn.readNetFromTensorflow(weights, model)
        self.name = "appearance"
        self.dimensions = (output_dim, 1)

    def estimate(self, rgb_image, detections=None):
        for det in detections:
            pass
            # TODO prepare batch inference
