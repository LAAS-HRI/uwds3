import numpy as np
import cv2


class StaticSaliencyEstimator(object):
    def __init__(self):
        self.estimator = cv2.saliency.StaticSaliencySpectralResidual_create()

    def estimate(self, rgb_image, depth_image, use_depth=True):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        success, saliency_map = self.estimator.computeSaliency(gray_image)
        saliency_map_normalized = cv2.normalize(saliency_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if use_depth is True:
            normalized_depth_image = cv2.normalize(depth_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            inverse_depth_image = 1.0 - normalized_depth_image
            saliency_with_depth = inverse_depth_image*saliency_map
            saliency_with_depth_normalized = cv2.normalize(saliency_with_depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            saliency_heatmap = cv2.applyColorMap((saliency_with_depth_normalized*255).astype("uint8"), cv2.COLORMAP_JET)
        else:
            saliency_heatmap = cv2.applyColorMap((saliency_map_normalized*255).astype("uint8"), cv2.COLORMAP_JET)
        return saliency_map, saliency_heatmap
