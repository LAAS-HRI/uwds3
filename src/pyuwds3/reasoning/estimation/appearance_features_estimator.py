import cv2
import numpy as np
from ..types import Features


class AppearanceFeaturesEstimator(object):
    def __init__(self, weights, model, output_dim):
        self.model = cv2.dnn.readNetFromTensorflow(weights, model)
        self.name = "appearance"
        self.dimensions = (output_dim, 1)

    def estimate(self, rgb_image, detections=None):
        """ """
        cropped_imgs = []

        for det in detections:
            xmin = int(det.bbox.xmin)
            ymin = int(det.bbox.ymin)
            w = int(det.bbox.width())
            h = int(det.bbox.height())
            if w > 27 and h > 27:
                if self.align is True:
                    aligned_face = self.align_face(rgb_image, det.features["facial_landmarks"])
                    cropped_imgs.append(aligned_face)
                else:
                    cropped_imgs.append(rgb_image[ymin:ymin+h, xmin:xmin+w])

        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          1.0 / 255,
                                          (96, 96),
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for det, features in zip(detections, self.model.forward()):
                confidence = max(1.0, float(h)/rgb_image.shape[0]/2.0)
                if self.name not in det.features:
                    det.features[self.name] = Features(self.name, self.dimensions, np.array(features), confidence)
                else:
                    det.features[self.name].update(np.array(features), confidence)
