import cv2
import numpy as np
from pyuwds3.types.features import Features
from .face_frontalizer_estimator import FaceFrontalizerEstimator


class FacialFeaturesEstimator(object):
    """Represents the facial description estimator"""
    def __init__(self, face_3d_model_filename, embedding_model_filename, frontalize=False):
        """FacialFeaturesEstimator constructor"""
        self.name = "facial_description"
        self.dimensions = (128, 0)
        self.model = cv2.dnn.readNetFromTorch(embedding_model_filename)
        if frontalize is True:
            self.frontalizer = FaceFrontalizerEstimator(face_3d_model_filename)
        else:
            self.frontalizer = None

    def estimate(self, rgb_image, faces, camera_matrix=None, dist_coeffs=None):
        """Extracts the facial description features"""
        cropped_imgs = []
        for f in faces:
            if "facial_description" not in f.features:
                xmin = int(f.bbox.xmin)
                ymin = int(f.bbox.ymin)
                w = int(f.bbox.width())
                h = int(f.bbox.height())
                if w > 27 and h > 27:
                    cropped_imgs.append(rgb_image[ymin:ymin+h, xmin:xmin+w])
                    if self.frontalizer is not None:
                        frontalized_img = self.frontalizer.estimate(rgb_image, f, camera_matrix, dist_coeffs)
                        cropped_imgs.append(np.round(frontalized_img).astype(np.uint8))
        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          1.0 / 255,
                                          (96, 96),
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for f, features in zip(faces, self.model.forward()):
                f.features[self.name] = Features(self.name, self.dimensions, np.array(features), float(h)/rgb_image.shape[0])
