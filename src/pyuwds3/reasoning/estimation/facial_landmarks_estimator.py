import cv2
import numpy as np
import dlib
from ...types.landmarks import FacialLandmarks


class FacialLandmarksEstimator(object):
    def __init__(self, shape_predictor_config_file):
        """ """
        self.name = "facial_landmarks"
        self.dimensions = (68, 2)
        self.dlib_predictor = dlib.shape_predictor(shape_predictor_config_file)

    def estimate(self, rgb_image, faces):
        """ """
        image_height, image_width, _ = rgb_image.shape
        for f in faces:
            if f.label == "face":
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                shape = self.dlib_predictor(gray, dlib.rectangle(int(f.bbox.xmin), int(f.bbox.ymin), int(f.bbox.xmax), int(f.bbox.ymax)))
                coords = np.zeros((68, 2), dtype=np.float32)
                for i in range(0, 68):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                f.features[self.name] = FacialLandmarks(coords, image_width, image_height)
                mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, f.features[self.name].face_contours(), 255)
                mask = mask[int(f.bbox.ymin):int(f.bbox.ymax), int(f.bbox.xmin):int(f.bbox.xmax)]
                f.mask = mask.astype("uint8")
