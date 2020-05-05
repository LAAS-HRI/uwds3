import numpy as np
import cv2
from .vector.vector2d import Vector2D
from .features import Features


class FacialLandmarks68Index(object):
    POINT_OF_SIGHT = 27
    RIGHT_EYE_CORNER = 36
    LEFT_EYE_CORNER = 45
    NOSE = 30
    MOUTH_UP = 51
    MOUTH_DOWN = 57
    MOUTH_UP = 51
    RIGHT_MOUTH_CORNER = 48
    LEFT_MOUTH_CORNER = 54
    RIGHT_EAR = 0
    LEFT_EAR = 16
    CHIN = 8


class FacialLandmarks(Features):
    """Represents a 68 2D point facial landmarks"""
    def __init__(self, landmarks, image_width, image_height):
        """FacialLandmarks constructor"""
        super(FacialLandmarks, self).__init__("facial_landmarks",
                                              (68, 2),
                                              landmarks,
                                              1.)
        self.image_width = image_width
        self.image_height = image_height

    def draw(self, image, color, thickness):
        """Draws the facial landmarks"""
        # print facial_contours
        cv2.polylines(image, [self.right_eye_contours()], True, color, thickness=thickness)
        cv2.polylines(image, [self.right_eyebrow_contours()], False, color, thickness=thickness)
        cv2.polylines(image, [self.left_eye_contours()], True, color, thickness=thickness)
        cv2.polylines(image, [self.left_eyebrow_contours()], False, color, thickness=thickness)
        cv2.polylines(image, [self.lips_contours()], True, color, thickness=thickness)
        cv2.polylines(image, [self.mouth_contours()], True, color, thickness=thickness)
        cv2.polylines(image, [self.nose_contours()], False, color, thickness=thickness)
        cv2.polylines(image, [self.chin_contours()], False, color, thickness=thickness)

    def to_array(self, normalize=True):
        features = np.zeros((68, 2), dtype=np.float32)
        if normalize is True:
            features[:, 0] = self.data[:, 0]/float(self.image_width)
            features[:, 1] = self.data[:, 1]/float(self.image_height)
        else:
            features = self.data
        return features.flatten()

    def right_eye_contours(self):
        return np.round(self.data[36:42]).astype("int32")

    def right_eyebrow_contours(self):
        return np.round(self.data[17:22]).astype("int32")

    def left_eye_contours(self):
        return np.round(self.data[42:48]).astype("int32")

    def left_eyebrow_contours(self):
        return np.round(self.data[22:27]).astype("int32")

    def mouth_contours(self):
        return np.round(self.data[60:67]).astype("int32")

    def lips_contours(self):
        return np.round(self.data[48:60]).astype("int32")

    def nose_contours(self):
        return np.round(self.data[27:36]).astype("int32")

    def chin_contours(self):
        return np.round(self.data[0:17]).astype("int32")

    def face_contours(self):
        return np.round(np.concatenate([self.data[0:16], np.array([self.data[25], self.data[24], self.data[19], self.data[18]])], axis=0)).astype("int32")
