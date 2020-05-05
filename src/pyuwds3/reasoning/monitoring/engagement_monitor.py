import rospy
import numpy as np
import cv2
from .monitor import Monitor

MAX_DEPTH = 1.5
MIN_EYE_PATCH_WIDTH = 5
MIN_EYE_PATCH_HEIGHT = 3
EYE_INPUT_WIDTH = 32
EYE_INPUT_HEIGHT = 16


class EngagementState(object):
    DISENGAGED = 0
    ENGAGED = 1
    DISTRACTED = 2


class EngagementMonitor(Monitor):
    """ Robust engagement monitor based on eye-contact classification
    """
    def __init__(self, model, weights, input_size=28):
        """ Monitor constructor
        """
        super(EngagementMonitor, self).__init__()
        self.model = cv2.readNetFromTensorflow(model, weights)
        self.input_size = input_size
        self.engagement_state = None
        self.face_to_monitor = None

    def monitor(rgb_image, face_tracks, person_tracks):
        """ Monitor the engagement of the persons
        """
        for t in face_tracks:
            r_eye_contours = t.features["facial_landmarks"].right_eye_contours()
            xmin, ymin, w, h = cv2.boundingRect(r_eye_contours)
            r_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH
            l_eye_contours = t.features["facial_landmarks"].left_eye_contours()
            xmin, ymin, w, h = cv2.boundingRect(l_eye_contours)
            l_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH

            if l_eye_detected is True and r_eye_detected is True:
                ## TODO prepare batch for inference
                pass

    def mark_engaged(self):
        """
        """

    def mark_distracted(self):
        """
        """

    def mark_disengaged(self):
        """
        """
