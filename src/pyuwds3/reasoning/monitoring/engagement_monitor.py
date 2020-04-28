import rospy
import numpy as np
import cv2
from .monitor import Monitor

MAX_DEPTH = 1.5


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
        """ Monitor the engagement of the closest face
        """
        face_to_monitor = None
        min_depth = 1000.0
        for t in face_tracks:
            if t.is_confirmed():
                if t.has_camera() is True and t.is_located() is True:
                    if t.bbox.depth < min_depth:
                        min_depth = t.bbox.depth
                        face_to_monitor = t

        if face_to_monitor is not None:
            pass # extract eye image
            

    def mark_engaged(self):
        """
        """

    def mark_distracted(self):
        """
        """
