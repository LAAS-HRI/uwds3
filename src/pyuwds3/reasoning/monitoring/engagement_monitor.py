import rospy
import numpy as np
from ...types.detection import Detection
from ..assignment.linear_assignment import LinearAssignment
from ...utils.bbox_metrics import overlap
import cv2
from .monitor import Monitor


MAX_DEPTH = 10.0
MIN_EYE_PATCH_WIDTH = 5
MIN_EYE_PATCH_HEIGHT = 3
EYE_INPUT_WIDTH = 60
EYE_INPUT_HEIGHT = 36
WIDTH_MARGIN = 0.4
HEIGHT_MARGIN = 0.4

ENGAGEMENT_START = 3.0
ENGAGEMENT_STOP_DURATION = 5.0

ALPHA = 1.0


def overlap_cost(track_a, track_b):
    """Returns the overlap cost"""
    return 1 - overlap(track_a.bbox, track_b.bbox)


class EngagementState(object):
    DISENGAGED = 0
    ENGAGED = 1
    DISTRACTED = 2


class EyeState(object):
    LOOK_AT_ME = 0
    LOOK_AWAY = 1


class EngagementMonitor(Monitor):
    """ Robust engagement monitor based on eye-contact classification
    """
    def __init__(self, internal_simulator, weigths, model, input_size=(36, 60)):
        """ Monitor constructor
        """
        super(EngagementMonitor, self).__init__(internal_simulator=internal_simulator)
        self.model = cv2.dnn.readNetFromTensorflow(weigths, model)
        self.input_size = input_size

        self.previous_eye_contact_prob = {}
        self.previous_eye_states = {}
        self.previous_face_tracks_map = {}

        self.start_eye_contact = {}
        self.engagement_states = {}

        self.overlap_assignement = LinearAssignment(overlap_cost, max_distance=0.9)

    def monitor(self, rgb_image, face_tracks, person_tracks, time=None):
        """ Monitor the engagement of the persons
        """
        self.cleanup_relations()

        next_eye_states = {}
        face_tracks_map = {}

        eye_contact_prob = {}

        eyes_to_process = []
        face_to_process = []

        for f in face_tracks:
            if f.is_confirmed() and f.is_located():
                if f.bbox.depth < MAX_DEPTH:
                    r_eye_contours = f.features["facial_landmarks"].right_eye_contours()
                    xmin, ymin, w, h = cv2.boundingRect(r_eye_contours)
                    r_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "r_eye", 1.0)
                    r_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH
                    l_eye_contours = f.features["facial_landmarks"].left_eye_contours()
                    xmin, ymin, w, h = cv2.boundingRect(l_eye_contours)
                    l_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "r_eye", 1.0)
                    l_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH

                    if l_eye_detected is True and r_eye_detected is True:
                        face_tracks_map[f.id] = f
                        if l_eye_detection.bbox.area() > r_eye_detection.bbox.area():
                            biggest_eye = l_eye_detection
                        else:
                            biggest_eye = r_eye_detection
                        xmin = biggest_eye.bbox.xmin
                        ymin = biggest_eye.bbox.ymin
                        h = biggest_eye.bbox.height()
                        w = biggest_eye.bbox.width()
                        w_margin = int((w * WIDTH_MARGIN/2.0))
                        h_margin = int((h * HEIGHT_MARGIN/2.0))
                        biggest_eye_patch = rgb_image[ymin-h_margin:ymin+h+h_margin, xmin-w_margin:xmin+w+w_margin]
                        biggest_eye_patch = cv2.cvtColor(biggest_eye_patch, cv2.COLOR_RGB2GRAY)
                        eyes_to_process.append(biggest_eye_patch)
                        face_to_process.append(f)

        if len(eyes_to_process) > 0:
            blob = cv2.dnn.blobFromImages(eyes_to_process,
                                          1.0/255,
                                          self.input_size,
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            output = self.model.forward()
            for f, result in zip(face_to_process, output):
                ec_prob = result.flatten()
                if f.id in self.previous_eye_contact_prob:
                    previous_ec_prob = self.previous_eye_contact_prob[f.id]
                    filtered_ec_prob = previous_ec_prob + ALPHA * (ec_prob - previous_ec_prob)
                else:
                    filtered_ec_prob = ec_prob
                eye_contact_prob[f.id] = filtered_ec_prob
                eye_contact = filtered_ec_prob > 0.5
                #print filtered_ec_prob

                # TODO add hysteresis ?

                # compute next state
                if eye_contact:
                    next_eye_states[f.id] = EyeState.LOOK_AT_ME
                else:
                    next_eye_states[f.id] = EyeState.LOOK_AWAY

            for face_id in self.previous_eye_states.keys():
                face = self.previous_face_tracks_map[face_id]
                if face_id not in next_eye_states:
                    self.assign_and_trigger_action(face, "look back", person_tracks, time)
                elif self.previous_eye_states[face_id] == EyeState.LOOK_AWAY and \
                        next_eye_states[face_id] == EyeState.LOOK_AT_ME:
                    self.assign_and_trigger_action(face, "look at me", person_tracks, time)
                elif self.previous_eye_states[face_id] == EyeState.LOOK_AT_ME and \
                        next_eye_states[face_id] == EyeState.LOOK_AWAY:
                    self.assign_and_trigger_action(face, "look away", person_tracks, time)

        self.previous_face_tracks_map = face_tracks_map
        self.previous_eye_contact_prob = eye_contact_prob
        self.previous_eye_states = next_eye_states

        return self.relations

    def assign_and_trigger_action(self, face, action, person_tracks, time):
        """ Assign an action to the person that overlap with the given face and trigger it
        """
        matches, unmatched_objects, unmatched_person = self.overlap_assignement.match(person_tracks, [face])
        if len(matches > 0):
            _, person_indice = matches[0]
            person = person_tracks[person_indice]
            self.trigger_event(person, action, time=time)
