import numpy as np
import cv2

START = "start"
END = "endseq"


class HandGestureType(object):
    TEST1 = 0
    TEST2 = 1
    TEST3 = 3
    TEST4 = 4
    TEST5 = 5

    names = {TEST1: "test1", TEST2: "test2", TEST3: "test3", TEST4: "test4", TEST5: "test5"}

    features = {TEST1: np.array([1, 0, 0, 0, 0]),
                TEST2: np.array([0, 1, 0, 0, 0]),
                TEST3: np.array([0, 0, 1, 0, 0]),
                TEST4: np.array([0, 0, 0, 1, 0]),
                TEST5: np.array([0, 0, 0, 0, 1])}


class RouteGenerator(object):
    """ """
    def __init__(self,
                 pre_trained_embedding_file,
                 hand_classif_model,
                 hand_classif_weights,
                 captioner_model,
                 captioner_weights,
                 config_file,
                 max_depth=25):
        self.hand_classif_model = cv2.readNetFromTensorflow(hand_classif_model, hand_classif_weights)
        self.captioner_model = cv2.readNetFromTensorflow(captioner_model, hand_classif_weights)

    def generate_hand_gesture(self, depth_image, goal_track):
        """
        """
        height, width = depth_image.shape
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        blob = cv2.dnn.blobFromImages(depth_image,
                                      1.0 / 255,
                                      (128, 128),
                                      (0, 0, 0),
                                      swapRB=False,
                                      crop=False)
        goal = goal_track.bbox.features(width, height).astype("float32")
        gesture_id = self.model.setInput([blob, goal])
        return gesture_id

    def generate_route_description(object, hand_gesture_id):
        """
        """
        pass
