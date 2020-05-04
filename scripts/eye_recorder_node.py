#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pyuwds3.types.detection import Detection
from pyuwds3.reasoning.detection.ssd_detector import SSDDetector
from pyuwds3.reasoning.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from pyuwds3.reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, centroid_cost


DEFAULT_SENSOR_QUEUE_SIZE = 10
MIN_EYE_PATCH_WIDTH = 5
MIN_EYE_PATCH_HEIGHT = 3
EYE_INPUT_WIDTH = 32
EYE_INPUT_HEIGHT = 16


class EyeRecorderNode(object):
    def __init__(self):
        """
        """
        rgb_image_topic = rospy.get_param("~rgb_image_topic", "")

        face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        face_detector_weights_filename = rospy.get_param("~face_detector_weights_filename", "")
        face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.face_detector = SSDDetector(face_detector_model_filename,
                                         face_detector_weights_filename,
                                         face_detector_config_filename)

        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)

        self.output_data_directory = rospy.get_param("~output_data_directory", "/tmp/")

        self.bridge = CvBridge()
        self.name = rospy.get_param("~name", "eye-contact")

        self.face_tracker = MultiObjectTracker(iou_cost,
                                               centroid_cost,
                                               0.98,
                                               None,
                                               5,
                                               60,
                                               120,
                                               use_tracker=True)

        if not os.path.exists(self.output_data_directory):
            os.makedirs(self.output_data_directory)
            data_path = self.output_data_directory+"/"+self.name
            if not os.path.exists(data_path):
                os.makedirs(data_path)

        self.view_publisher = rospy.Publisher("myself_view", Image, queue_size=1)

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.observation_callback)

    def observation_callback(self, bgr_image_msg):
        """
        """
        bgr_image = self.bridge.imgmsg_to_cv2(bgr_image_msg, "bgr8")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        detections = self.face_detector.detect(rgb_image)

        tracks = self.face_tracker.update(rgb_image, detections)

        self.facial_landmarks_estimator.estimate(rgb_image, tracks)

        view_image = bgr_image.copy()

        biggest_face = None
        for track in tracks:
            if biggest_face is None:
                biggest_face = track
            else:
                if track.bbox.area() > biggest_face.bbox.area():
                    biggest_face = track

        if biggest_face is not None:
            l_eye_contours = biggest_face.features["facial_landmarks"].right_eye_contours()
            xmin, ymin, w, h = cv2.boundingRect(l_eye_contours)
            l_eye_detected = False
            if h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH:
                l_eye_mask = cv2.fillConvexPoly(np.zeros(rgb_image.shape[:2], dtype=np.uint8), l_eye_contours, 255)[ymin:ymin+h, xmin:xmin+w]
                l_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "l_eye", 1.0, mask=l_eye_mask)
                l_eye_patch = bgr_image[ymin:ymin+h, xmin:xmin+w]
                l_eye_detection.bbox.draw(view_image, (0, 200, 0), 1)
                l_eye_detected = True

            r_eye_contours = biggest_face.features["facial_landmarks"].left_eye_contours()
            xmin, ymin, w, h = cv2.boundingRect(r_eye_contours)
            r_eye_detected = False
            if h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH:
                r_eye_mask = cv2.fillConvexPoly(np.zeros(rgb_image.shape[:2], dtype=np.uint8), r_eye_contours, 255)[ymin:ymin+h, xmin:xmin+w]
                r_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "r_eye", 1.0, mask=r_eye_mask)
                r_eye_patch = bgr_image[ymin:ymin+h, xmin:xmin+w]
                r_eye_detection.bbox.draw(view_image, (0, 200, 0), 1)
                r_eye_detected = True

            if l_eye_detected is True and r_eye_detected is True:
                l_eye_patch_resized = cv2.resize(l_eye_patch, (EYE_INPUT_WIDTH, EYE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                r_eye_patch_resized = cv2.resize(r_eye_patch, (EYE_INPUT_WIDTH, EYE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                view_image[0:EYE_INPUT_HEIGHT, 0:EYE_INPUT_WIDTH] = l_eye_patch_resized
                view_image[0:EYE_INPUT_HEIGHT, EYE_INPUT_WIDTH:EYE_INPUT_WIDTH*2] = r_eye_patch_resized

            biggest_face.features["facial_landmarks"].draw(view_image, (0, 200, 0), 1)

        self.view_publisher.publish(self.bridge.cv2_to_imgmsg(view_image, "bgr8"))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("eye_recorder", anonymous=False)
    recorder = EyeRecorderNode().run()
