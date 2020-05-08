#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import uuid
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


DEFAULT_SENSOR_QUEUE_SIZE = 1
MIN_EYE_PATCH_WIDTH = 0
MIN_EYE_PATCH_HEIGHT = 0
EYE_INPUT_WIDTH = 60
EYE_INPUT_HEIGHT = 36
WIDTH_MARGIN = 0.4
HEIGHT_MARGIN = 0.4


class GazeRecorderNode(object):
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

        self.max_samples = rospy.get_param("~max_samples", 50)
        self.nb_sample = 0

        self.bridge = CvBridge()
        self.label = rospy.get_param("~label", "contact")

        self.record_faces = rospy.get_param("~record_faces", False)
        self.record_landmarks = rospy.get_param("~record_landmarks", False)
        self.record_eyes = rospy.get_param("~record_eyes", True)

        self.face_tracker = MultiObjectTracker(iou_cost,
                                               centroid_cost,
                                               0.98,
                                               None,
                                               5,
                                               60,
                                               120,
                                               use_tracker=True)

        self.data_path = self.output_data_directory+"/"+self.label

        if not os.path.exists(self.output_data_directory):
            os.makedirs(self.output_data_directory)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.view_publisher = rospy.Publisher("myself_view", Image, queue_size=1)

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.observation_callback)

        self.last_landmarks = None

    def observation_callback(self, bgr_image_msg):
        """
        """
        if self.nb_sample < self.max_samples:
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
                biggest_face.bbox.draw(view_image, (0, 200, 0), 2)
                sample_uuid = str(uuid.uuid4()).replace("-", "")

                facial_landmarks = biggest_face.features["facial_landmarks"].to_array()

                l_eye_contours = biggest_face.features["facial_landmarks"].left_eye_contours()
                biggest_face.bbox.draw(view_image, (0, 200, 0), 2)
                xmin, ymin, w, h = cv2.boundingRect(l_eye_contours)
                l_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH
                if l_eye_detected:
                    l_eye_mask = cv2.fillConvexPoly(np.zeros(rgb_image.shape[:2], dtype=np.uint8), l_eye_contours, 255)[ymin:ymin+h, xmin:xmin+w]
                    l_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "l_eye", 1.0, mask=l_eye_mask)
                    w_margin = int((w * WIDTH_MARGIN/2.0))
                    h_margin = int((h * HEIGHT_MARGIN/2.0))
                    l_eye_patch = bgr_image[ymin-h_margin:ymin+h+h_margin, xmin-w_margin:xmin+w+w_margin]
                    l_eye_detection.bbox.draw(view_image, (0, 200, 0), 2)

                r_eye_contours = biggest_face.features["facial_landmarks"].right_eye_contours()
                xmin, ymin, w, h = cv2.boundingRect(r_eye_contours)
                r_eye_detected = h > MIN_EYE_PATCH_HEIGHT and w > MIN_EYE_PATCH_WIDTH
                if r_eye_detected:
                    r_eye_mask = cv2.fillConvexPoly(np.zeros(rgb_image.shape[:2], dtype=np.uint8), r_eye_contours, 255)[ymin:ymin+h, xmin:xmin+w]
                    r_eye_detection = Detection(xmin, ymin, xmin+w, ymin+h, "r_eye", 1.0, mask=r_eye_mask)
                    w_margin = int((w * WIDTH_MARGIN/2.0))
                    h_margin = int((h * HEIGHT_MARGIN/2.0))
                    r_eye_patch = bgr_image[ymin-h_margin:ymin+h+h_margin, xmin-w_margin:xmin+w+w_margin]
                    r_eye_detection.bbox.draw(view_image, (0, 200, 0), 2)

                if l_eye_detected is True and r_eye_detected is True:
                    l_eye_patch_resized = cv2.resize(l_eye_patch, (EYE_INPUT_WIDTH, EYE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                    r_eye_patch_resized = cv2.resize(r_eye_patch, (EYE_INPUT_WIDTH, EYE_INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
                    view_image[0:EYE_INPUT_HEIGHT, 0:EYE_INPUT_WIDTH] = l_eye_patch_resized
                    view_image[0:EYE_INPUT_HEIGHT, EYE_INPUT_WIDTH:EYE_INPUT_WIDTH*2] = r_eye_patch_resized

                    biggest_face.features["facial_landmarks"].draw(view_image, (0, 200, 0), 1)

                    sample_uuid = str(uuid.uuid4()).replace("-", "")

                    try:
                        if self.record_faces:
                            xmin = int(biggest_face.bbox.xmin)
                            xmax = int(biggest_face.bbox.xmax)
                            ymin = int(biggest_face.bbox.ymin)
                            ymax = int(biggest_face.bbox.ymax)
                            face_image = rgb_image[ymin:ymax, xmin:xmax]
                            cv2.imwrite(self.data_path+"/"+sample_uuid+"-face.png", face_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                        rospy.loginfo("[gaze_recorder] sample: {} id: {}".format(self.nb_sample, sample_uuid))
                        if self.record_eyes:
                            cv2.imwrite(self.data_path+"/"+sample_uuid+"-l_eye.png", l_eye_patch_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                            cv2.imwrite(self.data_path+"/"+sample_uuid+"-r_eye.png", r_eye_patch_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                        if self.record_landmarks:
                            with open(self.data_path+"/"+sample_uuid+"-landmarks.txt", 'w') as file:
                                facial_landmarks_str = str(facial_landmarks).replace("[", "")
                                facial_landmarks_str = facial_landmarks_str.replace("]", "")
                                facial_landmarks_str = "-".join(facial_landmarks_str.split()).replace("-", " ")
                                file.write(facial_landmarks_str)
                        self.nb_sample += 1
                    except Exception as e:
                        rospy.logwarn("Exeption occured: {}".format(e))
            self.view_publisher.publish(self.bridge.cv2_to_imgmsg(view_image, "bgr8"))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("gaze_recorder", anonymous=False)
    recorder = GazeRecorderNode().run()
