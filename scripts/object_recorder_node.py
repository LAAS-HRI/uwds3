#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import uuid
import rospy
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pyuwds.reasoning.detection.foreground_detector import ForegroundDetector
from pyuwds3.reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, centroid_cost

DEFAULT_SENSOR_QUEUE_SIZE = 10


class ObjectRecorderNode(object):
    def __init__(self):
        """
        """
        rgb_image_topic = rospy.get_param("~rgb_image_topic", "")

        self.output_data_directory = rospy.get_param("~output_data_directory", "/tmp/")

        self.bridge = CvBridge()
        self.name = rospy.get_param("~name", "thing")

        self.object_detector = ForegroundDetector()

        self.object_tracker = MultiObjectTracker(iou_cost,
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

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.observation_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

    def observation_callback(self, bgr_image_msg):
        """
        """
        bgr_image = self.bridge.imgmsg_to_cv2(bgr_image_msg, "bgr8")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        detections = self.object_detector.detect(rgb_image)
        tracks = self.object_tracker.update(rgb_image, detections)

        biggest_track = None
        for track in tracks:
            if biggest_track is None:
                biggest_track = track
            else:
                if track.bbox.area() > biggest_track.bbox.area():
                    biggest_track = track

        if biggest_track is not None:
            pass
