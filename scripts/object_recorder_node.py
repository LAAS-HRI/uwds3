#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import uuid
import rospy
import cv2
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pyuwds3.reasoning.detection.foreground_detector import ForegroundDetector
from pyuwds3.reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, centroid_cost

DEFAULT_SENSOR_QUEUE_SIZE = 10


class ObjectRecorderNode(object):
    def __init__(self):
        """
        """
        rgb_image_topic = rospy.get_param("~rgb_image_topic", "")

        self.output_data_directory = rospy.get_param("~output_data_directory", "/tmp/")

        self.bridge = CvBridge()
        self.label = rospy.get_param("~label", "thing")

        self.object_detector = ForegroundDetector()

        self.object_tracker = MultiObjectTracker(iou_cost,
                                                 centroid_cost,
                                                 0.98,
                                                 None,
                                                 40,
                                                 10,
                                                 120,
                                                 use_tracker=True)

        self.data_path = self.output_data_directory+"/"+self.label

        self.max_samples = rospy.get_param("~max_samples", 10)

        self.nb_sample = 0

        if not os.path.exists(self.output_data_directory):
            os.makedirs(self.output_data_directory)

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.rgb_image_sub = rospy.Subscriber(rgb_image_topic, Image, self.observation_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

    def observation_callback(self, bgr_image_msg):
        """
        """
        if self.nb_sample < self.max_samples:
            bgr_image = self.bridge.imgmsg_to_cv2(bgr_image_msg, "bgr8")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            detections = self.object_detector.detect(rgb_image)
            tracks = self.object_tracker.update(rgb_image, detections)

            biggest_object = None
            for track in tracks:
                if track.is_confirmed():
                    if biggest_object is None:
                        biggest_object = track
                    else:
                        if track.bbox.area() > biggest_object.bbox.area():
                            biggest_object = track

            if biggest_object is not None:
                xmin = biggest_object.bbox.xmin
                xmax = biggest_object.bbox.xmax
                ymin = biggest_object.bbox.ymin
                ymax = biggest_object.bbox.ymax
                object_image = bgr_image[int(ymin+1):int(ymax-1), int(xmin+1):int(xmax-1)]
                sample_uuid = str(uuid.uuid4()).replace("-", "")
                try:
                    rospy.loginfo("[object_recorder] sample: {} id: {}".format(self.nb_sample, sample_uuid))
                    cv2.imwrite(self.data_path+"/"+sample_uuid+"-rgb.png", object_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                    cv2.imwrite(self.data_path+"/"+sample_uuid+"-mask.png", biggest_object.mask, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
                    self.nb_sample += 1
                except Exception as e:
                    rospy.logwarn("[object_recorder] Exception occured: {}".format(e))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("object_recorder", anonymous=False)
    recorder = ObjectRecorderNode().run()
