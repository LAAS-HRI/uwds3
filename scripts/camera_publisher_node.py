#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import rospy
import cv2
import sensor_msgs
import numpy as np
from cv_bridge import CvBridge


class CameraPublisher(object):
    """ """
    def __init__(self):
        """Default constructor"""

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.camera_publisher = rospy.Publisher(self.rgb_image_topic, sensor_msgs.msg.Image, queue_size=1)

        self.camera_pub_frequency = rospy.get_param("~camera_pub_frequency", 20)

        self.bridge = CvBridge()
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/rgb/camera_info")
        self.camera_info = sensor_msgs.msg.CameraInfo()
        self.camera_info_publisher = rospy.Publisher(self.camera_info_topic, sensor_msgs.msg.CameraInfo, queue_size=1)

        self.camera_frame_id = rospy.get_param("~camera_frame_id", "camera_link")
        self.camera_info.header.frame_id = self.camera_frame_id

        self.capture = cv2.VideoCapture(0)
        ok, frame = self.capture.read()

        height, width, _ = frame.shape

        focal_length = height
        center = (height/2, width/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype="double")
        P_matrix = np.array([[focal_length, 0, center[0], 0],
                            [0, focal_length, center[1], 0],
                            [0, 0, 1, 0]], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        self.camera_info.D = list(dist_coeffs)
        self.camera_info.K = list(camera_matrix.flatten())
        self.camera_info.P = list(P_matrix.flatten())
        self.camera_info.width = width
        self.camera_info.height = height

        self.timer = rospy.Timer(rospy.Duration(1.0/self.camera_pub_frequency), self.timer_callback)
        rospy.loginfo("Camera publisher ready !")
        while not rospy.is_shutdown():
            rospy.spin()

        self.capture.release()

    def timer_callback(self, event):
        ok, frame = self.capture.read()
        if ok:
            bgr_image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            bgr_image_msg.header.stamp = rospy.Time().now()
            self.camera_info.header = bgr_image_msg.header
            bgr_image_msg.header.frame_id = self.camera_frame_id
            self.camera_publisher.publish(bgr_image_msg)
            self.camera_info_publisher.publish(self.camera_info)


if __name__ == '__main__':
    rospy.init_node("camera_publisher", anonymous=False)
    c = CameraPublisher()
