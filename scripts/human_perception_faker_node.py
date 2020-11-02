#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import CameraInfo
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.utils.world_publisher import WorldPublisher
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.shape.sphere import Sphere

DEFAULT_SENSOR_QUEUE_SIZE = 5


class HumanPerceptionFakerNode(object):
    def __init__(self):
        face_pose_str = rospy.get_param("~face_global_pose", "0 0 0 0 0 0")
        float_list = np.array([float(i) for i in face_pose_str.split()])
        face_pose = Vector6D(x=float_list[0], y=float_list[1], z=float_list[2], rx=float_list[3], ry=float_list[4], rz=float_list[5])
        self.fake_face = SceneNode(label="face", pose=face_pose)
        self.fake_face.shapes.append(Sphere(d=0.15))
        self.fake_face.id = "face"
        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "")
        self.tf_bridge = TfBridge()
        self.world_publisher = WorldPublisher("human_tracks")
        self.image_info_sub = rospy.Subscriber(self.rgb_camera_info_topic, CameraInfo, self.callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

    def callback(self, msg):
        header = msg.header
        header.frame_id = self.global_frame_id
        self.world_publisher.publish([self.fake_face], [], header)
        self.tf_bridge.publish_tf_frames([self.fake_face], [], header)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    rospy.init_node("human_perception_faker", anonymous=False)
    HumanPerceptionFakerNode().run()
