#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from pyuwds3.types.camera import Camera
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.vector.vector6d_stable import Vector6DStable
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.shape.mesh import Mesh
from uwds3_msgs.msg import SceneChangesStamped
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.utils.view_publisher import ViewPublisher
from pyuwds3.utils.marker_publisher import MarkerPublisher
from ar_track_alvar_msgs.msg import AlvarMarkers
import yaml

DEFAULT_SENSOR_QUEUE_SIZE = 1


class ArPerceptionNode(object):
    def __init__(self):
        """
        """
        self.tf_bridge = TfBridge()

        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        self.bridge = CvBridge()
        self.robot_camera = None
        self.camera_info = None

        self.events = []

        self.robot_camera_clipnear = rospy.get_param("~robot_camera_clipnear", 0.1)
        self.robot_camera_clipfar = rospy.get_param("~robot_camera_clipfar", 25.0)

        self.publish_tf = rospy.get_param("~publish_tf", False)

        self.publish_viz = rospy.get_param("~publish_viz", True)

        self.scene_publisher = rospy.Publisher("ar_tracks", SceneChangesStamped, queue_size=1)
        self.view_publisher = ViewPublisher("ar_perception")
        self.marker_publisher = MarkerPublisher("ar_markers")

        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")
        rospy.loginfo("[ar_perception] Subscribing to '/{}' topic...".format(self.rgb_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.rgb_camera_info_topic, CameraInfo, self.camera_info_callback)

        self.ar_pose_marker_sub = rospy.Subscriber("ar_pose_marker", AlvarMarkers, self.observation_callback)

        self.ar_nodes = {}

        self.cad_models_search_path = rospy.get_param("~cad_models_search_path", "")

        self.ar_tag_config = rospy.get_param("~ar_tag_config", "")

        if self.ar_tag_config != "":
            with open(self.ar_tag_config, 'r') as config:
                self.ar_config = yaml.safe_load(config)

        for marker in self.ar_config:
            if marker["id"] not in self.ar_nodes:
                node = SceneNode()
                node.label = marker["label"]
                position = marker["position"]
                orientation = marker["orientation"]
                color = marker["color"]
                shape = Mesh("file://"+self.cad_models_search_path + "/" + marker["file"],
                             x=position["x"], y=position["y"], z=position["z"],
                             rx=orientation["x"], ry=orientation["y"], rz=orientation["z"])
                shape.color[0] = color["r"]
                shape.color[1] = color["g"]
                shape.color[2] = color["b"]
                shape.color[3] = color["a"]
                node.shapes.append(shape)
                self.ar_nodes[marker["id"]] = node

    def camera_info_callback(self, msg):
        """ """
        if self.camera_info is None:
            rospy.loginfo("[ar_perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.robot_camera = Camera().from_msg(msg,
                                              clipnear=self.robot_camera_clipnear,
                                              clipfar=self.robot_camera_clipfar)

    def observation_callback(self, ar_marker_msgs):
        """
        """
        if self.robot_camera is not None:
            header = ar_marker_msgs.header
            header.stamp = rospy.Time()
            header.frame_id = self.global_frame_id

            success, view_pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.camera_frame_id)

            if success is not True:
                rospy.logwarn("[ar_perception] The camera sensor is not localized in world space (frame '{}'), please check if the sensor frame is published in /tf".format(self.global_frame_id))
            else:
                all_nodes = []

                for marker in ar_marker_msgs.markers:
                    if marker.id in self.ar_nodes:
                        pose = Vector6D().from_msg(marker.pose.pose)
                        if self.ar_nodes[marker.id].pose is None:
                            self.ar_nodes[marker.id].pose = Vector6DStable(x=pose.pos.x, y=pose.pos.y, z=pose.pos.z,
                                                                           rx=pose.rot.x, ry=pose.rot.y, rz=pose.rot.z, time=header.stamp)
                        else:
                            self.ar_nodes[marker.id].pose.pos.update(x=pose.pos.x, y=pose.pos.y, z=pose.pos.z, time=header.stamp)
                            self.ar_nodes[marker.id].pose.rot.update(x=pose.rot.x, y=pose.rot.y, z=pose.rot.z, time=header.stamp)
                        all_nodes.append(self.ar_nodes[marker.id])

                self.publish_changes(all_nodes, [], header)

                if self.publish_viz is True:
                    self.marker_publisher.publish(all_nodes, header)

                if self.publish_tf is True:
                    self.tf_bridge.publish_tf_frames(all_nodes, [], header)

    def publish_changes(self, tracks, events, header):
        """ """
        scene_changes = SceneChangesStamped()
        scene_changes.header.frame_id = self.global_frame_id
        for track in tracks:
            if track.is_confirmed():
                scene_changes.changes.nodes.append(track.to_msg(header))
        for event in events:
            scene_changes.changes.events.append(event.to_msg(header))
        self.scene_publisher.publish(scene_changes)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("ar_perception", anonymous=False)
    perception = ArPerceptionNode().run()
