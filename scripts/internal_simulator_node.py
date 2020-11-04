#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import rospy
from uwds3_msgs.msg import WorldStamped
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.camera import Camera
from sensor_msgs.msg import CameraInfo
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.utils.view_publisher import ViewPublisher
from pyuwds3.utils.marker_publisher import MarkerPublisher
from pyuwds3.utils.world_publisher import WorldPublisher
from pyuwds3.reasoning.simulation.internal_simulator import InternalSimulator
from pyuwds3.reasoning.monitoring.physics_monitor import PhysicsMonitor
from pyuwds3.reasoning.monitoring.human_perspective_monitor import HumanPerspectiveMonitor
from pyuwds3.reasoning.monitoring.robot_perspective_monitor import RobotPerspectiveMonitor
from pyuwds3.utils.egocentric_spatial_relations import is_right_of, is_left_of, is_behind
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.vector.vector3d import Vector3D
from pyuwds3.types.situation import Fact
from pyuwds3.types.camera import HumanCamera
from uwds3_msgs.srv import GetPerspective

DEFAULT_SENSOR_QUEUE_SIZE = 10


class InternalSimulatorNode(object):
    """ Standalone node for the internal simulator """
    def __init__(self):
        """ """
        self.tf_bridge = TfBridge()

        self.n_frame = rospy.get_param("~n_frame", 4)
        self.frame_count = 0

        self.robot_camera = None
        self.camera_info = None
        self.camera_frame_id = None

        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")
        self.base_frame_id = rospy.get_param("~base_frame_id", "odom")

        self.use_ar_tags = rospy.get_param("use_ar_tags", True)
        self.ar_tags_topic = rospy.get_param("ar_tags_topic", "ar_tracks")
        if self.use_ar_tags is True:
            self.ar_tags_tracks = []
            self.ar_tags_sub = rospy.Subscriber(self.ar_tags_topic, WorldStamped, self.ar_tags_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_motion_capture = rospy.get_param("use_motion_capture", True)
        self.motion_capture_topic = rospy.get_param("motion_capture_topic", "motion_capture_tracks")
        if self.use_motion_capture is True:
            self.motion_capture_tracks = []
            self.motion_capture_sub = rospy.Subscriber(self.motion_capture_topic, WorldStamped, self.motion_capture_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_object_perception = rospy.get_param("use_object_perception", True)
        self.object_perception_topic = rospy.get_param("object_perception_topic", "object_tracks")
        if self.use_object_perception is True:
            self.object_tracks = []
            self.object_sub = rospy.Subscriber(self.object_perception_topic, WorldStamped, self.object_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_human_perception = rospy.get_param("use_human_perception", True)
        self.human_perception_topic = rospy.get_param("human_perception_topic", "human_tracks")
        if self.use_human_perception is True:
            self.human_tracks = []
            self.human_sub = rospy.Subscriber(self.human_perception_topic, WorldStamped, self.human_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.publish_tf = rospy.get_param("~publish_tf", False)
        self.publish_viz = rospy.get_param("~publish_viz", True)
        self.publish_markers = rospy.get_param("~publish_markers", True)

        self.world_publisher = WorldPublisher("corrected_tracks")
        self.other_world_publisher = WorldPublisher("other_view_tracks")
        self.marker_publisher = MarkerPublisher("corrected_markers")

        self.robot_camera_clipnear = rospy.get_param("~robot_camera_clipnear", 0.1)
        self.robot_camera_clipfar = rospy.get_param("~robot_camera_clipfar", 25.0)

        use_simulation_gui = rospy.get_param("~use_simulation_gui", True)
        simulation_config_filename = rospy.get_param("~simulation_config_filename", "")
        cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")
        static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")
        robot_urdf_file_path = rospy.get_param("~robot_urdf_file_path", "")

        self.internal_simulator = InternalSimulator(use_simulation_gui,
                                                    simulation_config_filename,
                                                    cad_models_additional_search_path,
                                                    static_entities_config_filename,
                                                    robot_urdf_file_path,
                                                    self.global_frame_id,
                                                    self.base_frame_id)

        self.other_view_publisher = ViewPublisher("other_view")

        self.robot_perspective_monitor = RobotPerspectiveMonitor(self.internal_simulator)

        self.use_physical_monitoring = rospy.get_param("use_physical_monitoring", True)
        if self.use_physical_monitoring is True:
            self.physics_monitor = PhysicsMonitor(self.internal_simulator)

        self.use_perspective_monitoring = rospy.get_param("use_perspective_monitoring", True)
        if self.use_perspective_monitoring is True:
            self.perspective_monitor = HumanPerspectiveMonitor(self.internal_simulator)

        rospy.Service("/uwds3/get_perspective", GetPerspective, self.handle_perspective_taking)

        self.perspective_facts = []
        self.egocentric_facts = []
        self.physics_facts = []

        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")
        rospy.loginfo("[internal_simulator] Subscribing to '/{}' topic...".format(self.rgb_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.rgb_camera_info_topic, CameraInfo, self.camera_info_callback)

    def handle_perspective_taking(self, req):
        camera = HumanCamera()
        view_pose = Vector6D().from_msg(req.point_of_view.pose)
        egocentric_relations = []
        if req.use_target is True:
            target_point = Vector3D().from_msg(req.target.point)
            _, _, _, visible_nodes = self.internal_simulator.get_camera_view(view_pose, camera, target=target_point)
        else:
            _, _, _, visible_nodes = self.internal_simulator.get_camera_view(view_pose, camera)
        for node1 in visible_nodes:
            for node2 in visible_nodes:
                if node1 != node2:
                    bbox1 = node1.bbox
                    bbox2 = node2.bbox
                    if is_right_of(bbox1, bbox2) is True:
                        description = node1.description+"("+node1.id[:6]+") is right of "+node2.description+"("+node2.id[:6]+")"
                        egocentric_relations.append(Fact(node1.id, description, predicate="right_of", object=node2.id))
                    if is_left_of(bbox1, bbox2) is True:
                        description = node1.description+"("+node1.id[:6]+") is left of "+node2.description+"("+node2.id[:6]+")"
                        egocentric_relations.append(Fact(node1.id, description, predicate="left_of", object=node2.id))
                    if is_behind(bbox1, bbox2) is True:
                        description = node1.description+"("+node1.id[:6]+") is behind "+node2.description+"("+node2.id[:6]+")"
                        egocentric_relations.append(Fact(node1.id, description, predicate="behind", object=node2.id))
        return visible_nodes, egocentric_relations, True, ""

    def object_perception_callback(self, world_msg):
        object_tracks = []
        for node in world_msg.world.scene:
            object_tracks.append(SceneNode().from_msg(node))
        self.object_tracks = object_tracks

    def human_perception_callback(self, world_msg):
        human_tracks = []
        for node in world_msg.world.scene:
            human_tracks.append(SceneNode().from_msg(node))
        self.human_tracks = human_tracks

    def ar_tags_callback(self, world_msg):
        ar_tags_tracks = []
        for node in world_msg.world.scene:
            ar_tags_tracks.append(SceneNode().from_msg(node))
        self.ar_tags_tracks = ar_tags_tracks

    def motion_capture_callback(self, world_msg):
        motion_capture_tracks = []
        for node in world_msg.world.scene:
            motion_capture_tracks.append(SceneNode().from_msg(node))
        self.motion_capture_tracks = motion_capture_tracks

    def camera_info_callback(self, msg):
        """ """
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
            self.camera_info = msg
            self.camera_frame_id = msg.header.frame_id
            self.robot_camera = Camera().from_msg(msg,
                                                  clipnear=self.robot_camera_clipnear,
                                                  clipfar=self.robot_camera_clipfar)
        if self.internal_simulator.is_robot_loaded() is True:
            success, view_pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.camera_frame_id)

            if success is not True:
                rospy.logwarn("[human_perception] The camera sensor is not localized in world space (frame '{}'), please check if the sensor frame is published in /tf".format(self.global_frame_id))
            else:
                header = msg.header
                header.frame_id = self.global_frame_id
                self.frame_count %= self.n_frame

                object_tracks = self.ar_tags_tracks + self.object_tracks
                person_tracks = [f for f in self.human_tracks if f.label == "person"]

                corrected_object_tracks, self.physics_facts = self.physics_monitor.monitor(object_tracks, person_tracks, header.stamp)

                if self.use_perspective_monitoring is True:
                    if self.frame_count == 3:
                        monitoring_timer = cv2.getTickCount()
                        perspective_facts = []
                        face_tracks = [f for f in self.human_tracks if f.label == "face"]
                        person_tracks = [f for f in self.human_tracks if f.label == "person"]
                        success, other_image, other_visible_tracks, perspective_facts = self.perspective_monitor.monitor(face_tracks, person_tracks, header.stamp)
                        monitoring_fps = cv2.getTickFrequency() / (cv2.getTickCount()-monitoring_timer)
                        if success:
                            self.perspective_facts = [s for s in perspective_facts if s.predicate == "visible_by"]
                            self.other_world_publisher.publish(other_visible_tracks, perspective_facts+self.physics_facts, header)
                            self.other_view_publisher.publish(other_image, other_visible_tracks, header.stamp, fps=monitoring_fps)

                _, self.egocentric_facts = self.robot_perspective_monitor.monitor(object_tracks, person_tracks, self.robot_camera, view_pose, header.stamp)

                corrected_tracks = self.internal_simulator.get_static_entities() + self.human_tracks + corrected_object_tracks

                events = self.physics_facts + self.perspective_facts + self.egocentric_facts

                self.world_publisher.publish(corrected_tracks, events, header)

                if self.publish_tf is True:
                    self.tf_bridge.publish_tf_frames(corrected_tracks, action_events , header)

                if self.publish_markers is True:
                    self.marker_publisher.publish(corrected_tracks, header)

                self.frame_count += 1

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("internal_simulator", anonymous=False)
    perception = InternalSimulatorNode().run()
