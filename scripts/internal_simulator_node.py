import cv2
import rospy
import math
import numpy as np
from uwds3_msgs.msg import SceneChangesStamped
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.reasoning.simulation.internal_simulator import InternalSimulator
from pyuwds3.reasoning.monitoring.tabletop_action_monitor TabletopActionMonitor

DEFAULT_SENSOR_QUEUE_SIZE = 3


class InternalSimulatorNode(object):
    """ Standalone node for the internal simulator """
    def __init__(self):
        """ """
        self.tf_bridge = TfBridge()

        self.robot_camera = None
        self.camera_info = None
        self.camera_frame_id = None

        self.use_ar_tags = rospy.get_param("use_ar_tags", True)
        self.ar_tags_topic = rospy.get_param("ar_tags_topic", "ar_tags_tracks")
        if self.use_ar_tags is True:
            self.ar_tags_tracks = []
            self.ar_tags_sub = rospy.Subscriber(self.ar_tags_topic, SceneChangesStamped, self.ar_tags_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_motion_capture = rospy.get_param("use_motion_capture", True)
        self.motion_capture_topic = rospy.get_param("motion_capture_topic", "motion_capture_tracks")
        if self.use_motion_capture is True:
            self.motion_capture_tracks = []
            self.motion_capture_sub = rospy.Subscriber(self.motion_capture_topic, SceneChangesStamped, self.motion_capture_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_object_perception = rospy.get_param("use_object_perception", True)
        self.object_perception_topic = rospy.get_param("object_perception_topic", "motion_capture_tracks")
        if self.use_object_perception is True:
            self.object_tracks = []
            self.object_tracks = rospy.Subscriber(self.object_perception_topic, SceneChangesStamped, self.object_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_human_perception = rospy.get_param("use_human_perception", True)
        self.human_perception_topic = rospy.get_param("human_perception_topic", "ar_tracks")
        if self.use_human_perception is True:
            self.human_tracks = []
            self.human_tracks = rospy.Subscriber(self.human_perception_topic, SceneChangesStamped, self.human_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.publish_tf = rospy.get_param("~publish_tf", False)

        self.publish_viz = rospy.get_param("~publish_viz", True)

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

        self.use_physical_monitoring = rospy.get_param("use_physical_monitoring", True)
        if self.use_physical_monitoring is True:
            pass

        self.use_perspective_monitoring = rospy.get_param("use_perspective_monitoring", True)
        if self.use_perspective_monitoring is True:
            pass

        rospy.loginfo("[internal_simulator] Subscribing to '/{}' topic...".format(self.depth_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.depth_camera_info_topic, CameraInfo, self.camera_info_callback)

    def object_tracks_callback(self, scene_changes_msg):
        ar_tags_tracks = []
        for node in scene_changes_msg.changes.nodes:
            ar_tags_tracks.append(SceneNode().from_msg(node))
        self.ar_tags_tracks = ar_tags_tracks

    def human_tracks_callback(self, scene_changes_msg):
        ar_tags_tracks = []
        for node in scene_changes_msg.changes.nodes:
            ar_tags_tracks.append(SceneNode().from_msg(node))
        self.ar_tags_tracks = ar_tags_tracks

    def ar_tags_callback(self, scene_changes_msg):
        ar_tags_tracks = []
        for node in scene_changes_msg.changes.nodes:
            ar_tags_tracks.append(SceneNode().from_msg(node))
        self.ar_tags_tracks = ar_tags_tracks

    def motion_capture_callback(self, scene_changes_msg):
        motion_capture_tracks = []
        for node in scene_changes_msg.changes.nodes:
            motion_capture_tracks.append(SceneNode().from_msg(node))
        self.motion_capture_tracks = motion_capture_tracks

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

    def camera_info_callback(self, msg):
        """ """
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
            self.camera_info = msg
            self.camera_frame_id = msg.header.frame_id
            self.robot_camera = Camera().from_msg(msg,
                                                  clipnear=self.robot_camera_clipnear,
                                                  clipfar=self.robot_camera_clipfar)

        # TODO: monitoring
