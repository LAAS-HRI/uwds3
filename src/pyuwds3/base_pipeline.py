import cv2
import rospy
import math
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from uwds3_msgs.msg import WorldStamped
from cv_bridge import CvBridge
from .utils.tf_bridge import TfBridge
from .utils.world_publisher import WorldPublisher
from .reasoning.simulation.internal_simulator import InternalSimulator
from .utils.static_word_embeddings import StaticWordEmbeddings
from .utils.view_publisher import ViewPublisher
from .utils.marker_publisher import MarkerPublisher
from .types.camera import Camera
from .types.scene_node import SceneNode
from .reasoning.knowledge.beliefs_base import BeliefsBase


DEFAULT_SENSOR_QUEUE_SIZE = 3


class BasePipeline(object):
    """ Base class to implement perception pipelines """
    def __init__(self):
        """ """
        self.tf_bridge = TfBridge()

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/camera/depth/camera_info")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        self.bridge = CvBridge()

        self.robot_camera = None
        self.camera_info = None
        self.camera_frame_id = None

        self.use_ar_tags = rospy.get_param("use_ar_tags", True)
        self.ar_tags_topic = rospy.get_param("ar_tags_topic", "ar_tags_tracks")
        if self.use_ar_tags is True:
            self.ar_tags_tracks = []
            self.ar_tags_sub = rospy.Subscriber(self.ar_tags_topic, WorldStamped, self.ar_tags_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_motion_capture = rospy.get_param("use_motion_capture", True)
        self.motion_capture_topic = rospy.get_param("motion_capture_topic", "motion_capture_tracks")
        if self.use_motion_capture is True:
            self.motion_capture_tracks = []
            self.motion_capture_sub = rospy.Subscriber(self.motion_capture_topic, WorldStamped, self.motion_capture_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.n_frame = rospy.get_param("~n_frame", 4)
        self.frame_count = 0

        self.use_depth = rospy.get_param("~use_depth", False)

        self.publish_tf = rospy.get_param("~publish_tf", False)

        self.publish_viz = rospy.get_param("~publish_viz", True)

        self.world_publisher = WorldPublisher("tracks")

        self.view_publisher = ViewPublisher("tracks_image")
        self.marker_publisher = MarkerPublisher("tracks_markers")

        self.use_word_embeddings = rospy.get_param("~use_word_embeddings", False)
        static_word_embeddings_filename = rospy.get_param("~static_word_embeddings_filename", "")

        use_simulation_gui = rospy.get_param("~use_simulation_gui", True)
        simulation_config_filename = rospy.get_param("~simulation_config_filename", "")
        cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")
        static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")
        robot_urdf_file_path = rospy.get_param("~robot_urdf_file_path", "")

        self.robot_camera_clipnear = rospy.get_param("~robot_camera_clipnear", 0.1)
        self.robot_camera_clipfar = rospy.get_param("~robot_camera_clipfar", 25.0)

        self.last_update = rospy.Time().now()

        self.beliefs_base = BeliefsBase()

        if self.use_word_embeddings is True:
            self.word_embeddings = StaticWordEmbeddings(static_word_embeddings_filename)

        self.internal_simulator = InternalSimulator(use_simulation_gui,
                                                    simulation_config_filename,
                                                    cad_models_additional_search_path,
                                                    static_entities_config_filename,
                                                    robot_urdf_file_path,
                                                    self.global_frame_id,
                                                    self.base_frame_id)

        self.initialize_pipeline(self.internal_simulator, self.beliefs_base)

        rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.depth_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.depth_camera_info_topic, CameraInfo, self.camera_info_callback)

        if self.use_depth is True:
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.depth_image_topic))
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], DEFAULT_SENSOR_QUEUE_SIZE, 0.1, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)
        else:
            rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self.observation_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

    def camera_info_callback(self, msg):
        """ """
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.robot_camera = Camera().from_msg(msg,
                                              clipnear=self.robot_camera_clipnear,
                                              clipfar=self.robot_camera_clipfar)

    def initialize_pipeline(self, internal_simulator, beliefs_base):
        raise NotImplementedError("You should implement the initialization of the pipeline.")

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

    def observation_callback(self, bgr_image_msg, depth_image_msg=None):
        """ """
        if self.robot_camera is not None:
            header = bgr_image_msg.header
            header.frame_id = self.global_frame_id
            bgr_image = self.bridge.imgmsg_to_cv2(bgr_image_msg, "bgr8")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            if depth_image_msg is not None:
                depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg)
            else:
                depth_image = None

            _, self.image_height, self.image_width = bgr_image.shape

            success, view_pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.camera_frame_id)

            if success is not True:
                rospy.logwarn("[perception] The camera sensor is not localized in world space (frame '{}'), please check if the sensor frame is published in /tf".format(self.global_frame_id))
            else:
                if self.internal_simulator.is_robot_loaded():
                    self.frame_count %= self.n_frame
                    all_nodes, events = self.perception_pipeline(view_pose, rgb_image, depth_image=depth_image, time=header.stamp)

                    self.world_publisher.publish(all_nodes, events, header)

                    if self.publish_viz is True:
                        self.marker_publisher.publish(all_nodes, header)

                    if self.publish_tf is True:
                        self.tf_bridge.publish_tf_frames(all_nodes, events, header)

                    self.frame_count += 1
                else:
                    rospy.logwarn("[perception] Waiting for the robot to be loaded...")

    def perception_pipeline(self, view_pose, rgb_image, depth_image=None, time=None):
        raise NotImplementedError("You should implement the perception pipeline.")