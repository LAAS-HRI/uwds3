import rospy
import cv2
import math
from .base_pipeline import BasePipeline
from .reasoning.detection.ssd_detector import SSDDetector
from .reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost
from .reasoning.estimation.object_pose_estimator import ObjectPoseEstimator
from .reasoning.estimation.shape_estimator import ShapeEstimator
from .reasoning.estimation.color_features_estimator import ColorFeaturesEstimator

from .utils.view_publisher import ViewPublisher


class NavigationPipeline(BasePipeline):
    def __init__(self):
        super(NavigationPipeline, self).__init__()

    def initialize_pipeline(self, internal_simulator, beliefs_base):
        """ """
        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        self.person_detector = SSDDetector(detector_weights_filename,
                                           detector_model_filename,
                                           detector_config_filename,
                                           300)

        self.color_features_estimator = ColorFeaturesEstimator()

        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.8)
        self.max_color_distance = rospy.get_param("~max_color_distance", 0.2)

        self.n_init = rospy.get_param("~n_init", 1)
        self.max_disappeared = rospy.get_param("~max_disappeared", 7)
        self.max_age = rospy.get_param("~max_age", 10)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 0.98,
                                                 self.max_color_distance,
                                                 self.n_init,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 use_tracker=True)

        self.shape_estimator = ShapeEstimator()

        self.object_pose_estimator = ObjectPoseEstimator()

        ########################################################
        # Visualization
        ########################################################

        self.other_view_publisher = ViewPublisher("other_view")
        self.myself_view_publisher = ViewPublisher("myself_view")

    def perception_pipeline(self, view_pose, rgb_image, depth_image=None, time=None):
        """ """
        ######################################################
        # Simulation
        ######################################################
        pipeline_timer = cv2.getTickCount()

        myself = self.internal_simulator.get_myself()

        static_nodes = self.internal_simulator.get_static_entities()

        ######################################################
        # Detection
        ######################################################

        if self.frame_count == 0:
            person_detections = self.person_detector.detect(rgb_image, depth_image=depth_image)
            detections = person_detections
        else:
            detections = []
        ####################################################################
        # Features estimation
        ####################################################################

        self.color_features_estimator.estimate(rgb_image, detections)

        ######################################################
        # Tracking
        ######################################################
        if self.frame_count == 0:
            person_tracks = self.person_tracker.update(rgb_image, person_detections, depth_image=depth_image)
        else:
            person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)

        tracks = person_tracks

        ########################################################
        # Pose & Shape estimation
        ########################################################

        self.object_pose_estimator.estimate(tracks, view_pose, self.robot_camera)

        self.shape_estimator.estimate(rgb_image, tracks, self.robot_camera)

        ########################################################
        # Monitoring
        ########################################################

        events = []
        pipeline_fps = cv2.getTickFrequency() / (cv2.getTickCount()-pipeline_timer)
        ########################################################
        # Visualization
        ########################################################
        self.myself_view_publisher.publish(rgb_image, tracks, time, overlay_image=None, fps=pipeline_fps)

        all_nodes = [myself]+static_nodes+tracks

        return all_nodes, events
