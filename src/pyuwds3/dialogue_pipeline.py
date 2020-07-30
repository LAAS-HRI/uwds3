import rospy
import cv2
from .base_pipeline import BasePipeline
from .reasoning.detection.ssd_detector import SSDDetector
from .reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost, centroid_cost
from .reasoning.estimation.head_pose_estimator import HeadPoseEstimator
from .reasoning.estimation.object_pose_estimator import ObjectPoseEstimator
from .reasoning.estimation.shape_estimator import ShapeEstimator
from .reasoning.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from .reasoning.estimation.facial_features_estimator import FacialFeaturesEstimator
from .reasoning.estimation.color_features_estimator import ColorFeaturesEstimator
from .reasoning.estimation.semantic_features_estimator import SemanticFeaturesEstimator
from .reasoning.monitoring.perspective_monitor import PerspectiveMonitor
from .reasoning.monitoring.engagement_monitor import EngagementMonitor
from .utils.view_publisher import ViewPublisher


class DialoguePipeline(BasePipeline):
    def __init__(self):
        """
        """
        super(DialoguePipeline, self).__init__()

    def initialize_pipeline(self, internal_simulator, beliefs_base):
        """
        """
        ######################################################
        # Detection
        ######################################################
        detector_weigths_filename = rospy.get_param("~detector_weigths_filename", "")
        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        face_detector_weigths_filename = rospy.get_param("~face_detector_weigths_filename", "")
        face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.person_detector = SSDDetector(detector_weigths_filename,
                                           detector_model_filename,
                                           detector_config_filename)

        self.face_detector = SSDDetector(face_detector_weigths_filename,
                                         face_detector_model_filename,
                                         face_detector_config_filename)

        ####################################################################
        # Features estimation
        ####################################################################

        facial_features_model_filename = rospy.get_param("~facial_features_model_filename", "")
        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")
        face_3d_model_filename = rospy.get_param("~face_3d_model_filename", "")

        self.facial_features_estimator = FacialFeaturesEstimator(shape_predictor_config_filename, facial_features_model_filename)

        self.color_features_estimator = ColorFeaturesEstimator()

        ######################################################
        # Tracking
        ######################################################

        self.n_init = rospy.get_param("~n_init", 1)
        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.8)
        self.max_color_distance = rospy.get_param("~max_color_distance", 0.8)
        self.max_face_distance = rospy.get_param("~max_face_distance", 0.8)
        self.max_centroid_distance = rospy.get_param("~max_centroid_distance", 0.8)
        self.max_lost = rospy.get_param("~max_lost", 4)
        self.max_age = rospy.get_param("~max_age", 20)

        self.face_tracker = MultiObjectTracker(iou_cost,
                                               centroid_cost,
                                               self.max_iou_distance,
                                               None,
                                               self.n_init,
                                               self.max_lost,
                                               self.max_age)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 self.max_iou_distance,
                                                 self.max_color_distance,
                                                 self.n_init,
                                                 self.max_lost,
                                                 self.max_age)

        ########################################################
        # Pose & Shape estimation
        ########################################################

        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)
        face_3d_model_filename = rospy.get_param("~face_3d_model_filename", "")
        self.head_pose_estimator = HeadPoseEstimator(face_3d_model_filename)

        self.shape_estimator = ShapeEstimator()
        self.object_pose_estimator = ObjectPoseEstimator()

        if self.use_word_embeddings is True:
            self.semantic_features_estimator = SemanticFeaturesEstimator(self.word_embeddings)

        ########################################################
        # Monitoring
        ########################################################

        eye_contact_detector_weigths_filename = rospy.get_param("~eye_contact_detector_weigths_filename", "")
        eye_contact_detector_model_filename = rospy.get_param("~eye_contact_detector_model_filename", "")

        self.engagement_monitor = EngagementMonitor(internal_simulator,
                                                    eye_contact_detector_weigths_filename,
                                                    eye_contact_detector_model_filename)

        self.perspective_monitor = PerspectiveMonitor(internal_simulator, beliefs_base)

        ########################################################
        # Visualization
        ########################################################

        self.other_view_publisher = ViewPublisher("other_view")
        self.myself_view_publisher = ViewPublisher("myself_view")

        self.events = []

    def perception_pipeline(self, view_pose, rgb_image, depth_image=None, time=None):
        """
        """
        ######################################################
        # Simulation
        ######################################################
        pipeline_timer = cv2.getTickCount()
        myself = self.internal_simulator.get_myself()

        static_nodes = self.internal_simulator.get_static_entities()

        ######################################################
        # Detection
        ######################################################
        detection_timer = cv2.getTickCount()

        detections = []
        if self.frame_count == 0:
            detections = self.person_detector.detect(rgb_image, depth_image=depth_image)
        elif self.frame_count == 1:
            detections = self.face_detector.detect(rgb_image, depth_image=depth_image)
        else:
            detections = []

        detection_fps = cv2.getTickFrequency() / (cv2.getTickCount()-detection_timer)
        ####################################################################
        # Features estimation
        ####################################################################
        features_timer = cv2.getTickCount()

        self.color_features_estimator.estimate(rgb_image, detections)

        features_fps = cv2.getTickFrequency() / (cv2.getTickCount()-features_timer)
        ######################################################
        # Tracking
        ######################################################
        tracking_timer = cv2.getTickCount()

        if self.frame_count == 0:
            person_detections = [d for d in detections if d.label == "person"]
            self.face_tracks = self.face_tracker.update(rgb_image, [], time=time)
            self.person_tracks = self.person_tracker.update(rgb_image, person_detections, time=time)
        elif self.frame_count == 1:
            self.face_tracks = self.face_tracker.update(rgb_image, detections, time=time)
            self.person_tracks = self.person_tracker.update(rgb_image, [], time=time)
        else:
            self.face_tracks = self.face_tracker.update(rgb_image, [], time=time)
            self.person_tracks = self.person_tracker.update(rgb_image, [], time=time)

        tracks = self.face_tracks + self.person_tracks

        tracking_fps = cv2.getTickFrequency() / (cv2.getTickCount()-tracking_timer)
        ########################################################
        # Pose & Shape estimation
        ########################################################
        pose_timer = cv2.getTickCount()

        self.facial_landmarks_estimator.estimate(rgb_image, self.face_tracks)
        self.head_pose_estimator.estimate(self.face_tracks, view_pose, self.robot_camera)
        self.object_pose_estimator.estimate(self.person_tracks, view_pose, self.robot_camera)

        self.shape_estimator.estimate(rgb_image, tracks, self.robot_camera)

        pose_fps = cv2.getTickFrequency() / (cv2.getTickCount()-pose_timer)
        ########################################################
        # Recognition
        ########################################################
        recognition_timer = cv2.getTickCount()

        if self.frame_count == 1:
            self.facial_features_estimator.estimate(rgb_image, self.face_tracks)

        recognition_fps = cv2.getTickFrequency() / (cv2.getTickCount() - recognition_timer)

        if self.use_word_embeddings is True:
            self.semantic_features_estimator.estimate(tracks+static_nodes)

        ########################################################
        # Monitoring
        ########################################################

        # if self.frame_count == 2:
        #     self.events = self.engagement_monitor.monitor(rgb_image, self.face_tracks, self.person_tracks, time)

        monitoring_timer = cv2.getTickCount()

        if self.frame_count == 3:
            success, other_image, other_visible_tracks, self.events = self.perspective_monitor.monitor_others(self.face_tracks)

        monitoring_fps = cv2.getTickFrequency() / (cv2.getTickCount()-monitoring_timer)
        pipeline_fps = cv2.getTickFrequency() / (cv2.getTickCount()-pipeline_timer)
        ########################################################
        # Visualization
        ########################################################
        self.myself_view_publisher.publish(rgb_image, tracks, overlay_image=None, fps=pipeline_fps, view_pose=view_pose, camera=self.robot_camera)

        if self.frame_count == 3:
            if success:
                self.other_view_publisher.publish(other_image, other_visible_tracks, fps=monitoring_fps)

        all_nodes = [myself]+static_nodes+tracks
        return all_nodes, self.events
