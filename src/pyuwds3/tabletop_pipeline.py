import rospy
import cv2
import math
from .base_pipeline import BasePipeline
from .reasoning.detection.ssd_detector import SSDDetector
from .reasoning.detection.foreground_detector import ForegroundDetector
from .reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost, centroid_cost
from .types.scene_node import SceneNode
from .reasoning.estimation.object_pose_estimator import ObjectPoseEstimator
from .reasoning.estimation.shape_estimator import ShapeEstimator
from .reasoning.estimation.color_features_estimator import ColorFeaturesEstimator
from .reasoning.monitoring.tabletop_action_monitor import TabletopActionMonitor
from .types.vector.vector6d import Vector6D
from .types.detection import Detection
from .utils.view_publisher import ViewPublisher
from .types.shape.box import Box


class TabletopPipeline(BasePipeline):
    def __init__(self):
        super(TabletopPipeline, self).__init__()

    def initialize_pipeline(self, internal_simulator, beliefs_base):
        """ """
        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        hand_detector_model_filename = rospy.get_param("~hand_detector_model_filename", "")
        hand_detector_weights_filename = rospy.get_param("~hand_detector_weights_filename", "")
        hand_detector_config_filename = rospy.get_param("~hand_detector_config_filename", "")

        self.person_detector = SSDDetector(detector_model_filename,
                                           detector_weights_filename,
                                           detector_config_filename,
                                           300)

        self.track_hands = rospy.get_param("~track_hands", False)

        if self.track_hands is True:
            self.hand_detector = SSDDetector(hand_detector_model_filename,
                                             hand_detector_weights_filename,
                                             hand_detector_config_filename,
                                             300)

        self.foreground_detector = ForegroundDetector()

        self.color_features_estimator = ColorFeaturesEstimator()

        self.n_init = rospy.get_param("~n_init", 1)
        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.8)
        self.max_color_distance = rospy.get_param("~max_color_distance", 0.2)
        self.max_centroid_distance = rospy.get_param("~max_centroid_distance", 0.8)
        self.max_disappeared = rospy.get_param("~max_disappeared", 7)
        self.max_age = rospy.get_param("~max_age", 10)

        self.object_tracker = MultiObjectTracker(iou_cost,
                                                 centroid_cost,
                                                 self.max_iou_distance,
                                                 None,
                                                 self.n_init,
                                                 60,
                                                 120,
                                                 use_tracker=True)

        if self.track_hands is True:
            self.hand_tracker = MultiObjectTracker(iou_cost,
                                                   color_cost,
                                                   0.98,
                                                   self.max_color_distance,
                                                   1,
                                                   self.max_disappeared,
                                                   self.max_age,
                                                   use_tracker=True)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 0.98,
                                                 self.max_color_distance,
                                                 1,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 use_tracker=True)

        self.table = None

        self.shape_estimator = ShapeEstimator()

        self.object_pose_estimator = ObjectPoseEstimator()

        self.action_monitor = TabletopActionMonitor(internal_simulator)
        ########################################################
        # Visualization
        ########################################################

        self.other_view_publisher = ViewPublisher("other_view")
        self.myself_view_publisher = ViewPublisher("myself_view")

        self.events = []

    def get_point_measure(self, rgb_image, depth_image, x, y, camera):
        depth_height, depth_width = depth_image.shape
        camera_matrix = camera.camera_matrix()
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        cx = camera_matrix[0][2]
        cy = camera_matrix[1][2]
        if not math.isnan(depth_image[y, x]):
            z = depth_image[y, x]/1000.0
            x = (x - cx) * z / fx
            y = (y - cy) * z / fy
            return Vector6D(x=x, y=y, z=z)
        return None

    def create_table(self, rgb_image, depth_image, view_pose, camera):
        """
        """
        if depth_image is None:
            return False

        image_height, image_width, _ = rgb_image.shape
        if depth_image is not None:
            x = int(image_width*(1/2.0))
            y = int(image_height*(3/4.0))
            if not math.isnan(depth_image[y, x]):
                table_depth = depth_image[y, x]/1000.0
        else:
            return False
        table_detection = Detection(0, int(image_height/2.0), int(image_width), image_height, "table", 1.0, table_depth)

        x = int(image_width*(1/2.0))
        y = int(image_height*(3/4.0))
        table_center = self.get_point_measure(rgb_image, depth_image, x, y, camera)
        if table_center is None:
            return False
        x = int(image_width*(1/2.0))
        y = int(image_height*(1/2.0))
        table_top_border = self.get_point_measure(rgb_image, depth_image, x, y, camera)
        if table_top_border is None:
            return False
        x = int(image_width*(1/2.0))
        y = int(image_height-15)
        table_bottom_border = self.get_point_measure(rgb_image, depth_image, x, y, camera)
        if table_bottom_border is None:
            return False

        center_to_top = math.sqrt(pow(table_top_border.pos.x-table_center.pos.x, 2) +
                                  pow(table_top_border.pos.y-table_center.pos.y, 2) +
                                  pow(table_top_border.pos.z-table_center.pos.z, 2))
        table_length = math.sqrt(pow(table_top_border.pos.x-table_bottom_border.pos.x, 2) +
                                 pow(table_top_border.pos.y-table_bottom_border.pos.y, 2) +
                                 pow(table_top_border.pos.z-table_bottom_border.pos.z, 2))

        table_width = table_length*((1+math.sqrt(5))/2.0) # assume a perfect rectangle
        center_in_world = view_pose+table_center
        real_z = center_in_world.position().z

        table_shape = Box(table_width, table_length, real_z, z=-real_z/2.0, y=center_to_top-table_length/2.0)
        self.table = SceneNode(detection=table_detection, is_static=True)
        self.table.shapes = [table_shape]
        self.table.update_pose(center_in_world.position())

        return True

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
        detection_timer = cv2.getTickCount()
        detections = []

        image_height, image_width, _ = rgb_image.shape

        if depth_image is not None:
            if self.table is None:
                self.create_table(rgb_image, depth_image, view_pose, self.robot_camera)

        if self.frame_count == 0:
            person_detections = self.person_detector.detect(rgb_image, depth_image=depth_image)
            detections = person_detections
        elif self.frame_count == 1:
            object_detections = self.foreground_detector.detect(rgb_image, depth_image=depth_image)
            detections = object_detections
        elif self.frame_count == 2:
            if self.track_hands is True:
                hand_detections = self.hand_detector.detect(rgb_image, depth_image=depth_image)
                detections = hand_detections
            else:
                detections = []
        else:
            detections = []

        detection_fps = cv2.getTickFrequency() / (cv2.getTickCount()-detection_timer)
        #print "detection: {:.2}hz".format(detection_fps)
        ####################################################################
        # Features estimation
        ####################################################################
        self.color_features_estimator.estimate(rgb_image, detections)
        ######################################################
        # Tracking
        ######################################################

        if self.frame_count == 0:
            self.object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
            self.person_tracks = self.person_tracker.update(rgb_image, person_detections, depth_image=depth_image)
            if self.track_hands is True:
                self.hand_tracks = self.hand_tracker.update(rgb_image, [], depth_image=depth_image)
        elif self.frame_count == 1:
            self.object_tracks = self.object_tracker.update(rgb_image, object_detections, depth_image=depth_image)
            self.person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)
            if self.track_hands is True:
                self.hand_tracks = self.hand_tracker.update(rgb_image, [], depth_image=depth_image)
        elif self.frame_count == 2:
            self.object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
            self.person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)
            if self.track_hands is True:
                self.hand_tracks = self.hand_tracker.update(rgb_image, hand_detections, depth_image=depth_image)
        else:
            self.object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
            self.person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)
            if self.track_hands is True:
                self.hand_tracks = self.hand_tracker.update(rgb_image, [], depth_image=depth_image)

        if self.table is not None:
            support_tracks = [self.table]
        else:
            support_tracks = []

        if self.track_hands is True:
            tracks = self.object_tracks + self.person_tracks + self.hand_tracks + support_tracks
        else:
            tracks = self.object_tracks + self.person_tracks + support_tracks

        ########################################################
        # Pose & Shape estimation
        ########################################################

        self.object_pose_estimator.estimate(self.object_tracks + self.person_tracks , view_pose, self.robot_camera)

        self.shape_estimator.estimate(rgb_image, tracks, self.robot_camera)

        ########################################################
        # Monitoring
        ########################################################

        corrected_object_tracks, self.events = self.action_monitor.monitor(support_tracks, self.object_tracks, self.person_tracks, [])

        corrected_tracks = corrected_object_tracks + self.person_tracks + support_tracks

        pipeline_fps = cv2.getTickFrequency() / (cv2.getTickCount()-pipeline_timer)
        ########################################################
        # Visualization
        ########################################################
        self.myself_view_publisher.publish(rgb_image, corrected_tracks, events=self.events, overlay_image=None, fps=pipeline_fps)

        all_nodes = [myself]+static_nodes+corrected_tracks
        return all_nodes, self.events
