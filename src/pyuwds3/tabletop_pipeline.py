import rospy
import math
from .base_pipeline import BasePipeline
from .reasoning.detection.ssd_detector import SSDDetector
from .reasoning.detection.foreground_detector import ForegroundDetector
from .reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost
from .reasoning.tracking.track import Track
from .reasoning.estimation.object_pose_estimator import ObjectPoseEstimator
from .reasoning.estimation.shape_estimator import ShapeEstimator
from .reasoning.estimation.color_features_estimator import ColorFeaturesEstimator
from .reasoning.monitoring.tabletop_action_monitor import TabletopActionMonitor
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.detection import Detection


class TabletopPipeline(BasePipeline):
    def __init__(self):
        super(TabletopPipeline, self).__init__()

    def initialize_pipeline(self):
        """ """
        detector_model_filename = rospy.get_param("~detector_model_filename", "")
        detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        detector_config_filename = rospy.get_param("~detector_config_filename", "")

        self.person_detector = SSDDetector(detector_model_filename,
                                           detector_weights_filename,
                                           detector_config_filename,
                                           300)

        self.foreground_detector = ForegroundDetector(interactive_mode=False)

        self.color_features_estimator = ColorFeaturesEstimator()

        self.n_init = rospy.get_param("~n_init", 1)
        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.8)
        self.max_color_distance = rospy.get_param("~max_color_distance", 0.2)
        self.max_centroid_distance = rospy.get_param("~max_centroid_distance", 0.8)
        self.max_disappeared = rospy.get_param("~max_disappeared", 7)
        self.max_age = rospy.get_param("~max_age", 10)

        self.object_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 0.35,
                                                 0.3,
                                                 self.n_init,
                                                 40,
                                                 self.max_age,
                                                 use_appearance_tracker=True)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 color_cost,
                                                 self.max_iou_distance,
                                                 self.max_color_distance,
                                                 5,
                                                 self.max_disappeared,
                                                 self.max_age,
                                                 use_appearance_tracker=True)

        self.table_track = None
        self.table_depth = None
        self.table_shape = None

        self.shape_estimator = ShapeEstimator()

        self.object_pose_estimator = ObjectPoseEstimator()

        self.action_monitor = TabletopActionMonitor()

    def perception_pipeline(self, view_matrix, rgb_image, depth_image=None, time=None):
        """ """
        ######################################################
        # Simulation
        ######################################################
        myself = self.internal_simulator.get_myself()

        static_nodes = self.internal_simulator.get_static_entities()

        ######################################################
        # Detection
        ######################################################

        detections = []

        image_height, image_width, _ = rgb_image.shape

        if depth_image is not None:
            if self.table_depth is None:
                x = int(image_width*(1/2.0))
                y = int(image_height*(3/4.0))
                if not math.isnan(depth_image[y, x]):
                    self.table_depth = depth_image[y, x]/1000.0
            table_depth = self.table_depth
        else:
            table_depth = None
        table_detection = Detection(0, int(image_height/2.0), int(image_width), image_height, "table", 1.0, table_depth)

        view_pose = Vector6D().from_transform(view_matrix)
        if depth_image is not None:
            if self.table_shape is None:
                fx = self.camera_matrix[0][0]
                fy = self.camera_matrix[1][1]
                cx = self.camera_matrix[0][2]
                cy = self.camera_matrix[1][2]
                x = int(image_width*(1/2.0))
                y = int(image_height*(3/4.0))
                if not math.isnan(depth_image[y, x]):
                    z = depth_image[y, x]/1000.0
                    x = (x - cx) * z / fx
                    y = (y - cy) * z / fy
                    table_border = Vector6D(x=x, y=y, z=z)
                    x = int(image_width*(1/2.0))
                    y = int(image_height*(1/2.0))
                    if not math.isnan(depth_image[y, x]):
                        z = depth_image[y, x]/1000.0
                        x = (x - cx) * z / fx
                        y = (y - cy) * z / fy
                        table_center = Vector6D(x=x, y=y, z=z)
                        table_length = (image_width - cx) * z / fx
                        table_width = 2.0*math.sqrt(pow(table_border.pos.x-table_center.pos.x, 2)+pow(table_border.pos.y-table_center.pos.y, 2)+pow(table_border.pos.z-table_center.pos.z, 2))
                        center_in_world = view_pose+table_center
                        real_z = center_in_world.position().z
                        print center_in_world.position()
                        self.table_shape = Box(table_width, table_length, real_z)
                        self.table_shape.pose.pos.z = -real_z/2.0
                        self.table_shape.color = self.shape_estimator.compute_dominant_color(rgb_image, table_detection.bbox)

        if self.frame_count == 0:
            person_detections = self.person_detector.detect(rgb_image, depth_image=depth_image)
            detections = person_detections
        elif self.frame_count == 1:
            object_detections = self.foreground_detector.detect(rgb_image, depth_image=depth_image)
            detections = object_detections
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
            object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
            person_tracks = self.person_tracker.update(rgb_image, person_detections, depth_image=depth_image)
        elif self.frame_count == 1:
            object_tracks = self.object_tracker.update(rgb_image, object_detections, depth_image=depth_image)
            person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)
        else:
            object_tracks = self.object_tracker.update(rgb_image, [], depth_image=depth_image)
            person_tracks = self.person_tracker.update(rgb_image, [], depth_image=depth_image)

        if self.table_track is None:
            self.table_track = Track(table_detection, 1, 4, 20)
        else:
            self.table_track.update(table_detection)
        if self.table_shape is not None:
            self.table_track.shapes = [self.table_shape]
        support_tracks = [self.table_track]

        tracks = object_tracks + person_tracks + support_tracks

        ########################################################
        # Pose & Shape estimation
        ########################################################

        self.object_pose_estimator.estimate(object_tracks + person_tracks + support_tracks, view_matrix, self.camera_matrix, self.dist_coeffs)

        self.shape_estimator.estimate(rgb_image, tracks, self.camera_matrix, self.dist_coeffs)

        ########################################################
        # Monitoring
        ########################################################

        events = self.action_monitor.monitor(support_tracks, object_tracks, person_tracks, [])

        all_nodes = [myself]+static_nodes+tracks
        return all_nodes, events
