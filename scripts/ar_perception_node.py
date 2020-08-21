#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from pyuwds3.types.camera import Camera
from uwds3_msgs.msg import SceneChangesStamped
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.utils.view_publisher import ViewPublisher
from pyuwds3.utils.marker_publisher import MarkerPublisher
from pyuwds3.reasoning.estimation.object_pose_estimator import ObjectPoseEstimator
from pyuwds3.reasoning.estimation.shape_estimator import ShapeEstimator
from pyuwds3.reasoning.estimation.head_pose_estimator import HeadPoseEstimator
from pyuwds3.reasoning.detection.ssd_detector import SSDDetector
from pyuwds3.reasoning.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from pyuwds3.reasoning.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, centroid_cost

DEFAULT_SENSOR_QUEUE_SIZE = 1


class ArPerceptionNode(object):
    def __init__(self):
        """
        """

        self.tf_bridge = TfBridge()

        self.n_frame = rospy.get_param("~n_frame", 4)
        self.frame_count = 0

        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        person_detector_weights_filename = rospy.get_param("~person_detector_weights_filename", "")
        person_detector_model_filename = rospy.get_param("~person_detector_model_filename", "")
        person_detector_config_filename = rospy.get_param("~person_detector_config_filename", "")

        face_detector_weights_filename = rospy.get_param("~face_detector_weights_filename", "")
        face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.face_detector = SSDDetector(face_detector_weights_filename,
                                         face_detector_model_filename,
                                         face_detector_config_filename)

        self.person_detector = SSDDetector(person_detector_weights_filename,
                                           person_detector_model_filename,
                                           person_detector_config_filename)

        shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")

        face_3d_model_filename = rospy.get_param("~face_3d_model_filename", "")
        self.head_pose_estimator = HeadPoseEstimator(face_3d_model_filename)
        self.facial_landmarks_estimator = FacialLandmarksEstimator(shape_predictor_config_filename)

        self.n_init = rospy.get_param("~n_init", 1)

        self.max_iou_distance = rospy.get_param("~max_iou_distance", 0.98)
        self.max_face_distance = rospy.get_param("~max_face_distance", 0.2)

        self.max_lost = rospy.get_param("~max_lost", 4)
        self.max_age = rospy.get_param("~max_age", 12)

        self.bridge = CvBridge()
        self.robot_camera = None
        self.camera_info = None

        self.events = []

        self.robot_camera_clipnear = rospy.get_param("~robot_camera_clipnear", 0.1)
        self.robot_camera_clipfar = rospy.get_param("~robot_camera_clipfar", 25.0)

        self.face_tracker = MultiObjectTracker(iou_cost,
                                               centroid_cost,
                                               0.98,
                                               None,
                                               5,
                                               60,
                                               120,
                                               use_tracker=True)

        self.person_tracker = MultiObjectTracker(iou_cost,
                                                 centroid_cost,
                                                 0.98,
                                                 None,
                                                 self.n_init,
                                                 self.max_lost,
                                                 self.max_age)

        self.shape_estimator = ShapeEstimator()
        self.object_pose_estimator = ObjectPoseEstimator()

        self.publish_tf = rospy.get_param("~publish_tf", False)

        self.publish_viz = rospy.get_param("~publish_viz", True)

        self.scene_publisher = rospy.Publisher("human_tracks", SceneChangesStamped, queue_size=1)
        self.view_publisher = ViewPublisher("human_perception")
        self.marker_publisher = MarkerPublisher("human_markers")

        self.use_depth = rospy.get_param("~use_depth", False)
        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        rospy.loginfo("[perception] Subscribing to '/{}' topic...".format(self.rgb_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.rgb_camera_info_topic, CameraInfo, self.camera_info_callback)

        ar_pose_marker_sub = rospy.Subscriber("ar_pose_marker", )


    def camera_info_callback(self, msg):
        """ """
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.robot_camera = Camera().from_msg(msg,
                                              clipnear=self.robot_camera_clipnear,
                                              clipfar=self.robot_camera_clipfar)

    def observation_callback(self, bgr_image_msg, depth_image_msg=None):
        """
        """
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
                rospy.logwarn("[human_perception] The camera sensor is not localized in world space (frame '{}'), please check if the sensor frame is published in /tf".format(self.global_frame_id))
            else:
                self.frame_count %= self.n_frame
                all_nodes, events = self.perception_pipeline(view_pose, rgb_image, depth_image=depth_image, time=header.stamp)

                self.publish_changes(all_nodes, events, header)

                if self.publish_viz is True:
                    self.marker_publisher.publish(all_nodes, header)

                if self.publish_tf is True:
                    self.tf_bridge.publish_tf_frames(all_nodes, events, header)

                self.frame_count += 1

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

    def perception_pipeline(self, view_pose, rgb_image, depth_image=None, time=None):
        ######################################################
        # Detection
        ######################################################
        pipeline_timer = cv2.getTickCount()

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

        #self.color_features_estimator.estimate(rgb_image, detections)

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

        pipeline_fps = cv2.getTickFrequency() / (cv2.getTickCount()-pipeline_timer)
        ########################################################
        # Visualization
        ########################################################
        if self.publish_tf is True:
            self.view_publisher.publish(rgb_image, tracks, overlay_image=None, fps=pipeline_fps, view_pose=view_pose, camera=self.robot_camera)

        all_nodes = tracks
        return all_nodes, self.events

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("human_perception", anonymous=False)
    perception = HumanPerceptionNode().run()
