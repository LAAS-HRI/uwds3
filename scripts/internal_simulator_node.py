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
from pyuwds3.reasoning.monitoring.graphic_monitor import GraphicMonitor
from pyuwds3.reasoning.monitoring.perspective_monitor import PerspectiveMonitor
from pyuwds3.reasoning.monitoring.heatmap import Heatmap
from pyuwds3.types.vector.vector6d_stable import Vector6DStable
from pyuwds3.types.shape.mesh import Mesh
DEFAULT_SENSOR_QUEUE_SIZE = 3


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

        self.use_ar_tags = rospy.get_param("~use_ar_tags", True)
        self.ar_tags_topic = rospy.get_param("ar_tags_topic", "ar_tracks")
        self.ar_tags_topic="ar_tracks"
        self.use_ar_tags = True
        if self.use_ar_tags is True:
            self.ar_tags_tracks = []
            self.ar_tags_sub = rospy.Subscriber(self.ar_tags_topic, WorldStamped, self.ar_tags_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)


        self.use_motion_capture = rospy.get_param("~use_motion_capture", False)
        self.motion_capture_topic = rospy.get_param("~motion_capture_topic", "motion_capture_tracks")
        self.use_motion_capture=True
        self.motion_capture_topic="mocap_tracks"
        if self.use_motion_capture is True:
            self.motion_capture_tracks = []
            self.motion_capture_sub = rospy.Subscriber(self.motion_capture_topic, WorldStamped, self.motion_capture_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_object_perception = rospy.get_param("~use_object_perception", True)
        self.object_perception_topic = rospy.get_param("object_perception_topic", "tabletop_object_tracks")
        if self.use_object_perception is True:
            self.object_tracks = []
            self.object_sub = rospy.Subscriber(self.object_perception_topic, WorldStamped, self.object_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)

        self.use_human_perception = rospy.get_param("~use_human_perception", True)
        self.human_perception_topic = rospy.get_param("human_perception_topic", "human_tracks")
        if self.use_human_perception is True:
            self.human_tracks = []
            self.human_sub = rospy.Subscriber(self.human_perception_topic, WorldStamped, self.human_perception_callback, queue_size=DEFAULT_SENSOR_QUEUE_SIZE)





        self.publish_tf = rospy.get_param("~publish_tf", False)
        self.publish_viz = rospy.get_param("~publish_viz", True)
        # self.publish_markers = rospy.get_param("~publish_markers", True)

        self.world_publisher = WorldPublisher("corrected_tracks")
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
        self.use_heatmap_monitoring = False
        self.use_physical_monitoring = rospy.get_param("~use_physical_monitoring", True)
        if self.use_physical_monitoring is True:
            self.physics_monitor = PhysicsMonitor(self.internal_simulator)
        self.use_graphic_monitoring=rospy.get_param("~use_graphic_monitoring",42)
        self.use_graphic_monitoring = True
        print self.use_graphic_monitoring
        if self.use_graphic_monitoring is True:
            # bodynode = SceneNode(pose=Vector6DStable(x=-2,y=6,z=-9,rx=1.7, ry=3.54, rz=0))
            # headnode = SceneNode(pose=Vector6DStable(x=-2,y=4.6,z=2.2,rx=1.7, ry=.42, rz=0))
            # self.cad_models_search_path = rospy.get_param("~cad_models_search_path", "")
            # # mesh_path = "/home/abonneau/catkin_ws/src/exp_director_task/mesh/obj/dt_cube.obj"
            # mesh_path = "package://exp_director_task/mesh/ikea_console.stl"
            # shape = Mesh(mesh_path,
            #              x=0, y=0, z=0,
            #              rx=0, ry=0, rz=0)
            # shape.color[0] = 1
            # shape.color[1] = 1
            # shape.color[2] = 1
            # shape.color[3] = 1
            # bodynode.shapes.append(shape)
            # self.internal_simulator.load_node(bodynode)
            # self.internal_simulator.load_node(headnode)
            self.physics_monitor = GraphicMonitor(internal_simulator=self.internal_simulator)#,head=headnode)

        self.use_perspective_monitoring = rospy.get_param("~use_perspective_monitoring", True)
        self.use_perspective_monitoring = False
        self.use_heatmap_monitoring = True
        self.heatmap_monitor = Heatmap(self.internal_simulator)
        if self.use_perspective_monitoring is True:
            self.perspective_monitor = PerspectiveMonitor(self.internal_simulator, None)

        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")
        rospy.loginfo("[internal_simulator] Subscribing to '/{}' topic...".format(self.rgb_camera_info_topic))
        self.camera_info_subscriber = rospy.Subscriber(self.rgb_camera_info_topic, CameraInfo, self.camera_info_callback)


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
            # print self.internal_simulator.entity_id_map
            ar_tags_tracks.append(SceneNode().from_msg(node))
        self.ar_tags_tracks = ar_tags_tracks
        #world_msg.header.frame_id[1:]
        if world_msg.header.frame_id[1:] != '':
            s,pose =self.tf_bridge.get_pose_from_tf(self.global_frame_id, world_msg.header.frame_id[1:])
        else:
            pose=None
        self.physics_monitor.monitor(ar_tags_tracks, pose, world_msg.header)

    def motion_capture_callback(self, world_msg):
        motion_capture_tracks = []
        for node in world_msg.world.scene:
            motion_capture_tracks.append(SceneNode().from_msg(node))
        self.physics_monitor.mocap(motion_capture_tracks,world_msg.header)

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
            header = msg.header
            header.frame_id = self.global_frame_id
            self.frame_count %= self.n_frame

            if self.use_perspective_monitoring is True:
                if self.frame_count == 3:
                    monitoring_timer = cv2.getTickCount()
                    face_tracks = [f for f in self.human_tracks if f.label == "face"]
                    success, other_image, other_visible_tracks, self.events = self.perspective_monitor.monitor_others(face_tracks)
                    monitoring_fps = cv2.getTickFrequency() / (cv2.getTickCount()-monitoring_timer)
                    if success:
                        self.other_view_publisher.publish(other_image, other_visible_tracks, header.stamp, fps=monitoring_fps)
            if self.use_heatmap_monitoring  is True:
                if (self.frame_count == 3) or True:
                    # monitoring_timer = cv2.getTickCount()
                    success, view_pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.camera_frame_id)
                    # monitoring_fps = cv2.getTickFrequency() / (cv2.getTickCount()-monitoring_timer)
                    if success:
                        self.heatmap_monitor.update_heatmaps(view_pose,self.robot_camera)
                        self.heatmap_monitor.show_heatmap(view_pose,self.robot_camera,header.stamp,1)

                        # self.heatmap_monitor.show_heatmap(view_pose,self.robot_camera,header.stamp, fps=monitoring_fps)

            object_tracks = self.ar_tags_tracks + self.object_tracks
            person_tracks = [f for f in self.human_tracks if f.label == "person"]

            corrected_object_tracks, action_events = self.physics_monitor.monitor(object_tracks, person_tracks, header.stamp)

            corrected_tracks = self.internal_simulator.get_static_entities() + self.human_tracks + corrected_object_tracks

            self.world_publisher.publish(corrected_tracks, action_events, header)

            if self.publish_tf is True:
                self.tf_bridge.publish_tf_frames(corrected_tracks, action_events, header)

            if self.publish_markers is True:
                self.marker_publisher.publish(corrected_tracks, header)

            self.frame_count += 1

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("internal_simulator", anonymous=False)
    perception = InternalSimulatorNode().run()
