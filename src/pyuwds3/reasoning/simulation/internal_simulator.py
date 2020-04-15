import rospy
import cv2
import pybullet as p
import numpy as np
from sensor_msgs.msg import JointState
from pyuwds3.types.scene_node import SceneNode, SceneNodeType
# from uwds3_msgs.msg import SceneChangesStamped, PrimitiveShape
# from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix, translation_from_matrix
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.shape.box import Box
from pyuwds3.types.shape.cylinder import Cylinder
from pyuwds3.types.shape.sphere import Sphere
from pyuwds3.types.shape.mesh import Mesh
from pyuwds3.types.detection import Detection
import yaml


class InternalSimulator(object):
    def __init__(self,
                 use_gui,
                 cad_models_additional_search_path,
                 env_urdf_file_path,
                 static_entities_config_filename,
                 robot_urdf_file_path,
                 global_frame_id,
                 base_frame_id,
                 position_tolerance=0.005,
                 simulation_step=(1./100.0),
                 load_robot=True):

        self.tf_bridge = TfBridge()

        self.entity_map = {}

        self.entity_id_map = {}
        self.reverse_entity_id_map = {}

        self.joint_id_map = {}
        self.reverse_joint_id_map = {}

        self.constraint_id_map = {}

        self.markers_id_map = {}

        self.robot_joints_command = []
        self.robot_joints_command_indices = []

        self.position_tolerance = position_tolerance

        self.global_frame_id = global_frame_id
        self.base_frame_id = base_frame_id

        self.robot_urdf_file_path = robot_urdf_file_path

        self.use_gui = use_gui
        if self.use_gui is True:
            self.client_simulator_id = p.connect(p.GUI)
        else:
            self.client_simulator_id = p.connect(p.DIRECT)

        p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

        if cad_models_additional_search_path != "":
            p.setAdditionalSearchPath(cad_models_additional_search_path)

        self.static_nodes = []

        self.simulation_step = simulation_step
        p.setTimeStep(self.simulation_step)

        if static_entities_config_filename != "":
            with open(static_entities_config_filename, 'r') as stream:
                static_entities = yaml.load(stream)
                for entity in static_entities:
                    start_pose = Vector6D(x=entity["position"]["x"],
                                          y=entity["position"]["y"],
                                          z=entity["position"]["z"],
                                          rx=entity["orientation"]["x"],
                                          ry=entity["orientation"]["x"],
                                          rz=entity["orientation"]["z"])

                    success, static_node = self.load_urdf(entity["id"],
                                                          entity["file"],
                                                          start_pose,
                                                          label=entity["label"],
                                                          description=entity["description"],
                                                          static=True)
                    if success:
                        self.static_nodes.append(static_node)

        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        self.simulation_timer = rospy.Timer(rospy.Duration(self.simulation_step*3.0), self.step_simulation)

        self.robot_loaded = False
        if load_robot is True:
            self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size=1)

    def load_urdf(self,
                  id,
                  filename,
                  start_pose,
                  remove_friction=False,
                  static=False,
                  label="thing",
                  description=""):
        """ """
        try:
            use_fixed_base = 1 if static is True else 0
            base_link_sim_id = p.loadURDF(filename, start_pose.position().to_array(), start_pose.quaternion(), useFixedBase=use_fixed_base, flags=p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)

            self.entity_id_map[id] = base_link_sim_id
            # Create a joint map to ease exploration
            self.reverse_entity_id_map[base_link_sim_id] = id
            self.joint_id_map[base_link_sim_id] = {}
            self.reverse_joint_id_map[base_link_sim_id] = {}
            for i in range(0, p.getNumJoints(base_link_sim_id)):
                info = p.getJointInfo(base_link_sim_id, i)
                self.joint_id_map[base_link_sim_id][info[1]] = info[0]
                self.reverse_joint_id_map[base_link_sim_id][info[0]] = info[1]
            # If file successfully loaded
            if base_link_sim_id < 0:
                raise RuntimeError("Invalid URDF")
            scene_node = SceneNode(pose=start_pose, is_static=True)
            scene_node.id = id
            scene_node.label = label
            scene_node.description = description
            sim_id = self.entity_id_map[id]
            visual_shapes = p.getVisualShapeData(sim_id)
            for shape in visual_shapes:
                link_id = shape[1]
                type = shape[2]
                dimensions = shape[3]
                mesh_file_path = shape[4]
                position = shape[5]
                orientation = shape[6]
                rgba_color = shape[7]

                if link_id != -1:
                    link_state = p.getLinkState(sim_id, link_id)
                    t_link = link_state[0]
                    q_link = link_state[1]
                    t_inertial = link_state[2]
                    q_inertial = link_state[3]

                    tf_world_link = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))
                    tf_inertial_link = np.dot(translation_matrix(t_inertial), quaternion_matrix(q_inertial))
                    world_transform = np.dot(tf_world_link, np.linalg.inv(tf_inertial_link))

                else:
                    t_link, q_link = p.getBasePositionAndOrientation(sim_id)
                    world_transform = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))

                if type == p.GEOM_SPHERE:
                    primitive_shape = Sphere(dimensions[0]*2.0)
                elif type == p.GEOM_BOX:
                    primitive_shape = Box(dimensions[0], dimensions[1], dimensions[2])
                elif type == p.GEOM_CYLINDER:
                    primitive_shape = Cylinder(dimensions[1]*2.0, dimensions[0])
                elif type == p.GEOM_PLANE:
                    primitive_shape = Box(dimensions[0], dimensions[1], 0.0001)
                elif type == p.GEOM_MESH:
                    primitive_shape = Mesh("file://"+mesh_file_path,
                                           scale_x=dimensions[0],
                                           scale_y=dimensions[1],
                                           scale_z=dimensions[2])
                else:
                    raise NotImplementedError("Shape capsule not supported at the moment")

                if link_id != -1:
                    shape_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                    shape_transform = np.dot(world_transform, shape_transform)
                    shape_transform = np.linalg.inv(np.dot(np.linalg.inv(shape_transform), scene_node.pose.transform()))
                    position = translation_from_matrix(shape_transform)
                    orientation = quaternion_from_matrix(shape_transform)

                    primitive_shape.pose.pos.x = position[0]
                    primitive_shape.pose.pos.y = position[1]
                    primitive_shape.pose.pos.z = position[2]

                    primitive_shape.pose.from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
                else:
                    shape_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                    shape_transform = np.dot(world_transform, shape_transform)
                    position = translation_from_matrix(shape_transform)
                    orientation = quaternion_from_matrix(shape_transform)

                    scene_node.pose.pos.x = position[0]
                    scene_node.pose.pos.y = position[1]
                    scene_node.pose.pos.z = position[2]

                    scene_node.pose.from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])

                if len(rgba_color) > 0:
                    primitive_shape.color[0] = rgba_color[0]
                    primitive_shape.color[1] = rgba_color[1]
                    primitive_shape.color[2] = rgba_color[2]
                    primitive_shape.color[3] = rgba_color[3]

                scene_node.shapes.append(primitive_shape)
            self.entity_map[id] = scene_node
            return True, scene_node
            rospy.loginfo("[simulation] '{}' File successfully loaded".format(filename))
        except Exception as e:
            rospy.logwarn("[simulation] Error loading URDF '{}': {}".format(filename, e))
            return False, None

    def get_myself(self):
        node = self.get_entity("myself")
        node.type = SceneNodeType.MYSELF
        node.label = "robot"
        node.description = "I"
        return node

    def get_static_entities(self):
        return self.static_nodes

    def get_entity(self, id):
        if id not in self.entity_map:
            raise ValueError("Invalid id provided : '{}'".format(id))
        scene_node = self.entity_map[id]
        sim_id = self.entity_id_map[id]
        visual_shapes = p.getVisualShapeData(sim_id)
        for idx, shape in enumerate(visual_shapes):
            primitive_shape = scene_node.shapes[idx]
            link_id = shape[1]
            position = shape[5]
            orientation = shape[6]

            if link_id != -1:
                link_state = p.getLinkState(sim_id, link_id)
                t_link = link_state[0]
                q_link = link_state[1]
                t_inertial = link_state[2]
                q_inertial = link_state[3]

                tf_world_link = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))
                tf_inertial_link = np.dot(translation_matrix(t_inertial), quaternion_matrix(q_inertial))
                world_transform = np.dot(tf_world_link, np.linalg.inv(tf_inertial_link))

            else:
                t_link, q_link = p.getBasePositionAndOrientation(sim_id)
                world_transform = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))

            if link_id != -1:
                shape_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                shape_transform = np.dot(world_transform, shape_transform)
                shape_transform = np.linalg.inv(np.dot(np.linalg.inv(shape_transform), scene_node.pose.transform()))
                position = translation_from_matrix(shape_transform)
                orientation = quaternion_from_matrix(shape_transform)

                primitive_shape.pose.pos.x = position[0]
                primitive_shape.pose.pos.y = position[1]
                primitive_shape.pose.pos.z = position[2]

                primitive_shape.pose.from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
            else:
                shape_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                shape_transform = np.dot(world_transform, shape_transform)
                position = translation_from_matrix(shape_transform)
                orientation = quaternion_from_matrix(shape_transform)

                scene_node.pose.pos.x = position[0]
                scene_node.pose.pos.y = position[1]
                scene_node.pose.pos.z = position[2]

                scene_node.pose.from_quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
        return self.entity_map[id]

    def step_simulation(self, event):
        if self.robot_loaded is True:
            base_link_sim_id = self.entity_id_map["myself"]
            p.setJointMotorControlArray(base_link_sim_id,
                                        self.robot_joints_command_indices,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.robot_joints_command,
                                        forces=np.full(len(self.robot_joints_command), 10000.0),
                                        physicsClientId=self.client_simulator_id)
        p.stepSimulation()

    def get_camera_view(self, camera_pose, camera, prior_tracks=[], target_position=None, occlusion_threshold=0.01, rendering_ratio=1.0):
        visible_tracks = []
        rot = quaternion_matrix(camera_pose.quaternion())
        trans = translation_matrix(camera_pose.position().to_array().flatten())
        if target_position is None:
            target = translation_matrix([0.0, 0.0, 1000.0])
            target = translation_from_matrix(np.dot(np.dot(trans, rot), target))
        else:
            target = target_position.position().to_array()
        view_matrix = p.computeViewMatrix(camera_pose.position().to_array(), target, [0, 0, 1])

        width = camera.width
        height = camera.height

        rendered_width = int(width*rendering_ratio)
        rendered_height = int(height*rendering_ratio)

        projection_matrix = p.computeProjectionMatrixFOV(camera.fov,
                                                         float(rendered_width)/rendered_height,
                                                         camera.clipnear,
                                                         camera.clipfar)

        if self.use_gui is True:
            camera_image = p.getCameraImage(rendered_width,
                                            rendered_height,
                                            viewMatrix=view_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            projectionMatrix=projection_matrix)
        else:
            camera_image = p.getCameraImage(rendered_width,
                                            rendered_height,
                                            viewMatrix=view_matrix,
                                            renderer=p.ER_TINY_RENDERER,
                                            projectionMatrix=projection_matrix)

        rgb_image = cv2.resize(np.array(camera_image[2]), (width, height))[:,:,:3]
        depth_image = np.array(camera_image[3], np.float32).reshape((rendered_height, rendered_width))

        far = camera.clipfar
        near = camera.clipnear
        real_depth_image = far * near / (far - (far - near) * depth_image)

        mask_image = camera_image[4]
        mask_image_resized = cv2.resize(np.array(camera_image[4]).copy().astype("uint8"), (width, height))
        unique, counts = np.unique(np.array(mask_image).flatten(), return_counts=True)

        for sim_id, count in zip(unique, counts):
            if sim_id > 0:
                cv_mask = np.array(mask_image.copy())
                cv_mask[cv_mask != sim_id] = 0
                cv_mask[cv_mask == sim_id] = 255
                xmin, ymin, w, h = cv2.boundingRect(cv_mask.astype(np.uint8))
                mask = cv_mask[ymin:ymin+h, xmin:xmin+w]
                visible_area = w*h+1
                screen_area = rendered_width*rendered_height+1
                if screen_area-visible_area == 0:
                    confidence = 1.0
                else:
                    confidence = visible_area/float(screen_area-visible_area)
                #TODO compute occlusion score as a ratio between visible 2d bbox and projected 2d bbox areas
                if confidence > occlusion_threshold:

                    depth = real_depth_image[int(ymin+h/2.0)][int(xmin+w/2.0)]
                    xmin = int(xmin/rendering_ratio)
                    ymin = int(ymin/rendering_ratio)
                    w = int(w/rendering_ratio)
                    h = int(h/rendering_ratio)

                    id = self.reverse_entity_id_map[sim_id]
                    scene_node = self.get_entity(id)

                    det = Detection(int(xmin), int(ymin), int(xmin+w), int(ymin+h), id, confidence, depth=depth, mask=mask)
                    track = SceneNode(detection=det)
                    track.static = scene_node.static
                    track.id = id
                    track.mask = det.mask
                    track.shapes = scene_node.shapes
                    track.pose = scene_node.pose
                    track.label = scene_node.label
                    track.description = scene_node.description
                    visible_tracks.append(track)

        # prior_tracks_cleaned = [t for t in prior_tracks if t.has_mask() is True and t.is_confirmed() is True and t.is_located() is True]
        # if len(prior_tracks) > 0:
        #     refined_visible_tracks = prior_tracks
        #     tracks = visible_tracks + prior_tracks_cleaned
        #     tracks = sorted(tracks, key=lambda x: x.bbox.depth, reverse=False)
        #     foreground = np.full((rendered_height, rendered_width), 0)
        #     for idx, trk in enumerate(tracks):
        #         if trk.id in self.entity_id_map:
        #             sim_id = self.entity_id_map[trk.id]
        #             cv_mask = cv2.bitwise_and(np.array(mask_image.copy()), 255-foreground)
        #             cv_mask[cv_mask != sim_id] = 0
        #             cv_mask[cv_mask == sim_id] = 255
        #             xmin, ymin, w, h = cv2.boundingRect(cv_mask.astype(np.uint8))
        #             mask = cv_mask[ymin:ymin+h, xmin:xmin+w]
        #             visible_area = w*h+1
        #             screen_area = rendered_width*rendered_height+1
        #             if screen_area-visible_area == 0:
        #                 confidence = 1.0
        #             else:
        #                 confidence = visible_area/float(screen_area-visible_area)
        #             #TODO compute occlusion score as a ratio between visible 2d bbox and projected 2d bbox areas
        #             if confidence > occlusion_threshold:
        #
        #                 depth = real_depth_image[int(ymin+h/2.0)][int(xmin+w/2.0)]
        #                 xmin = int(xmin/rendering_ratio)
        #                 ymin = int(ymin/rendering_ratio)
        #                 w = int(w/rendering_ratio)
        #                 h = int(h/rendering_ratio)
        #
        #                 id = self.reverse_entity_id_map[sim_id]
        #                 scene_node = self.get_entity(id)
        #
        #                 det = Detection(int(xmin), int(ymin), int(xmin+w), int(ymin+h), id, confidence, depth=depth, mask=mask)
        #                 track = SceneNode(detection=det)
        #                 track.static = scene_node.static
        #                 track.id = id
        #                 track.mask = det.mask
        #                 track.shapes = scene_node.shapes
        #                 track.pose = scene_node.pose
        #                 track.label = scene_node.label
        #                 track.description = scene_node.description
        #                 refined_visible_tracks.append(track)
        #         track_full_mask = np.full((rendered_height, rendered_width), 0)
        #         xmin = int(trk.bbox.xmin*rendering_ratio)
        #         ymin = int(trk.bbox.ymin*rendering_ratio)
        #         w = int(trk.bbox.width()*rendering_ratio)
        #         h = int(trk.bbox.height()*rendering_ratio)
        #         try:
        #             track_mask = cv2.resize(trk.mask, (w, h))
        #             track_full_mask[ymin:ymin+h, xmin:xmin+w] = (track_mask)
        #         except Exception:
        #             pass
        #         foreground = cv2.bitwise_or(foreground, track_full_mask)
            # visible_tracks = refined_visible_tracks
        return rgb_image, real_depth_image, mask_image_resized, visible_tracks

    def update_entity_pose(self, id, pose):
        if id not in self.entity_id_map:
            raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
        base_link_sim_id = self.entity_id_map[id]
        t = pose.position().to_array().flatten()
        q = pose.quaternion()
        t_current, q_current = p.getBasePositionAndOrientation(base_link_sim_id)
        update_position = not np.allclose(np.array(t_current), t, atol=self.position_tolerance)
        update_orientation = not np.allclose(np.array(q_current), q, atol=self.position_tolerance)
        if update_position is True or update_orientation is True:
            p.resetBasePositionAndOrientation(base_link_sim_id, t, q, physicsClientId=self.client_simulator_id)

    def joint_states_callback(self, joint_states_msg):
        success, pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.base_frame_id)
        if success is True:
            if self.robot_loaded is False:
                try:
                    self.load_urdf("myself", self.robot_urdf_file_path, pose)
                    self.robot_loaded = True
                except Exception as e:
                    rospy.logwarn("[simulation] Exception occured: {}".format(e))
            try:
                self.update_entity_pose("myself", pose)
            except Exception as e:
                rospy.logwarn("[simulation] Exception occured: {}".format(e))
        if self.robot_loaded is True:
            joint_indices = []
            target_positions = []
            base_link_sim_id = self.entity_id_map["myself"]
            for joint_state_index, joint_name in enumerate(joint_states_msg.name):
                joint_sim_index = self.joint_id_map[base_link_sim_id][joint_name]
                info = p.getJointInfo(base_link_sim_id, joint_sim_index, physicsClientId=self.client_simulator_id)
                joint_name_sim = info[1]
                assert(joint_name == joint_name_sim)
                joint_position = joint_states_msg.position[joint_state_index]
                joint_indices.append(joint_sim_index)
                target_positions.append(joint_position)

            self.robot_joints_command = target_positions
            self.robot_joints_command_indices = joint_indices
