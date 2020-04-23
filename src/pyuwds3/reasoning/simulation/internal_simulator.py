import rospy
import cv2
import pybullet as p
import numpy as np
import copy
from sensor_msgs.msg import JointState
from ...types.scene_node import SceneNode, SceneNodeType
from tf.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix, translation_from_matrix
from ...utils.tf_bridge import TfBridge
from ...types.vector.vector6d import Vector6D
from ...types.shape.box import Box
from ...types.shape.cylinder import Cylinder
from ...types.shape.sphere import Sphere
from ...types.shape.mesh import Mesh
from ...types.detection import Detection
import yaml

INF = 1e+7

DEFAULT_DENSITY = 998.57


class InternalSimulationMode(object):
    SIMULATING = 0
    MONITORING = 1


class InternalSimulator(object):
    def __init__(self,
                 use_gui,
                 cad_models_additional_search_path,
                 static_entities_config_filename,
                 robot_urdf_file_path,
                 global_frame_id,
                 base_frame_id,
                 position_tolerance=0.05,
                 load_robot=True,
                 end_effectors=[]):
        """
        """
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

        self.robot_acting = False

        self.mode = InternalSimulationMode.MONITORING

        self.end_effectors = end_effectors

        if not p.isNumpyEnabled():
            rospy.logwarn("Numpy is not enabled, rendering can be slow")

        self.use_gui = use_gui
        if self.use_gui is True:
            self.client_simulator_id = p.connect(p.GUI)
        else:
            self.client_simulator_id = p.connect(p.DIRECT)

        if cad_models_additional_search_path != "":
            p.setAdditionalSearchPath(cad_models_additional_search_path)

        self.static_nodes = []

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
        p.setRealTimeSimulation(1)

        self.robot_loaded = False
        if load_robot is True:
            self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size=1)

    def load_urdf(self,
                  id,
                  filename,
                  start_pose,
                  static=False,
                  label="thing",
                  description=""):
        """ Load an URDF file in the simulator
        """
        try:
            use_fixed_base = 1 if static is True else 0
            flags = p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES or p.URDF_MERGE_FIXED_LINKS
            base_link_sim_id = p.loadURDF(filename, start_pose.position().to_array(), start_pose.quaternion(), useFixedBase=use_fixed_base, flags=flags)
            self.entity_id_map[id] = base_link_sim_id
            # Create a joint map to ease exploration
            self.reverse_entity_id_map[base_link_sim_id] = id
            self.joint_id_map[base_link_sim_id] = {}
            self.reverse_joint_id_map[base_link_sim_id] = {}
            for i in range(0, p.getNumJoints(base_link_sim_id)):
                info = p.getJointInfo(base_link_sim_id, i)
                self.joint_id_map[base_link_sim_id][info[1]] = info[0]
                self.reverse_joint_id_map[base_link_sim_id][info[0]] = info[1]
                p.changeDynamics(base_link_sim_id, info[0], frictionAnchor=1)
            # If file successfully loaded
            if base_link_sim_id < 0:
                raise RuntimeError("Invalid URDF")
            scene_node = SceneNode(pose=start_pose, is_static=True)
            scene_node.id = id
            scene_node.label = label
            scene_node.description = description
            if label == "myself":
                scene_node.type = SceneNodeType.MYSELF
                scene_node.label = "robot"
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

    def load_node(self, scene_node):
        """ Load a scene node in the simulator
        """
        if scene_node.is_located():
            base_pose = scene_node.pose
            if scene_node.has_shape():
                visual_shape_ids = []
                collision_shape_ids = []
                shape_masses = []

                for shape in scene_node.shapes:
                    shape_pose = shape.pose
                    t = shape_pose.position().to_array()
                    q = shape_pose.quaternion()
                    if shape.is_cylinder():
                        shape_type = p.GEOM_CYLINDER
                        radius = shape.radius()
                        height = shape.height()
                        collision_shape_id = p.createCollisionShape(shape_type, radius=radius,
                                                                    height=height,
                                                                    collisionFramePosition=t,
                                                                    collisionFrameOrientation=q)
                        visual_shape_id = p.createVisualShape(shape_type, radius=radius,
                                                              length=height,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color)
                        if visual_shape_id >= 0 and collision_shape_id >= 0:
                            mass = DEFAULT_DENSITY * shape.volume()
                            shape_masses.append(mass)
                            collision_shape_ids.append(collision_shape_id)
                            visual_shape_ids.append(visual_shape_id)

                    elif shape.is_sphere():
                        shape_type = p.GEOM_SPHERE
                        radius = shape.radius()
                        collision_shape_id = p.createCollisionShape(shape_type, radius=radius,
                                                                    collisionFramePosition=t,
                                                                    collisionFrameOrientation=q)
                        visual_shape_id = p.createVisualShape(shape_type, radius=radius,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color)
                        if visual_shape_id >= 0 and collision_shape_id >= 0:
                            mass = DEFAULT_DENSITY * shape.volume()
                            shape_masses.append(mass)
                            collision_shape_ids.append(collision_shape_id)
                            visual_shape_ids.append(visual_shape_id)

                    elif shape.is_box():
                        shape_type = p.GEOM_BOX
                        dim = [shape.x/2.0, shape.y/2.0, shape.z/2.0]
                        collision_shape_id = p.createCollisionShape(shape_type, halfExtents=dim,
                                                                    collisionFramePosition=t,
                                                                    collisionFrameOrientation=q)
                        visual_shape_id = p.createVisualShape(shape_type, halfExtents=dim,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color)
                        if visual_shape_id >= 0 and collision_shape_id >= 0:
                            mass = DEFAULT_DENSITY * shape.volume()
                            shape_masses.append(mass)
                            collision_shape_ids.append(collision_shape_id)
                            visual_shape_ids.append(visual_shape_id)

                    elif shape.is_mesh():
                        shape_type = p.GEOM_MESH
                        mesh_resource = shape.mesh_resource
                        scale = [shape.scale_x, shape.scale_y, shape.scale_z]
                        rospy.logwarn("[simulation] Collision shape not supported for Mesh primitives")
                        visual_shape_id = p.createVisualShape(shape_type,
                                                              meshScale=scale,
                                                              fileName=mesh_resource,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color)
                        if visual_shape_id >= 0:
                            mass = 0.0
                            shape_masses.append(mass)
                            visual_shape_ids.append(visual_shape_id)
                    else:
                        pass
                assert len(shape_masses) == max(len(collision_shape_ids), len(visual_shape_ids))
                if len(scene_node.shapes) == 1 and len(shape_masses) > 0:
                    t = base_pose.position().to_array()
                    q = base_pose.quaternion()
                    c_id = collision_shape_ids[0] if len(collision_shape_ids) == 1 else -1
                    sim_id = p.createMultiBody(baseMass=shape_masses[0],
                                               basePosition=t,
                                               baseOrientation=q,
                                               baseCollisionShapeIndex=c_id,
                                               baseVisualShapeIndex=visual_shape_ids[0])
                    p.changeDynamics(sim_id, -1, frictionAnchor=1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
                    self.entity_id_map[scene_node.id] = sim_id
                    self.entity_map[scene_node.id] = copy.deepcopy(scene_node)
                    return True
                else:
                    print scene_node.id
                    print len(shape_masses)
                    print len(scene_node.shapes)
                    rospy.logwarn("[simulation] Multibody shape not supported, consider using load URDF")
        return False

    def get_myself(self):
        """ Fetch the robot scene node
        """
        node = self.get_entity("myself")
        return node

    def get_static_entities(self):
        """ Fetch the static scene nodes given by the config file at start
        """
        return self.static_nodes

    def get_entity(self, id):
        """ Fetch an entity in the simulator and perform a lazzy update of the corresponding scene node
        """
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

    def get_camera_view(self, camera_pose, camera, target_position=None, occlusion_threshold=0.01, rendering_ratio=1.0):
        """ Render the rgb, depth and mask images from any point or view and compute the corresponding visible nodes
        """
        visible_nodes = []
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
                    visible_nodes.append(track)

        return rgb_image, real_depth_image, mask_image_resized, visible_nodes

    def test_aabb_collision(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """ Return True if the aabb is in contact with an other object
        """
        contacts = p.getOverlappingObjects([xmin, ymin, zmin], [xmax, ymax, zmax])
        if contacts is not None:
            if len(contacts) > 0:
                return True
        return False
    #
    # def compute_raycast_batch(self, poses):
    #     """
    #     """
    #     pass

    # def is_reachable(self, position):
    #     """ Return True if the position if reachable from the robot
    #     """
    #     base_link_sim_id = self.entity_id_map["myself"]
    #     #self.joint_id_map[base_link_sim_id][]
    #     p.calculateInverseKinematics(base_link_sim_id)
    #     pass
    #
    # def get_aabb(self, id):
    #     """
    #     """
    #     sim_id = self.entity_id_map[id]
    #     aabb_min, aabb_max = p.getAABB(sim_id)
    #     xmin, ymin, zmin = aabb_min
    #     xmax, ymax, zmax = aabb_max
    #     return xmin, ymin, zmin, xmax, ymax, zmax

    def update_constraint(self, id, pose):
        """ Update a constraint
        """
        base_link_sim_id = self.entity_id_map[id]
        if id not in self.constraint_id_map:
            constraint_id = p.createConstraint(base_link_sim_id, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 1])
            self.constraint_id_map[id] = constraint_id
        else:
            constraint_id = self.constraint_id_map[id]
        t = pose.position().to_array()
        q = pose.quaternion()
        p.changeConstraint(constraint_id, jointChildPivot=t, jointChildFrameOrientation=q, maxForce=INF)

    def remove_constraint(self, id):
        """ Remove a constraint
        """
        if id in self.constraint_id:
            p.removeConstraint(self.constraint_id_map[id])

    # def update_entity_pose(self, id, pose):
    #     if id not in self.entity_id_map:
    #         raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
    #     base_link_sim_id = self.entity_id_map[id]
    #     t = pose.position().to_array().flatten()
    #     q = pose.quaternion()
    #     p.resetBasePositionAndOrientation(base_link_sim_id, t, q, physicsClientId=self.client_simulator_id)

    def is_entity_loaded(self, id):
        """ Returns True if the entity is loaded
        """
        return id in self.entity_id_map

    def is_robot_loaded(self):
        """ Returns True if the robot is robot_loaded
        """
        return self.robot_loaded

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
                self.update_constraint("myself", pose)
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
                state = p.getJointState(base_link_sim_id, joint_sim_index)
                current_position = state[0]
                if abs(joint_position - current_position) >= self.position_tolerance:
                    joint_indices.append(joint_sim_index)
                    target_positions.append(joint_position)
                if len(target_positions) > 0:
                    p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
                    p.setJointMotorControlArray(base_link_sim_id,
                                                joint_indices,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPositions=target_positions,
                                                forces=np.full(len(target_positions), 50000))
                    #p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
                else:
                    p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
