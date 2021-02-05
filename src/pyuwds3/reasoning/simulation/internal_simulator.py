import rospy
import cv2
import pybullet as p
import os.path
import numpy as np
from sensor_msgs.msg import JointState
from ...types.scene_node import SceneNode, SceneNodeType

from pyuwds3.utils.marker_publisher import MarkerPublisher
from tf.transformations import translation_matrix, quaternion_matrix
from tf.transformations import quaternion_from_matrix, translation_from_matrix
from ...utils.tf_bridge import TfBridge
from ...types.vector.vector6d import Vector6D
from ...types.shape.box import Box
from ...types.shape.cylinder import Cylinder
from ...types.shape.sphere import Sphere
from ...types.shape.mesh import Mesh
from ...types.detection import Detection
import yaml
import inspect

INF = 1e+7
DEFAULT_DENSITY = 998.57
import rospkg
rospack = rospkg.RosPack()
class InternalSimulator(object):
    """
    """
    def __init__(self,
                 use_gui,
                 simulation_config_filename,
                 cad_models_additional_search_path,
                 static_entities_config_filename,
                 robot_urdf_file_path,
                 global_frame_id,
                 base_frame_id,
                 load_robot=True,
                 update_robot_at_each_step=True):
        """ Internal simulator constructor
        """
        # if simulation_config_filename != "":
        #     with open(simulation_config_filename, 'r') as config:
        #         self.simulator_config = yaml.safe_load(config)
        # else:
        #     self.load_robot = False

        self.tf_bridge = TfBridge()
        self.joint_map_reset = {}
        self.nodes_map = {}

        self.my_id = None
        self.marker_publisher = MarkerPublisher("adream",lifetime=999999)
        self.update_robot_at_each_step=update_robot_at_each_step

        self.entity_id_map = {}
        self.reverse_entity_id_map = {}

        self.joint_id_map = {}
        self.reverse_joint_id_map = {}

        self.constraint_id_map = {}

        self.markers_id_map = {}

        self.robot_joints_command = []
        self.robot_joints_command_indices = []
        # self.position_tolerance = self.simulator_config["base_config"]["position_tolerance"]

        # if "controller_config" in self.simulator_config:
        #     self.use_controller = True
        # else:
        #     self.use_controller = False

        self.global_frame_id = global_frame_id
        self.base_frame_id = base_frame_id

        self.robot_urdf_file_path = robot_urdf_file_path

        self.robot_moving = False

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
        self.not_static_nodes = []

        if static_entities_config_filename != "":
            with open(static_entities_config_filename, 'r') as stream:
                static_entities = yaml.safe_load(stream)
                for entity in static_entities:
                    start_pose = Vector6D(x=entity["position"]["x"],
                                          y=entity["position"]["y"],
                                          z=entity["position"]["z"],
                                          rx=entity["orientation"]["x"],
                                          ry=entity["orientation"]["x"],
                                          rz=entity["orientation"]["z"])

                    success, static_node = self.load_urdf(entity["file"],
                                                          start_pose,
                                                          id=entity["id"],
                                                          label=entity["label"],
                                                          description=entity["description"],
                                                          static=True)
                    header=rospy.Header()
                    header.frame_id="/map"
                    header.stamp=rospy.Time()
                    self.marker_publisher.publish([static_node],header)
                    if not success:
                        rospy.logwarn("[simulator] Unable to load {} node, skip it.".format(entity["id"]))

        p.setGravity(0, 0, 0, physicsClientId=self.client_simulator_id)
        p.setRealTimeSimulation(0, physicsClientId=self.client_simulator_id)

        self.robot_loaded = False
        self.joint_states_last_update = None
        self.load_robot = load_robot
        if load_robot is True:
            self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size=1)

        # if self.use_controller is True:
        #     pass
        #     # TODO add controller to play arm traj, only PD is available in bullet, it is sufficient ?

    def load_urdf(self,
                  filename,
                  start_pose,
                  static=False,
                  id="",
                  label="thing",
                  description="unknown",is_urdf=True,color=None):
        """ Load an URDF file in the simulator
        """
        try:
            scene_node = SceneNode(pose=start_pose, is_static=static)
            if id != "":
                scene_node.id = id
            scene_node.label = label
            scene_node.description = description
            if label == "myself":
                scene_node.type = SceneNodeType.MYSELF
                scene_node.label = "robot"
            use_fixed_base = 1 if static is True else 0
            flags = p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES or p.URDF_MERGE_FIXED_LINKS
            if is_urdf:
                # print filename
                base_link_sim_id = p.loadURDF(filename, start_pose.position().to_array(), start_pose.quaternion(), useFixedBase=use_fixed_base, physicsClientId=self.client_simulator_id)
            else:

                use_fixed_base = 0 if static is True else 1

                if "file://" in filename:
                    filename = filename.split("file://")[-1]
                if "package://" in filename:

                    a=filename.split("package://")[-1]
                    k=a.split('/')
                    path=rospack.get_path(k[0])
                    for i in k[1:]:
                        path+='/'+i
                    filename=path
                collision_shape_id = p.createCollisionShape(p.GEOM_MESH,fileName=filename,flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=self.client_simulator_id)
                if not color is None:
                    visual_shape_id = p.createVisualShape(p.GEOM_MESH,fileName=filename,rgbaColor=color, physicsClientId=self.client_simulator_id)
                else:
                    visual_shape_id = p.createVisualShape(p.GEOM_MESH,fileName=filename, physicsClientId=self.client_simulator_id)
                base_link_sim_id = p.createMultiBody(
                                            use_fixed_base,
                                            collision_shape_id,visual_shape_id,
                                            start_pose.position().to_array(),
                                            start_pose.quaternion(),flags = flags, physicsClientId=self.client_simulator_id)
                # base_link_sim_id = p.createMultiBody(
                #                             use_fixed_base,
                #                             collision_shape_id,visual_shape_id,
                #                             [0,0,0],[1,0,0,0],flags = flags)

            self.entity_id_map[scene_node.id] = base_link_sim_id
            # Create a joint map to ease exploration
            self.reverse_entity_id_map[base_link_sim_id] = scene_node.id
            self.joint_id_map[base_link_sim_id] = {}
            self.reverse_joint_id_map[base_link_sim_id] = {}
            for i in range(0, p.getNumJoints(base_link_sim_id, physicsClientId=self.client_simulator_id)):
                info = p.getJointInfo(base_link_sim_id, i, physicsClientId=self.client_simulator_id)
                self.joint_id_map[base_link_sim_id][info[1]] = info[0]
                self.reverse_joint_id_map[base_link_sim_id][info[0]] = info[1]
                p.changeDynamics(base_link_sim_id, info[0], frictionAnchor=1)
            # If file successfully loaded
            if base_link_sim_id < 0:
                raise RuntimeError("Invalid URDF provided: {}".format(filename))

            sim_id = self.entity_id_map[scene_node.id]
            visual_shapes = p.getVisualShapeData(sim_id, physicsClientId=self.client_simulator_id)
            for shape in visual_shapes:
                link_id = shape[1]
                type = shape[2]
                dimensions = shape[3]
                mesh_file_path = shape[4]
                position = shape[5]
                orientation = shape[6]
                rgba_color = shape[7]

                if link_id != -1:
                    link_state = p.getLinkState(sim_id, link_id, physicsClientId=self.client_simulator_id)
                    t_link = link_state[0]
                    q_link = link_state[1]
                    t_inertial = link_state[2]
                    q_inertial = link_state[3]

                    tf_world_link = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))
                    tf_inertial_link = np.dot(translation_matrix(t_inertial), quaternion_matrix(q_inertial))
                    world_transform = np.dot(tf_world_link, np.linalg.inv(tf_inertial_link))
                else:
                    t_link, q_link = p.getBasePositionAndOrientation(sim_id, physicsClientId=self.client_simulator_id)
                    world_transform = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))

                if type == p.GEOM_SPHERE:
                    primitive_shape = Sphere(d=dimensions[0]*2.0)
                elif type == p.GEOM_BOX:
                    primitive_shape = Box(dim_x=dimensions[0], dim_y=dimensions[1], dim_z=dimensions[2])
                elif type == p.GEOM_CYLINDER:
                    primitive_shape = Cylinder(w=dimensions[1]*2.0, h=dimensions[0])
                elif type == p.GEOM_PLANE:
                    primitive_shape = Box(dim_x=dimensions[0], dim_y=dimensions[1], dim_z=0.0001)
                elif type == p.GEOM_MESH:
                    primitive_shape = Mesh(mesh_resource="file://"+mesh_file_path,
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

                primitive_shape.color[0] = rgba_color[0]
                primitive_shape.color[1] = rgba_color[1]
                primitive_shape.color[2] = rgba_color[2]
                primitive_shape.color[3] = rgba_color[3]

                scene_node.shapes.append(primitive_shape)
            self.nodes_map[scene_node.id] = scene_node
            if static is True:
                self.static_nodes.append(scene_node)
            else:
                self.not_static_nodes.append(scene_node)
            return True, scene_node
            rospy.loginfo("[simulation] '{}' File successfully loaded".format(filename))
        except Exception as e:
            rospy.logwarn("[simulation] Error loading URDF '{}': {}".format(filename, e))
            return False, None

    def test_raycast(self, start_position, end_position):
        """
        """
        ray_start = start_position.to_array().flatten()
        ray_end = end_position.to_array().flatten()
        result = p.rayTest(ray_start, ray_end, physicsClientId=self.client_simulator_id)
        if result is not None:
            distance = ray_start[2] * result[0][2]
            sim_id = result[0][0]
            hit_object_id = self.reverse_entity_id_map[sim_id]
            hit_object = self.nodes_map[hit_object_id]
            return True, distance, hit_object
        return False, None, None

    def get_aabb(self, scene_node):
        if scene_node.id in self.nodes_map:
            sim_id = self.entity_id_map[scene_node.id]
            aabb = p.getAABB(sim_id, physicsClientId=self.client_simulator_id)
            return True, aabb
        return False, None

    def get_overlapping_nodes(self, scene_node, margin=0.0):
        if scene_node.id in self.nodes_map:
            nodes_overlapping = []
            sim_id = self.entity_id_map[scene_node.id]
            aabb = p.getAABB(sim_id, physicsClientId=self.client_simulator_id)
            sim_ids_overlapping = p.getOverlappingObjects(aabb[0], aabb[1], physicsClientId=self.client_simulator_id)
            for sim_id in sim_ids_overlapping:
                object_id = self.reverse_entity_id_map[sim_id]
                nodes_overlapping.append(self.nodes_map[object_id])
            return True, nodes_overlapping
        return False, []

    def load_node(self, scene_node, static=False):
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
                                                                    collisionFrameOrientation=q, physicsClientId=self.client_simulator_id)
                        visual_shape_id = p.createVisualShape(shape_type, radius=radius,
                                                              length=height,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color, physicsClientId=self.client_simulator_id)
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
                                                                    collisionFrameOrientation=q, physicsClientId=self.client_simulator_id)
                        visual_shape_id = p.createVisualShape(shape_type, radius=radius,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color, physicsClientId=self.client_simulator_id)
                        if visual_shape_id >= 0 and collision_shape_id >= 0:
                            mass = DEFAULT_DENSITY * shape.volume()
                            shape_masses.append(mass)
                            collision_shape_ids.append(collision_shape_id)
                            visual_shape_ids.append(visual_shape_id)
                    # elif shape.is_plane():
                    #     shape_type = p.GEOM_PLANE
                    #     dim = [shape.x/2.0, shape.y/2.0, shape.z/2.0]
                    #     collision_shape_id = p.createCollisionShape(shape_type, plane_normal=dim,
                    #                                                 collisionFramePosition=t,
                    #                                                 collisionFrameOrientation=q
                    #                                                 )
                    #     visual_shape_id = p.createVisualShape(shape_type, plane_normal=dim,
                    #                                           visualFramePosition=t,
                    #                                           visualFrameOrientation=q,
                    #                                           rgbaColor=shape.color)
                    #     if visual_shape_id >= 0 and collision_shape_id >= 0:
                    #         mass = DEFAULT_DENSITY * shape.volume()
                    #         shape_masses.append(mass)
                    #         collision_shape_ids.append(collision_shape_id)
                    #         visual_shape_ids.append(visual_shape_id)
                    elif shape.is_box():

                        shape_type = p.GEOM_BOX
                        dim = [shape.x/2.0, shape.y/2.0, shape.z/2.0]
                        collision_shape_id = p.createCollisionShape(shape_type, halfExtents=dim,
                                                                    collisionFramePosition=t,
                                                                    collisionFrameOrientation=q, physicsClientId=self.client_simulator_id)
                        visual_shape_id = p.createVisualShape(shape_type, halfExtents=dim,
                                                              visualFramePosition=t,
                                                              visualFrameOrientation=q,
                                                              rgbaColor=shape.color, physicsClientId=self.client_simulator_id)
                        if visual_shape_id >= 0 and collision_shape_id >= 0:
                            mass = DEFAULT_DENSITY * shape.volume()
                            shape_masses.append(mass)
                            collision_shape_ids.append(collision_shape_id)
                            visual_shape_ids.append(visual_shape_id)

                    elif shape.is_mesh():
                        shape_type = p.GEOM_MESH
                        mesh_resource = shape.mesh_resource
                        if "package://" in mesh_resource:
                            a=mesh_resource.split("package://")[-1]
                            k=a.split('/')
                            path=rospack.get_path(k[0])
                            for i in k[1:]:
                                path+='/'+i
                            mesh_resource=path
                        mesh_resource_u = mesh_resource.replace("obj", "urdf")
                        mesh_resource_u = mesh_resource_u.replace("dae","urdf")
                        mesh_resource_u = mesh_resource_u.replace("stl","urdf")
                        mesh_resource_u = mesh_resource_u.replace("file://", "")
                        # print mesh_resource
                        is_urdf = os.path.isfile(mesh_resource_u)
                        if is_urdf:
                            mesh_resource = mesh_resource_u

                        success, node = self.load_urdf(mesh_resource,
                                                       base_pose,
                                                       id=scene_node.id,
                                                       label=scene_node.label,
                                                       description=scene_node.description,
                                                       static=static,is_urdf=is_urdf,color=shape.color)
                        if success is True:
                            return True
                        else:
                            return False
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
                                               baseVisualShapeIndex=visual_shape_ids[0], physicsClientId=self.client_simulator_id)

                    p.changeDynamics(sim_id, -1, frictionAnchor=1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING, physicsClientId=self.client_simulator_id)
                    self.entity_id_map[scene_node.id] = sim_id
                    self.reverse_entity_id_map[sim_id] = scene_node.id
                    self.nodes_map[scene_node.id] = scene_node
                    if static is True:
                        self.static_nodes.append(scene_node)
                    else:
                        self.not_static_nodes.append(scene_node)
                    if static is True:
                        self.update_constraint(scene_node.id, scene_node.pose)
                    return True
                else:
                    rospy.logwarn("[simulation] Multibody shape not supported at the moment, consider using load URDF")
        return False

    def get_myself(self):
        """ Fetch the robot scene node
        """
        node = self.get_entity(self.my_id)
        return node

    def get_static_entities(self):
        """ Fetch the static scene nodes given by the config file at start
        """
        return self.static_nodes

    def get_not_static_entities(self):
        """ Fetch the not static scene nodes
        """
        return [self.get_entity(o.id) for o in self.not_static_nodes]

    def get_entity(self, id):
        """ Fetch an entity in the simulator and perform a lazzy update of the corresponding scene node
        """
        if id not in self.nodes_map:
            raise ValueError("Invalid id provided : '{}'".format(id))
        scene_node = self.nodes_map[id]
        sim_id = self.entity_id_map[id]
        visual_shapes = p.getVisualShapeData(sim_id, physicsClientId=self.client_simulator_id)
        for idx, shape in enumerate(visual_shapes):
            primitive_shape = scene_node.shapes[idx]
            link_id = shape[1]
            position = shape[5]
            orientation = shape[6]

            if link_id != -1:
                link_state = p.getLinkState(sim_id, link_id, physicsClientId=self.client_simulator_id)
                t_link = link_state[0]
                q_link = link_state[1]
                t_inertial = link_state[2]
                q_inertial = link_state[3]

                tf_world_link = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))
                tf_inertial_link = np.dot(translation_matrix(t_inertial), quaternion_matrix(q_inertial))
                world_transform = np.dot(tf_world_link, np.linalg.inv(tf_inertial_link))

            else:
                t_link, q_link = p.getBasePositionAndOrientation(sim_id, physicsClientId=self.client_simulator_id)
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
        return self.nodes_map[id]

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
        view_matrix = p.computeViewMatrix(camera_pose.position().to_array(), target, [0, 0, 1], physicsClientId=self.client_simulator_id)

        width = camera.width
        height = camera.height

        rendered_width = int(width*rendering_ratio)
        rendered_height = int(height*rendering_ratio)

        projection_matrix = p.computeProjectionMatrixFOV(camera.fov(),
                                                         float(rendered_width)/rendered_height,
                                                         camera.clipnear,
                                                         camera.clipfar, physicsClientId=self.client_simulator_id)

        if self.use_gui is True:
            camera_image = p.getCameraImage(rendered_width,
                                            rendered_height,
                                            viewMatrix=view_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                            projectionMatrix=projection_matrix, physicsClientId=self.client_simulator_id)
        else:
            camera_image = p.getCameraImage(rendered_width,
                                            rendered_height,
                                            viewMatrix=view_matrix,
                                            renderer=p.ER_TINY_RENDERER,
                                            projectionMatrix=projection_matrix, physicsClientId=self.client_simulator_id)

        rgb_image = np.array(camera_image[2])[:, :, :3]
        rgb_image_resized = cv2.resize(rgb_image, (width, height))
        depth_image = np.array(camera_image[3], np.float32).reshape((rendered_height, rendered_width))

        far = camera.clipfar
        near = camera.clipnear
        real_depth_image = far * near / (far - (far - near) * depth_image)

        mask_image = camera_image[4]
        mask_image_resized = cv2.resize(np.array(camera_image[4]).copy().astype("uint8"), (width, height))
        unique, counts = np.unique(np.array(mask_image).flatten(), return_counts=True)

        # bgr_image_resized = cv2.cvtColor(rgb_image_resized, cv2.COLOR_RGB2BGR)

        # cv2.imwrite("/home/ysallami/Documents/presentation_hri_uwds3/img/perspective_input_rgb_image.png", bgr_image_resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

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

                    # xmin, ymin, w, h = cv2.boundingRect(cv_mask.astype(np.uint8))
                    # cv_mask = cv2.cvtColor(cv_mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    # cv_mask_resized = cv2.resize(cv_mask, (width, height))

        real_depth_image_resized = cv2.resize(real_depth_image, (width, height))
        # normalized_depth_image = cv2.normalize(depth_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # normalized_depth_image_resized = cv2.resize(normalized_depth_image, (width, height))

        # mask_image_normalized = cv2.normalize(mask_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # mask_image_viz = cv2.applyColorMap(mask_image_normalized.astype("uint8"), cv2.COLORMAP_HSV)
        # mask_image_viz_resized = cv2.resize(mask_image_viz, (width, height))


        return rgb_image_resized, real_depth_image_resized, mask_image_resized, visible_nodes

    def test_overlap(self, xmin, ymin, zmin, xmax, ymax, zmax):
        """ Return True if the aabb is in contact with an other object
        """
        contacts = p.getOverlappingObjects([xmin, ymin, zmin], [xmax, ymax, zmax], physicsClientId=self.client_simulator_id)
        if contacts is not None:
            if len(contacts) > 0:
                return True
        return False

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
        p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
        p.changeConstraint(constraint_id, jointChildPivot=t, jointChildFrameOrientation=q, maxForce=INF, physicsClientId=self.client_simulator_id)

    def remove_constraint(self, id):
        """ Remove a constraint
        """
        if id in self.constraint_id_map:
            p.removeConstraint(self.constraint_id_map[id], physicsClientId=self.client_simulator_id)
            del self.constraint_id_map[id]
            base_link_sim_id = self.entity_id_map[id]
            p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING, physicsClientId=self.client_simulator_id)

    def reset_entity_pose(self, id, pose):
        """ Reset the pose and the simulation for the given entity
        """
        if id not in self.entity_id_map:
            raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
        base_link_sim_id = self.entity_id_map[id]
        t = pose.position().to_array().flatten()
        q = pose.quaternion()
        p.resetBasePositionAndOrientation(base_link_sim_id, t, q, physicsClientId=self.client_simulator_id)

    def update_node(self, node):
        self.nodes_map[node.id] = node

    def is_entity_loaded(self, id):
        """ Returns True if the entity is loaded
        """
        return id in self.entity_id_map

    def is_robot_loaded(self):
        """ Returns True if the robot is robot_loaded
        """
        return self.robot_loaded

    def is_robot_moving(self):
        """ Returns True if the robot is moving
        """
        return self.robot_moving

    def step_simulation(self):
        p.stepSimulation( physicsClientId=self.client_simulator_id)

    # def joint_states_callback(self, joint_states_msg):
    #     """
    #     """
    #
    #     success, pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.base_frame_id)
    #
    #     if success is True:
    #         if self.robot_loaded is False:
    #             try:
    #                 success, node = self.load_urdf(self.robot_urdf_file_path,
    #                                             pose,
    #                                             label="myself",
    #                                             description="myself")
    #                 rospy.loginfo("[simulation] Robot loaded")
    #                 self.my_id = node.id
    #                 self.robot_loaded = True
    #             except Exception as e:
    #                 rospy.logwarn("[simulation] Exception occured: {}".format(e))
    #         else:
    #             self.update_constraint(self.my_id, pose)
    #     if self.robot_loaded is True:
    #
    #         joint_indices = []
    #         target_positions = []
    #         base_link_sim_id = self.entity_id_map[self.my_id]
    #         for joint_state_index, joint_name in enumerate(joint_states_msg.name):
    #             joint_sim_index = self.joint_id_map[base_link_sim_id][joint_name]
    #             info = p.getJointInfo(base_link_sim_id, joint_sim_index, physicsClientId=self.client_simulator_id)
    #             joint_name_sim = info[1]
    #             assert(joint_name == joint_name_sim)
    #             joint_position = joint_states_msg.position[joint_state_index]
    #             state = p.getJointState(base_link_sim_id, joint_sim_index)
    #             current_position = state[0]
    #             if abs(joint_position - current_position) >= self.position_tolerance:
    #                 joint_indices.append(joint_sim_index)
    #                 target_positions.append(joint_position)
    #             if len(target_positions) > 0:
    #                 self.robot_moving = True
    #                 p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    #
    #                 p.setJointMotorControlArray(base_link_sim_id,
    #                                             joint_indices,
    #                                             controlMode=p.POSITION_CONTROL,
    #                                             targetPositions=target_positions,forces = [0]*len(joint_indices))
    #
    #             else:
    #                 self.robot_moving = False
    #                 p.changeDynamics(base_link_sim_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
    #

    def change_joint(self,main_id,joint_id,joint_position):
        base_link_sim_id = self.entity_id_map[main_id]
        print p.getNumJoints(base_link_sim_id, physicsClientId=self.client_simulator_id)
        p.resetJointState(base_link_sim_id,joint_id,0, physicsClientId=self.client_simulator_id)


    def joint_states_callback(self, joint_states_msg):
        """
        """
        # print "jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj"
        if not self.load_robot:
            return

        success, pose = self.tf_bridge.get_pose_from_tf(self.global_frame_id, self.base_frame_id)
        # if not success:
        #     print "hhhhhhhhhhhhhhhhhhhhppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp"
        if success is True:
            # print "oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
            if self.robot_loaded is False:
                try:
                    # curframe = inspect.currentframe()
                    # calframe = inspect.getouterframes(curframe, 2)
                    # print ("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
                    # print('caller name:', calframe[1][3])
                    self.robot_loaded = True
                    success, node = self.load_urdf(self.robot_urdf_file_path,
                                                pose,
                                                label="myself",
                                                description="myself")
                    if not success:
                        self.robot_loaded = False
                    rospy.loginfo("[simulation] Robot loaded")
                    self.my_id = node.id
                except Exception as e:
                    rospy.logwarn("[simulation, joint_states_callback] Exception occured: {}".format(e))
                    self.robot_loaded = False
            # else:
            #     self.update_constraint(self.my_id, pose)
        if self.robot_loaded is True:

            joint_indices = []
            target_positions = []
            base_link_sim_id = self.entity_id_map[self.my_id]
            for joint_state_index, joint_name in enumerate(joint_states_msg.name):
                joint_sim_index = self.joint_id_map[base_link_sim_id][joint_name]
                info = p.getJointInfo(base_link_sim_id, joint_sim_index, physicsClientId=self.client_simulator_id)
                joint_name_sim = info[1]
                # joint_index = info[0]
                assert(joint_name == joint_name_sim)
                joint_position = joint_states_msg.position[joint_state_index]
                state = p.getJointState(base_link_sim_id, joint_sim_index, physicsClientId=self.client_simulator_id)
                current_position = state[0]
                if not joint_sim_index in self.joint_map_reset:
                    self.joint_map_reset[joint_sim_index]=joint_position
                    p.resetJointState(base_link_sim_id,joint_sim_index,joint_position, physicsClientId=self.client_simulator_id)
                else:
                    if self.joint_map_reset[joint_sim_index]!=joint_position:
                        self.joint_map_reset[joint_sim_index]=joint_position
                        p.resetJointState(base_link_sim_id,joint_sim_index,joint_position, physicsClientId=self.client_simulator_id)

        self.load_robot = self.update_robot_at_each_step
