import rospy
import numpy as np
import cv2
import uuid
import uwds3_msgs
from tf.transformations import euler_matrix
import uwds3_msgs.msg
from .camera import Camera, HumanCamera
from .shape.cylinder import Cylinder
from .shape.sphere import Sphere
from .shape.mesh import Mesh
from .vector.vector6d_stable import Vector6DStable
from .bbox_stable import BoundingBoxStable
from .vector.vector6d import Vector6D
from ..reasoning.tracking.medianflow_tracker import MedianFlowTracker


class SceneNodeType(object):
    OBJECT = uwds3_msgs.msg.SceneNode.OBJECT
    MYSELF = uwds3_msgs.msg.SceneNode.MYSELF
    OTHERS = uwds3_msgs.msg.SceneNode.OTHERS


class SceneNodeState(object):
    TENTATIVE = uwds3_msgs.msg.SceneNode.TENTATIVE
    CONFIRMED = uwds3_msgs.msg.SceneNode.CONFIRMED
    LOST = uwds3_msgs.msg.SceneNode.LOST
    OCCLUDED = uwds3_msgs.msg.SceneNode.OCCLUDED
    DELETED = uwds3_msgs.msg.SceneNode.DELETED


class SceneNode(object):
    def __init__(self,
                 detection=None,
                 label="thing",
                 pose=None,
                 n_init=1,
                 max_lost=10,
                 max_age=30,
                 is_static=False,
                 vx=0., vy=0., vz=0.,
                 vrx=0., vry=0., vrz=0.,
                 ax=0., ay=0., az=0.,
                 arx=0., ary=0., arz=0.,
                 p_cov_c=0.85, m_cov_c=0.003,
                 p_cov_a=0.85, m_cov_a=1e-9,
                 p_cov_h=0.85, m_cov_h=1e-9,
                 p_cov_p=0.8, m_cov_p=0.01,
                 p_cov_r=0.06, m_cov_r=0.001,
                 time=None):
        """ Scene node constructor
        """
        self.id = str(uuid.uuid4()).replace("-", "")

        self.features = {}
        self.label = None
        self.bbox = None
        self.mask = None

        self.n_init = n_init
        self.max_lost = max_lost
        self.max_age = max_age

        self.static = is_static
        if self.static is True:
            self.state = SceneNodeState.CONFIRMED
        else:
            if self.n_init == 1:
                self.state = SceneNodeState.CONFIRMED
            else:
                self.state = SceneNodeState.TENTATIVE

        self.hits = 0
        self.age = 0

        if detection is None:
            self.label = label
        else:
            self.label = detection.label
            self.update_bbox(detection)

        if self.label == "myself":
            self.type = SceneNodeType.MYSELF
        elif self.label in ["person", "face", "hand"]:
            self.type = SceneNodeType.OTHERS
        else:
            self.type = SceneNodeType.OBJECT

        self.parent = ""
        self.type = type
        self.description = ""
        self.perceived = False

        self.p_cov_p = p_cov_p
        self.m_cov_p = m_cov_p
        self.p_cov_r = p_cov_r
        self.m_cov_r = m_cov_r

        self.p_cov_c = p_cov_c
        self.m_cov_c = m_cov_c
        self.p_cov_a = p_cov_a
        self.m_cov_a = m_cov_a
        self.p_cov_h = p_cov_h
        self.m_cov_h = m_cov_h

        if pose is not None:
            self.pose = Vector6DStable(x=pose.pos.x, y=pose.pos.y, z=pose.pos.z,
                                       rx=pose.rot.x, ry=pose.rot.y, rz=pose.rot.z,
                                       vx=vx, vy=vy, vz=vz,
                                       vrx=vrx, vry=vry, vrz=vrz,
                                       ax=ax, ay=ay, az=az,
                                       arx=arx, ary=ary, arz=arz,
                                       p_cov_p=self.p_cov_p, m_cov_p=self.m_cov_p,
                                       p_cov_r=self.p_cov_r, m_cov_r=self.m_cov_r,
                                       time=time)
        else:
            self.pose = None

        self.shapes = []

        self.tracker = None

        if self.label == "face":
            self.camera = HumanCamera()
        else:
            self.camera = None

        self.last_update = rospy.Time().now()
        self.expiration_duration = 1.0

    def update_bbox(self, detection, detected=True, time=None):
        """ Update the 2D bbox corresponding to the scene node
        """
        if self.bbox is None or self.state == SceneNodeState.LOST:
            self.bbox = BoundingBoxStable(detection.bbox.xmin,
                                          detection.bbox.ymin,
                                          detection.bbox.xmax,
                                          detection.bbox.ymax,
                                          depth=detection.bbox.depth,
                                          time=time)
        else:
            self.bbox.update(detection.bbox.xmin,
                             detection.bbox.ymin,
                             detection.bbox.xmax,
                             detection.bbox.ymax,
                             depth=detection.bbox.depth,
                             time=time)
        w = self.bbox.width()
        h = self.bbox.height()
        if detection.mask is not None:
            if detection.mask.shape != (h, w):
                self.mask = cv2.resize(detection.mask.astype("uint8"), (w, h))
            else:
                self.mask = detection.mask
        for name, features in detection.features.items():
            self.features[name] = features
        if detected is True:
            self.age = 0
            self.hits += 1
        if self.state == SceneNodeState.TENTATIVE and self.hits >= self.n_init:
            self.state = SceneNodeState.CONFIRMED
        if self.state == SceneNodeState.OCCLUDED:
            self.state = SceneNodeState.CONFIRMED
        if self.state == SceneNodeState.LOST:
            self.state = SceneNodeState.CONFIRMED

    def predict_bbox(self, time=None):
        """ Predict the 2D bbox location based on motion model (kalman)
        """
        self.bbox.predict(time=time)

    def update_pose(self, position, rotation=None, time=None):
        """ Update the 3D pose in global frame
        """
        if self.pose is None:
            if rotation is None:
                self.pose = Vector6DStable(x=position.x,
                                           y=position.y,
                                           z=position.z,
                                           p_cov_p=self.p_cov_p,
                                           m_cov_p=self.m_cov_p,
                                           p_cov_r=self.p_cov_r,
                                           m_cov_r=self.m_cov_r,
                                           time=time)
            else:
                self.pose = Vector6DStable(x=position.x,
                                           y=position.y,
                                           z=position.z,
                                           rx=rotation.x,
                                           ry=rotation.y,
                                           rz=rotation.z,
                                           p_cov_p=self.p_cov_p,
                                           m_cov_p=self.m_cov_p,
                                           p_cov_r=self.p_cov_r,
                                           m_cov_r=self.m_cov_r,
                                           time=time)
        else:
            self.pose.pos.update(position.x, position.y, position.z, time=time)
            if rotation is not None:
                self.pose.rot.update(rotation.x, rotation.y, rotation.z, time=time)

    def predict_pose(self, time=None):
        """ Predict the 3D pose in global frame based on motion model (kalman)"""
        self.pose.predict(time=time)

    def start_tracker(self):
        """ Initialize the medianflow 2D tracker
        """
        self.tracker = MedianFlowTracker(self)

    def stop_tracker(self):
        """ Stop the 2D tracker
        """
        self.traker = None

    def mark_missed(self):
        """ Mark missed
        """
        self.age += 1
        if self.state == SceneNodeState.TENTATIVE:
            if self.age > self.n_init:
                self.state = SceneNodeState.DELETED
        elif self.state == SceneNodeState.CONFIRMED:
            if self.age > self.max_lost:
                self.state = SceneNodeState.LOST
        elif self.state == SceneNodeState.OCCLUDED:
            self.state = SceneNodeState.LOST
        elif self.state == SceneNodeState.LOST:
            if self.age > self.max_age:
                self.state = SceneNodeState.DELETED

    def mark_occluded(self):
        """ Mark occluded
        """
        if self.state == SceneNodeState.LOST:
            self.state = SceneNodeState.OCCLUDED

    def is_static(self):
        """ Return True if is static
        """
        return self.static

    def is_perceived(self):
        """ Return True if is perceived by a camera (real of virtual)
        """
        return self.bbox is not None

    def is_confirmed(self):
        """ Return True if is confirmed
        """
        return self.state == SceneNodeState.CONFIRMED

    def is_lost(self):
        """ Return True if is lost
        """
        return self.state == SceneNodeState.LOST

    def is_occluded(self):
        """ Return True if is occluded
        """
        return self.state == SceneNodeState.OCCLUDED

    def to_delete(self):
        """ Return True is to delete
        """
        return self.state == SceneNodeState.DELETED

    def is_located(self):
        """ Returns True if is located in 3D space
        """
        return self.pose is not None

    def has_shape(self):
        """ Returns True if has at least one shape
        """
        return len(self.shapes) > 0

    def has_camera(self):
        """ Return true if has camera
        """
        return self.camera is not None

    def has_mask(self):
        """ Return True if has a mask
        """
        return self.mask is not None

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.id = msg.id
        self.label = msg.label
        self.parent = msg.parent
        self.type = msg.type
        self.state = msg.state
        self.description = msg.description

        if msg.is_perceived is True:
            self.bbox = BoundingBoxStable().from_msg(msg.bbox)
        else:
            self.bbox = None

        if msg.is_located is True:
            x = msg.pose_stamped.pose.pose.position.x
            y = msg.pose_stamped.pose.pose.position.y
            z = msg.pose_stamped.pose.pose.position.z
            qx = msg.pose_stamped.pose.pose.orientation.x
            qy = msg.pose_stamped.pose.pose.orientation.y
            qz = msg.pose_stamped.pose.pose.orientation.z
            qw = msg.pose_stamped.pose.pose.orientation.w
            vx = msg.twist_stamped.twist.twist.linear.x
            vy = msg.twist_stamped.twist.twist.linear.y
            vz = msg.twist_stamped.twist.twist.linear.z
            vrx = msg.twist_stamped.twist.twist.angular.x
            vry = msg.twist_stamped.twist.twist.angular.y
            vrz = msg.twist_stamped.twist.twist.angular.z
            ax = msg.accel_stamped.accel.accel.linear.x
            ay = msg.accel_stamped.accel.accel.linear.y
            az = msg.accel_stamped.accel.accel.linear.z
            arx = msg.accel_stamped.accel.accel.angular.x
            ary = msg.accel_stamped.accel.accel.angular.y
            arz = msg.accel_stamped.accel.accel.angular.z
            self.pose = Vector6DStable(x=x, y=y, z=z,
                                       vx=vx, vy=vy, vz=vz,
                                       vrx=vrx, vry=vry, vrz=vrz,
                                       ax=ax, ay=ay, az=az,
                                       arx=arx, ary=ary, arz=arz).from_quaternion(qx, qy, qz, qw)
        else:
            self.pose = None

        msg.features = []
        for features in msg.features:
            msg.features.append(features.from_msg(features))

        if msg.has_shape is True:
            for shape in msg.shapes:
                if shape.type == uwds3_msgs.msg.PrimitiveShape.CYLINDER:
                    self.shapes.append(Cylinder().from_msg(shape))
                if shape.type == uwds3_msgs.msg.PrimitiveShape.SPHERE:
                    self.shapes.append(Sphere().from_msg(shape))
                if shape.type == uwds3_msgs.msg.PrimitiveShape.MESH:
                    self.shapes.append(Mesh().from_msg(shape))

        if msg.has_camera is True:
            self.camera = Camera().from_msg(msg.camera.info,
                                            clipnear=msg.camera.clipnear,
                                            clipfar=msg.camera.clipfar)
        else:
            self.camera = None

        return self

    def to_msg(self, header):
        """ Convert to ROS message
        """
        msg = uwds3_msgs.msg.SceneNode()

        msg.state = self.state

        if self.is_perceived():
            msg.is_perceived = True
            msg.bbox = self.bbox.to_msg()
        else:
            msg.is_perceived = False

        msg.id = self.id
        msg.label = self.label
        msg.description = self.description
        if self.is_located():
            msg.is_located = True
            msg.pose_stamped.header = header
            position = self.pose.position()
            msg.pose_stamped.pose.pose.position.x = position.x
            msg.pose_stamped.pose.pose.position.y = position.y
            msg.pose_stamped.pose.pose.position.z = position.z
            q = self.pose.quaternion()
            msg.pose_stamped.pose.pose.orientation.x = q[0]
            msg.pose_stamped.pose.pose.orientation.y = q[1]
            msg.pose_stamped.pose.pose.orientation.z = q[2]
            msg.pose_stamped.pose.pose.orientation.w = q[3]
            msg.twist_stamped.header = header
            linear_velocity = self.pose.linear_velocity()
            msg.twist_stamped.twist.twist.linear.x = linear_velocity.x
            msg.twist_stamped.twist.twist.linear.y = linear_velocity.y
            msg.twist_stamped.twist.twist.linear.z = linear_velocity.z
            angular_velocity = self.pose.angular_velocity()
            msg.twist_stamped.twist.twist.angular.x = angular_velocity.x
            msg.twist_stamped.twist.twist.angular.y = angular_velocity.y
            msg.twist_stamped.twist.twist.angular.z = angular_velocity.z
            msg.accel_stamped.header = header
            linear_acceleration = self.pose.linear_acceleration()
            msg.accel_stamped.accel.accel.linear.x = linear_acceleration.x
            msg.accel_stamped.accel.accel.linear.y = linear_acceleration.y
            msg.accel_stamped.accel.accel.linear.z = linear_acceleration.z
            angular_acceleration = self.pose.angular_acceleration()
            msg.accel_stamped.accel.accel.angular.x = angular_acceleration.x
            msg.accel_stamped.accel.accel.angular.y = angular_acceleration.y
            msg.accel_stamped.accel.accel.angular.z = angular_acceleration.z

        for features in self.features.values():
            msg.features.append(features.to_msg())

        if self.has_camera():
            msg.has_camera = True
            msg.camera.info.header = header
            msg.camera = self.camera.to_msg()

        if self.has_shape():
            msg.has_shape = True
            for shape in self.shapes:
                msg.shapes.append(shape.to_msg())

        msg.last_update = header.stamp
        msg.expiration_duration = rospy.Duration(self.expiration_duration)
        return msg

    def draw(self, image, color, thickness=1, view_pose=None, camera=None):
        """Draws the track
        """

        if self.is_confirmed() or self.is_occluded():
            track_color = (0, 200, 0)
            text_color = (50, 50, 50)
            mask_color = (0, 100, 0)
            if self.is_static():
                track_color = (0, 200, 0)
                text_color = (50, 50, 50)
                mask_color = (0, 100, 0)
        else:
            if self.is_lost():
                track_color = (0, 0, 200)
                text_color = (250, 250, 250)
                mask_color = (0, 0, 100)
            else:
                track_color = (200, 0, 0)
                text_color = (250, 250, 250)
                mask_color = (100, 0, 0)

        if self.is_confirmed():
            if self.is_located():
                if view_pose is not None and camera is not None:
                    view_matrix = view_pose.transform()
                    camera_matrix = camera.camera_matrix()
                    dist_coeffs = camera.dist_coeffs
                    sensor_pose = Vector6D().from_transform(np.dot(np.linalg.inv(view_matrix), self.pose.transform()))
                    rot = sensor_pose.rotation().to_array()
                    depth = sensor_pose.position().z
                    # for opencv convention
                    R = euler_matrix(rot[0][0], rot[1][0], rot[2][0], "rxyz")
                    rvec = cv2.Rodrigues(R[:3, :3])[0]
                    cv2.putText(image, "{:0.3}m".format(depth),
                                (self.bbox.xmin+5, self.bbox.ymin-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .6,
                                (255, 255, 255),
                                2)
                    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs,
                                      rvec,
                                      sensor_pose.position().to_array(), 0.1)
                else:
                    depth = self.bbox.depth
                    cv2.putText(image, "{0:.2}m".format(depth),
                                (self.bbox.xmin+5, self.bbox.ymin-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .6,
                                (255, 255, 255),
                                2)
            if self.mask is not None:
                pass
                # contours, hierarchy = cv2.findContours(self.mask.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # roi = image[int(self.bbox.ymin):int(self.bbox.ymax), int(self.bbox.xmin):int(self.bbox.xmax)]
                # overlay = roi.copy()
                # cv2.drawContours(overlay, contours, -1, mask_color, 2, cv2.LINE_8, hierarchy, 100)
                # alpha = 0.6
                # roi = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
            cv2.rectangle(image, (self.bbox.xmin, self.bbox.ymax-26),
                                 (self.bbox.xmax, self.bbox.ymax),
                                 (200, 200, 200), -1)

            self.bbox.draw(image, track_color, 2)
            self.bbox.draw(image, text_color, 1)

            cv2.putText(image,
                        "{}".format(self.id[:6]),
                        (self.bbox.xmax-75, self.bbox.ymax-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        text_color,
                        1)
            cv2.putText(image,
                        self.label,
                        (self.bbox.xmin+5, self.bbox.ymax-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, text_color, 1)

            if "facial_landmarks" in self.features:
                self.features["facial_landmarks"].draw(image,
                                                       track_color,
                                                       thickness)
        else:
            self.bbox.draw(image, track_color, 1)

    def __eq__(self, other):
        return other.id == self.id
