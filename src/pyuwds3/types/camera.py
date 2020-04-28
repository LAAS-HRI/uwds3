import math
import uwds3_msgs.msg
import numpy as np
from .vector.vector2d import Vector2D


class Camera(object):
    """Represents a camera sensor (real or virtual)"""

    def __init__(self,
                 width=480,
                 height=360,
                 clipnear=0.1,
                 clipfar=1e+3):
        """Camera constructor"""
        self.width = width
        self.height = height
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.dist_coeffs = np.zeros((4, 1))
        self.center = Vector2D(self.width/2.0, self.height/2.0)
        self.focal_length = Vector2D(self.width, self.height)

    def hfov(self):
        return math.degrees(2.0*math.atan2(self.width, 2*self.focal_length.x))

    def center(self):
        """Returns the camera's center"""
        return self.center

    def get_local_length(self):
        return self.focal_length

    def camera_matrix(self):
        """Returns the camera matrix"""
        return np.array([[self.focal_length.x, 0, self.center.x],
                        [0, self.focal_length.y, self.center.y],
                        [0, 0, 1]], dtype="double")

    def projection_matrix(self):
        """Returns the projection matrix"""
        return np.array([[self.focal_length.x, 0, self.center.x, 0],
                        [0, self.focal_length.y, self.center.y, 0],
                        [0, 0, 1, 0]], dtype="double")

    def from_msg(self, msg, clipnear=0.1, clipfar=1e+3):
        """ """
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.width = msg.width
        self.height = msg.height
        self.dist_coeffs = np.array(msg.D)
        self.focal_length.x = msg.K[0]
        self.focal_length.y = msg.K[4]
        self.center.x = msg.K[2]
        self.center.y = msg.K[5]
        return self

    def to_msg(self):
        """Converts into a ROS message"""
        msg = uwds3_msgs.msg.Camera()
        msg.fov = self.hfov()
        msg.clipnear = self.clipnear
        msg.clipfar = self.clipfar
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.distortion_model = "blob"
        msg.info.D = list(self.dist_coeffs.flatten())
        msg.info.K = list(self.camera_matrix().flatten())
        msg.info.P = list(self.projection_matrix().flatten())
        return msg

    def __str__(self):
        return "hfov:{} width:{} height:{} clipnear:{} clipfar:{}".format(self.fov,
                                                                         self.width,
                                                                         self.height,
                                                                         self.clipnear,
                                                                         self.clipfar)

class HumanVisualModel(object):
    WIDTH = 480 # image width resolution for rendering
    HEIGHT = 360  # image height resolution for rendering
    CLIPNEAR = 0.1 # clipnear
    CLIPFAR = 1e+3 # clipfar


class HumanCamera(Camera):
    def __init__(self):
        super(HumanCamera, self).__init__()
        self.width = HumanVisualModel.WIDTH
        self.height = HumanVisualModel.HEIGHT
        self.clipnear = HumanVisualModel.CLIPNEAR
        self.clipfar = HumanVisualModel.CLIPFAR
        self.dist_coeffs = np.zeros((4, 1))
