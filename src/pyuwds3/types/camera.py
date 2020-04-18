import uwds3_msgs
import numpy as np
from .vector.vector2d import Vector2D


class Camera(object):
    """Represents a camera sensor (real or virtual)"""

    def __init__(self,
                 fov=60.0,
                 width=480,
                 height=360,
                 clipnear=0.1,
                 clipfar=1e+3):
        """Camera constructor"""
        self.fov = fov
        self.width = width
        self.height = height
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.dist_coeffs = np.zeros((4, 1))

    def center(self):
        """Returns the camera's center"""
        return Vector2D(self.width/2, self.height/2)

    def camera_matrix(self):
        """Returns the camera matrix"""
        center = self.center()
        return np.array([[self.height, 0, center.x],
                        [0, self.height, center.y],
                        [0, 0, 1]], dtype="double")

    def projection_matrix(self):
        """Returns the projection matrix"""
        center = self.center()
        return np.array([[self.width, 0, center.x, 0],
                        [0, self.height, center.y, 0],
                        [0, 0, 1, 1]], dtype="double")

    def from_msg(self, msg, fov=60.0, clipnear=0.3, clipfar=1e+3):
        """ """
        self.fov = fov
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.width = msg.width
        self.height = msg.height
        self.dist_coeffs = np.array(msg.D)
        return self

    def to_msg(self):
        """Converts into a ROS message"""
        camera = uwds3_msgs.msg.Camera()
        camera.fov = self.fov
        camera.clipnear = self.clipnear
        camera.clipfar = self.clipfar
        camera.info.width = self.width
        camera.info.height = self.height
        camera.info.distortion_model = "blob"
        camera.info.D = list(self.dist_coeffs.flatten())
        camera.info.K = list(self.camera_matrix().flatten())
        camera.info.P = list(self.projection_matrix().flatten())
        return camera

    def __str__(self):
        return "fov:{} width:{} height:{} clipnear:{} clipfar:{}".format(self.fov,
                                                                         self.width,
                                                                         self.height,
                                                                         self.clipnear,
                                                                         self.clipfar)


class HumanVisualModel(object):
    FOV = 90.0 # human field of view
    WIDTH = 480 # image width resolution for rendering
    HEIGHT = 360  # image height resolution for rendering
    CLIPNEAR = 0.1 # clipnear
    CLIPFAR = 1e+3 # clipfar


class HumanCamera(Camera):
    def __init__(self):
        self.fov = HumanVisualModel.FOV
        self.width = HumanVisualModel.WIDTH
        self.height = HumanVisualModel.HEIGHT
        self.clipnear = HumanVisualModel.CLIPNEAR
        self.clipfar = HumanVisualModel.CLIPFAR
        self.dist_coeffs = np.zeros((4, 1))
