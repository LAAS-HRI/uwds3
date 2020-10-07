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
                 clipfar=25):
        """Camera constructor"""
        self.width = width
        self.height = height
        self.clipnear = clipnear
        self.clipfar = clipfar
        self.dist_coeffs = np.zeros((4, 1))
        self.center = Vector2D(self.width/2.0, self.height/2.0)
        self.focal_length = Vector2D(self.width, self.height)

    def fov(self):
        """ Returns the diagonal field of view (used in openGL rendering)
        """
        d = math.sqrt(pow(self.width, 2) + pow(self.height, 2))
        return math.degrees(2 * math.atan2(d/2.0, self.width))

    def center(self):
        """ Returns the camera's center
        """
        return self.center

    def get_focal_length(self):
        """ Returns the focal length
        """
        return self.focal_length

    def camera_matrix(self):
        """Returns the camera matrix"""
        return np.array([[self.focal_length.x, 0, self.center.x],
                        [0, self.focal_length.x, self.center.y],
                        [0, 0, 1]], dtype="double")

    def projection_matrix(self):
        """Returns the projection matrix"""
        return np.array([[self.focal_length.x, 0, self.center.x, 0],
                        [0, self.focal_length.x, self.center.y, 0],
                        [0, 0, 1, 0]], dtype="double")

    def from_msg(self, msg, clipnear=0.1, clipfar=1e+3):
        """
        """
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
        msg.fov = self.fov()
        msg.clipnear = self.clipnear
        msg.clipfar = self.clipfar
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.distortion_model = "blob"
        msg.info.D = list(self.dist_coeffs.flatten())
        msg.info.K = list(self.camera_matrix().flatten())
        msg.info.P = list(self.projection_matrix().flatten())
        return msg

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "fov:{:.3}\n\r".format(self.fov()) \
                + "width:{}\n\r".format(self.width) \
                + "height:{}\n\r".format(self.height) \
                + "clipnear:{}\n\r".format(self.clipnear) \
                + "clipfar:{}".format(self.clipfar) \



class HumanVisualModel(object):
    """
    """
    WIDTH = 480 # image width resolution for rendering
    HEIGHT = 360  # image height resolution for rendering
    CLIPNEAR = 0.1 # clipnear
    CLIPFAR = 25 # clipfar


class HumanCamera(Camera):
    def __init__(self):
        super(HumanCamera, self).__init__(width=HumanVisualModel.WIDTH,
                                          height=HumanVisualModel.HEIGHT,
                                          clipnear=HumanVisualModel.CLIPNEAR,
                                          clipfar=HumanVisualModel.CLIPFAR)
