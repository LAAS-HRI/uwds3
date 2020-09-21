import numpy as np
import cv2
import math
import uwds3_msgs.msg
from .vector.vector2d import Vector2D
from .features import Features


class BoundingBox(object):
    """Represents a 2D+depth aligned bounding box in the image space"""

    def __init__(self, xmin, ymin, xmax, ymax, depth=None):
        """BoundingBox constructor"""
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        self.depth = depth

    def center(self):
        """Returns the bbox's center in pixels"""
        return Vector2D(self.xmin + self.width()/2, self.ymin + self.height()/2)

    def width(self):
        """Returns the bbox's width in pixels"""
        return int(self.xmax - self.xmin)

    def height(self):
        """Returns the bbox's height in pixels"""
        return int(self.ymax - self.ymin)

    def diagonal(self):
        """Returns the bbox's diagonal in pixels"""
        return int(math.sqrt(pow(self.width(), 2)+pow(self.height(), 2)))

    def radius(self):
        """Returns the bbox's radius in pixels"""
        return int(self.diagonal()/2.0)

    def area(self):
        """Returns the bbox's area in pixels"""
        return (self.width()*self.height()+1)

    def has_depth(self):
        return self.depth is not None

    def draw(self, frame, color, thickness):
        """Draws the bbox"""
        cv2.rectangle(frame, (self.xmin, self.ymin), (self.xmax, self.ymax), color, thickness)

    def from_array(self, array):
        """ """
        assert array.shape == (5, 1) or array.shape == (6, 1)
        self.xmin = array[0][0]
        self.ymin = array[1][0]
        self.xmax = array[2][0]
        self.ymax = array[3][0]
        if array.shape == (6, 1):
            self.depth = array[4][0]

    def to_array(self):
        """ """
        if self.depth is not None:
            return np.array([self.xmin, self.ymin, self.xmax, self.ymax, self.depth], np.float32)
        else:
            return np.array([self.xmin, self.ymin, self.xmax, self.ymax], np.float32)

    def from_msg(self, msg):
        self.xmin = msg.xmin
        self.ymin = msg.ymin
        self.xmax = msg.xmax
        self.ymax = msg.ymax
        if msg.has_depth is True:
            self.depth = msg.depth
        else:
            self.depth = None
        return self

    def to_msg(self):
        msg = uwds3_msgs.msg.BoundingBox()
        msg.xmin = int(self.xmin)
        msg.ymin = int(self.ymin)
        msg.xmax = int(self.xmax)
        msg.ymax = int(self.ymax)
        if self.has_depth():
            msg.has_depth = True
            msg.depth = self.depth
        else:
            msg.has_depth = False
        return msg

    def to_xyxy(self):
        """ """
        return self.to_array()

    def from_mxywh(self, xmin, ymin, w, h):
        """ """
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmin + w)
        self.ymax = int(ymin + h)
        return self

    def to_mxywh(self):
        """ """
        if self.depth is None:
            return np.array([self.xmin, self.ymin, self.width(), self.height()], np.float32)
        else:
            return np.array([self.xmin, self.ymin, self.width(), self.height(), self.depth], np.float32)

    def from_cxywh(self, cx, cy, w, h):
        """ """
        self.xmin = int(cx - w/2.0)
        self.ymin = int(cy - h/2.0)
        self.xmax = int(cx + w/2.0)
        self.ymax = int(cy + h/2.0)
        return self

    def to_cxywh(self):
        """ """
        c = self.center()
        if self.depth is None:
            return np.array([c.x, c.y, self.width(), self.height()], np.float32)
        else:
            return np.array([c.x, c.y, self.width(), self.height(), self.depth], np.float32)

    def features(self, image_width, image_height, max_depth=25):
        """Returns the bbox geometric features"""
        features = [self.xmin/float(image_width),
                    self.ymin/float(image_height),
                    self.xmax/float(image_width),
                    self.ymax/float(image_height),
                    min(self.depth/float(max_depth), float(1.0))]
        return Features("geometric", np.array(features).flatten(), 0.89)

    def __str__(self):
        return "{}".format(self.to_array())
