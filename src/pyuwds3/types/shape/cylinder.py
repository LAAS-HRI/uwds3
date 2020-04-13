from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Cylinder(Shape):
    """Represents a 2D BoundingBox + depth in the world space (e.g. cylinder)"""
    def __init__(self, w, h,
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0):
        """Cylinder constructor"""
        self.type = ShapeType.CYLINDER
        self.pose = Vector6D(x=x, y=y, z=z,
                             rx=rx, ry=ry, rz=rz)
        self.w = w
        self.h = h
        self.color = np.zeros(4)
        self.color[3] = 1.0

    def center(self):
        """Returns the bbox's center in pixels"""
        return self.position

    def radius(self):
        """Returns the cylinder's radius in meters"""
        return self.width()/2.0

    def width(self):
        """Returns the cylinder's width in meters"""
        return self.w

    def height(self):
        """Returns the cylinder's height in meters"""
        return self.h

    def area(self):
        """Returns the cylinder's area in cube meters"""
        return 2.0*pi*self.radius()*self.height()

    def from_msg(self, msg):
        self.w = msg.dimensions[0]
        self.h = msg.dimensions[1]
        a = msg.color.a
        r = msg.color.r
        g = msg.color.g
        b = msg.color.b
        self.color = np.array([r, g, b, a])
        self.pose.from_msg(msg.pose)
        return self

    def to_msg(self):
        shape = uwds3_msgs.msg.PrimitiveShape()
        shape.type = self.type
        shape.dimensions.append(self.width())
        shape.dimensions.append(self.height())
        shape.scale.x = 1.0
        shape.scale.y = 1.0
        shape.scale.z = 1.0
        shape.color.r = self.color[0]
        shape.color.g = self.color[1]
        shape.color.b = self.color[2]
        shape.color.a = self.color[3]
        shape.pose = self.pose.to_msg()
        return shape
