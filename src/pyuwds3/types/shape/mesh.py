from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Mesh(Shape):
    """Represents a 3D Mesh"""
    def __init__(self, mesh_resource,
                 scale_x=1., scale_y=1., scale_z=1.,
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0):
        """Cylinder constructor"""
        self.type = ShapeType.MESH
        self.pose = Vector6D(x=x, y=y, z=z,
                             rx=rx, ry=ry, rz=rz)
        self.mesh_resource = mesh_resource
        self.color = np.zeros(4)
        self.scale = np.array([scale_x, scale_y, scale_z])
        self.color[3] = 1.0

    def center(self):
        """Returns the bbox's center in pixels"""
        return self.position

    def radius(self):
        """Returns the bbox's radius in meters"""
        return self.width()/2.0

    def width(self):
        """Returns the bbox's width in meters"""
        return max(self.x, self.y)

    def height(self):
        """Returns the bbox's height in meters"""
        return self.z

    def area(self):
        """Returns the bbox's area in cube meters"""
        raise NotImplementedError()

    def from_msg(self, msg):
        self.mesh_resource = msg.mesh_resource
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
        shape.mesh_resource = self.mesh_resource
        shape.scale.x = self.scale[0]
        shape.scale.y = self.scale[1]
        shape.scale.z = self.scale[2]
        shape.color.r = self.color[0]
        shape.color.g = self.color[1]
        shape.color.b = self.color[2]
        shape.color.a = self.color[3]
        shape.pose = self.pose.to_msg()
        return shape
