from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Mesh(Shape):
    """Represents a 3D Mesh"""
    def __init__(self, mesh_resource="", name="",
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0,
                 scale_x=1., scale_y=1., scale_z=1.,
                 r=0, g=0., b=0., a=1.):
        """Mesh constructor
        """
        super(Mesh, self).__init__(ShapeType.MESH,
                                   name=name,
                                   x=x, y=y, z=z,
                                   rx=rx, ry=ry, rz=rz,
                                   scale_x=scale_x,
                                   scale_y=scale_y,
                                   scale_z=scale_z,
                                   r=r, g=g, b=b, a=a)
        self.mesh_resource = mesh_resource

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.mesh_resource = msg.mesh_resource
        self.name = msg.name
        a = msg.color.a
        r = msg.color.r
        g = msg.color.g
        b = msg.color.b
        self.color = np.array([r, g, b, a])
        self.pose.from_msg(msg.pose)
        return self

    def to_msg(self):
        """ Convert to ROS message
        """
        shape = uwds3_msgs.msg.PrimitiveShape()
        shape.type = self.type
        shape.name = self.name
        shape.mesh_resource = self.mesh_resource
        shape.scale.x = self.scale.x
        shape.scale.y = self.scale.y
        shape.scale.z = self.scale.z
        shape.color.r = self.color[0]
        shape.color.g = self.color[1]
        shape.color.b = self.color[2]
        shape.color.a = self.color[3]
        shape.pose = self.pose.to_msg()
        return shape
