import numpy as np
import uwds3_msgs.msg
from ..vector.vector6d import Vector6D
from ..vector.vector3d import Vector3D


class ShapeType(object):
    BOX = uwds3_msgs.msg.PrimitiveShape.BOX
    CYLINDER = uwds3_msgs.msg.PrimitiveShape.CYLINDER
    SPHERE = uwds3_msgs.msg.PrimitiveShape.SPHERE
    MESH = uwds3_msgs.msg.PrimitiveShape.MESH


class Shape(object):
    def __init__(self, type, name="",
                 scale_x=1., scale_y=1., scale_z=1.,
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0,
                 r=0, g=0., b=0., a=1.):
        """ Shape constructor
        """
        self.name = name
        self.type = type
        self.pose = Vector6D(x=x, y=y, z=z,
                             rx=rx, ry=ry, rz=rz)
        self.scale = Vector3D(x=scale_x, y=scale_y, z=scale_z)
        self.color = np.zeros(4)
        self.color[0] = r
        self.color[1] = g
        self.color[2] = b
        self.color[3] = a

    def is_box(self):
        """ Returns True if is a box
        """
        return self.type == ShapeType.BOX

    def is_cylinder(self):
        """ Returns True if is a cylinder
        """
        return self.type == ShapeType.CYLINDER

    def is_sphere(self):
        """ Returns True if is a sphere
        """
        return self.type == ShapeType.SPHERE

    def is_mesh(self):
        """ Returns True if is a mesh
        """
        return self.type == ShapeType.MESH

    def from_msg(self, msg):
        """ Serialization method to implement
        """
        raise NotImplementedError()

    def to_msg(self):
        """ Serialization method to implement
        """
        raise NotImplementedError()
