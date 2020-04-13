import numpy as np
import uwds3_msgs.msg
from ..vector.vector6d import Vector6D


class ShapeType(object):
    UNKNOWN = -1
    BOX = uwds3_msgs.msg.PrimitiveShape.BOX
    CYLINDER = uwds3_msgs.msg.PrimitiveShape.CYLINDER
    SPHERE = uwds3_msgs.msg.PrimitiveShape.SPHERE
    MESH = uwds3_msgs.msg.PrimitiveShape.MESH


class Shape(object):
    def __init__(self):
        self.type = ShapeType.UNKNOWN
        self.pose = Vector6D()
        self.color = np.zeros(4)
        self.scale = np.ones(3)
        self.color[3] = 1.0

    def is_box(self):
        return self.type == ShapeType.BOX

    def is_cylinder(self):
        return self.type == ShapeType.CYLINDER

    def is_sphere(self):
        return self.type == ShapeType.SPHERE

    def is_mesh(self):
        return self.type == ShapeType.MESH

    def from_msg(self, msg):
        raise NotImplementedError()

    def to_msg(self):
        raise NotImplementedError()
