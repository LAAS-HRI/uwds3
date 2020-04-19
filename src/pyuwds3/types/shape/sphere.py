from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Sphere(Shape):
    """Represents a 3D sphere"""
    def __init__(self, d, name="",
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0,
                 scale_x=1., scale_y=1., scale_z=1.):
        """ Sphere constructor
        """
        super(Sphere, self).__init__(ShapeType.SPHERE,
                                     name=name,
                                     x=x, y=y, z=z,
                                     rx=rx, ry=ry, rz=rz,
                                     scale_x=scale_x,
                                     scale_y=scale_y,
                                     scale_z=scale_z)
        self.d = d

    def radius(self):
        """ Returns the sphere's radius in meters
        """
        return self.width()/2.0

    def width(self):
        """ Returns the sphere's width in meters
        """
        return self.d

    def height(self):
        """ Returns the sphere's height in meters
        """
        return self.d

    def volume(self):
        """ Returns the cylinder volume in cube meters
        """
        return(4.0*pi*pow(self.radius(), 2))

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.d = msg.dimensions[0]
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
        shape.dimensions.append(self.width())
        shape.scale.x = 1.0
        shape.scale.y = 1.0
        shape.scale.z = 1.0
        shape.color.r = self.color[0]
        shape.color.g = self.color[1]
        shape.color.b = self.color[2]
        shape.color.a = self.color[3]
        shape.pose = self.pose.to_msg()
        return shape
