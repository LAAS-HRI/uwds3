from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Cylinder(Shape):
    """ Represents a 2D BoundingBox + depth in the world space (e.g. cylinder)
    """
    def __init__(self, w=0., h=0., name="",
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0,
                 scale_x=1., scale_y=1., scale_z=1.):
        """ Cylinder constructor
        """
        super(Cylinder, self).__init__(ShapeType.CYLINDER,
                                       name=name,
                                       x=x, y=y, z=z,
                                       rx=rx, ry=ry, rz=rz,
                                       scale_x=scale_x,
                                       scale_y=scale_y,
                                       scale_z=scale_z)
        self.w = w
        self.h = h

    def radius(self):
        """ Returns the cylinder's radius in meters
        """
        return self.width()/2.0

    def width(self):
        """ Returns the cylinder's width in meters
        """
        return self.w

    def height(self):
        """ Returns the cylinder's height in meters
        """
        return self.h

    def volume(self):
        """ Returns the cylinder volume in cube meters
        """
        return(pi*pow(self.radius(), 2)*self.h)

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.w = msg.dimensions[0]
        self.h = msg.dimensions[1]
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
