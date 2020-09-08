from math import pi
import numpy as np
import uwds3_msgs
from .shape import Shape, ShapeType
from ..vector.vector6d import Vector6D


class Box(Shape):
    """Represents a 3D Box"""
    def __init__(self, dim_x=0., dim_y=0., dim_z=0., name="",
                 x=.0, y=.0, z=.0,
                 rx=.0, ry=.0, rz=.0,
                 scale_x=1., scale_y=1., scale_z=1.):
        """Box constructor
        """
        super(Box, self).__init__(ShapeType.BOX,
                                  name=name,
                                  x=x, y=y, z=z,
                                  rx=rx, ry=ry, rz=rz,
                                  scale_x=scale_x,
                                  scale_y=scale_y,
                                  scale_z=scale_z)
        self.x = dim_x
        self.y = dim_y
        self.z = dim_z

    def radius(self):
        """Returns the mesh's radius in meters
        """
        return max(self.x/2.0, max(self.y/2.0, self.z/2.0))

    def width(self):
        """Returns the bbox's width in meters (x dim)
        """
        return self.x

    def lenght(self):
        """Returns the bbox's lenght in meters (y dim)
        """
        return self.y

    def height(self):
        """Returns the bbox's height in meters (z dim)
        """
        return self.z

    def volume(self):
        """ Returns the bbox volume in cube meters
        """
        return(self.x*self.y*self.z)

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
        shape.dimensions.append(self.x)
        shape.dimensions.append(self.y)
        shape.dimensions.append(self.z)
        shape.scale.x = 1.0
        shape.scale.y = 1.0
        shape.scale.z = 1.0
        shape.color.r = self.color[0]
        shape.color.g = self.color[1]
        shape.color.b = self.color[2]
        shape.color.a = self.color[3]
        shape.pose = self.pose.to_msg()
        return shape
