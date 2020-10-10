import numpy as np
import geometry_msgs


class Vector3D(object):
    """Represents a 3D vector"""
    def __init__(self, x=.0, y=.0, z=.0):
        """Vector 3D constructor"""
        self.x = x
        self.y = y
        self.z = z

    def from_array(self, array):
        """ """
        assert array.shape == (3, 1)
        self.x = array[0][0]
        self.y = array[1][0]
        self.z = array[2][0]
        return self

    def to_array(self):
        """Returns the 3D vector's array representation"""
        return np.array([[self.x], [self.y], [self.z]])

    def __len__(self):
        """Returns the lenght of the vector"""
        return 3

    def __add__(self, vector):
        """Adds the given 3D vector"""
        assert len(vector) == 3
        return Vector3D(x=self.x+vector.x,
                        y=self.y+vector.y,
                        z=self.z+vector.z)

    def __sub__(self, vector):
        """Substracts the given 3D vector"""
        assert len(vector) == 3
        return Vector3D(x=self.x-vector.x,
                        y=self.y-vector.y,
                        z=self.z-vector.z)

    def __eq__(self, vector):
        return vector.x == self.x and vector.y == self.y and vector.z == self.z

    def to_msg(self):
        """Converts to ROS message"""
        return geometry_msgs.msg.Vector3(x=self.x, y=self.y, z=self.z)

    def from_msg(self, msg):
        """Converts to ROS message"""
        self.x = msg.x
        self.y = msg.y
        self.z = msg.z
        return self

    def __str__(self):
        return("{}".format(self.to_array()))
