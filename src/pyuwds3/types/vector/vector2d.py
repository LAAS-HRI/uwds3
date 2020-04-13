import numpy as np
import cv2


class Vector2D(object):
    """Represents a 2D vector"""
    def __init__(self, x=.0, y=.0):
        """Vector2D constructor"""
        self.x = x
        self.y = y

    def from_array(self, array):
        """Sets the vector from his array representation"""
        assert array.shape == (2, 1)
        self.x = array[0]
        self.y = array[1]

    def to_array(self):
        """Returns the 2D vector array representation"""
        return np.array([[self.x], [self.y]], np.float32)

    def draw(self, rgb_image, color, thickness):
        """Draws the 2D point"""
        cv2.circle(rgb_image, (self.x, self.y), 2, color, thickness=1)

    def __len__(self):
        """Returns the lenght of the vector"""
        return 2

    def __add__(self, vector):
        """Adds the given 2D vector"""
        assert len(vector) == 2
        return Vector2D(x=self.x+vector.x, y=self.y+vector.y)

    def __sub__(self, vector):
        """Subtracts the given 2D vector"""
        assert len(vector) == 2
        return Vector2D(x=self.x-vector.x, y=self.y-vector.y)

    def __str__(self):
        return("{}".format(self.to_array()))
