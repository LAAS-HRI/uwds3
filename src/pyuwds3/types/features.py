import uwds3_msgs
import numpy as np


class Features(object):
    """ Represent a features vector with a confidence
    """
    def __init__(self, name, dimensions, data, confidence):
        """ Features constructor
        """
        self.name = name
        self.data = data
        self.dimensions = dimensions
        self.confidence = confidence
        self.age = 1

    def update(self, data, confidence):
        """ Update the features using a simple filter
        """
        alpha = 1.0 / self.age
        self.data = self.data + (alpha * confidence * (data - self.data))
        self.confidence = 1 - alpha
        self.age += 1

    def to_array(self):
        """ Convert to numpy array
        """
        return np.array(self.data, np.float32)

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.name = msg.name
        self.dimensions = tuple(msg.dimensions)
        self.data = np.array(msg.data).reshape(tuple(self.dimensions))
        return self

    def to_msg(self):
        """ Converts into ROS message
        """
        return uwds3_msgs.msg.Features(name=self.name,
                                       dimensions=list(self.dimensions),
                                       data=list(self.data.flatten()),
                                       confidence=self.confidence)
