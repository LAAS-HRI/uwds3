import rospy
import numpy as np
import cv2


class ScalarStable(object):
    """Represents a stabilized scalar"""
    def __init__(self,
                 x=.0,
                 vx=.0,
                 p_cov=.03, m_cov=.01,
                 time=None):
        """ScalarStabilized constructor"""
        self.x = x
        self.vx = vx
        self.p_cov = p_cov
        self.m_cov = m_cov
        self.filter = cv2.KalmanFilter(2, 1)
        self.filter.statePost = self.to_array()
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.__update_noise_cov(p_cov, m_cov)
        if time is None:
            self.last_update = rospy.Time().now()
        else:
            self.last_update = time

    def from_array(self, array):
        """Updates the scalar stabilized state from array"""
        assert array.shape == (2, 1)
        self.x = array[0]
        self.vx = array[1]
        self.filter.statePre = self.filter.statePost

    def to_array(self):
        """Returns the scalar stabilizer state array representation"""
        return np.array([[self.x], [self.vx]], np.float32)

    def position(self):
        """Returns the scalar's position"""
        return self.x

    def velocity(self):
        """Returns the scalar's velocity"""
        return self.vx

    def update(self, x, time=None, m_cov=None):
        """Updates/Filter the scalar"""
        if m_cov is not None:
            self.__update_noise_cov(self.p_cov, m_cov)
        self.__update_time(time=time)
        self.filter.predict()
        measurement = np.array([[np.float32(x)]])
        assert measurement.shape == (1, 1)
        self.filter.correct(measurement)
        self.from_array(self.filter.statePost)

    def predict(self, time=None):
        """Predicts the scalar state"""
        self.__update_time(time=time)
        self.filter.predict()
        self.from_array(self.filter.statePost)

    def __update_noise_cov(self, p_cov, m_cov):
        """Updates the process and measurement covariances"""
        self.filter.processNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * p_cov

        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * m_cov

    def __update_transition(self, dt):
        self.filter.transitionMatrix = np.array([[1, dt],
                                                 [0, 1]], np.float32)

    def __update_time(self, time=None):
        if time is None:
            now = rospy.Time().now()
        else:
            now = time
        elapsed_time = now - self.last_update
        self.last_update = now
        self.__update_transition(elapsed_time.to_sec())

    def __len__(self):
        return 1

    def __add__(self, scalar):
        return self.x + scalar.x

    def __sub__(self, scalar):
        return self.x - scalar.x

    def __str__(self):
        return("{}".format(self.to_array()))
