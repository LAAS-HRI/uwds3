import rospy
import numpy as np
import cv2
from .vector2d import Vector2D


class Vector2DStable(Vector2D):
    """"Represents a 2D vector stabilized"""
    def __init__(self,
                 x=.0, y=.0,
                 vx=.0, vy=.0,
                 p_cov=.03, m_cov=.01,
                 time=None):
        """Vector2DStablized constructor"""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.statePost = self.to_array()
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.__update_noise_cov(p_cov, m_cov)
        if time is None:
            self.last_update = rospy.Time().now()
        else:
            self.last_update = time

    def from_array(self, array):
        """Updates the 2D vector stabilized state from array"""
        assert array.shape == (4, 1)
        self.x = array[0]
        self.y = array[1]
        self.vx = array[2]
        self.vy = array[3]
        self.filter.statePost = array
        self.filter.statePre = self.filter.statePost

    def to_array(self):
        """Returns the 2D vector stabilizer state array representation"""
        return np.array([[self.x], [self.y], [self.vx], [self.vy]], np.float32)

    def position(self):
        """Returns the 2D vector stabilized position"""
        return Vector2D(x=self.x, y=self.y)

    def velocity(self):
        """"Returns the 2D vector stabilized velocity"""
        return Vector2D(x=self.vx, y=self.vy)

    def update(self, x, y, time=None):
        """Updates/Filter the 2D vector"""
        self.__update_time(time=time)
        self.filter.predict()
        measurement = np.array([[x], [y]], np.float32)
        assert measurement.shape == (2, 1)
        self.filter.correct(measurement)
        self.from_array(self.filter.statePost)

    def predict(self, time=None):
        """Predicts the 2D vector based on motion model"""
        self.__update_time(time=time)
        self.filter.predict()
        self.from_array(self.filter.statePost)

    def __update_noise_cov(self, p_cov, m_cov):
        """Updates the process and measurement covariances"""
        self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * p_cov

        self.filter.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * m_cov

    def __update_transition(self, dt):
        self.filter.transitionMatrix = np.array([[1, 0, dt, 0],
                                                 [0, 1, 0, dt],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

    def __update_time(self, time=None):
        if time is None:
            now = rospy.Time().now()
        else:
            now = time
        elapsed_time = now - self.last_update
        self.last_update = now
        self.__update_transition(elapsed_time.to_sec())

    def __str__(self):
        return("{}".format(self.to_array()))
