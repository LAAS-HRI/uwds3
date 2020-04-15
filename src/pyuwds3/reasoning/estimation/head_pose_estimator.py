import cv2
from math import pi
import numpy as np
from tf.transformations import euler_matrix, euler_from_matrix, is_same_transform
from ...types.vector.vector6d import Vector6D
from ...types.vector.vector3d import Vector3D


MAX_DIST = 2.5

RX_OFFSET = 0.0
RY_OFFSET = 0.0
RZ_OFFSET = 0.0

RX_OFFSET = - pi/2.0
RY_OFFSET = pi
RZ_OFFSET = 0.0


class HeadPoseEstimator(object):
    def __init__(self, face_3d_model_filename):
        """HeadPoseEstimator constructor"""
        self.model_3d = np.load(face_3d_model_filename)/100.0/4.6889/2.0
        self.offset = Vector6D(rx=RX_OFFSET, ry=RY_OFFSET, rz=RZ_OFFSET).transform()

    def __check_consistency(self, tvec, rvec):
        consistent = True
        if tvec[2][0] > MAX_DIST or tvec[2][0] < 0:
            consistent = False
        return consistent

    def __add_offset(self, r, x, y, z):
        r[0][0] += x
        r[1][0] += y
        r[2][0] += z

    def __rodrigues2euler(self, rvec):
        R = cv2.Rodrigues(rvec)[0]
        T = np.zeros((4, 4))
        T[3, 3] = 1.0
        euler = np.array(euler_from_matrix(R, "sxyz"))
        euler[2] *= -1
        return euler.reshape((3, 1))

    def __euler2rodrigues(self, rot):
        rot[2][0] = 0
        R = euler_matrix(rot[0][0], rot[1][0], -rot[2][0], "sxyz")
        rvec = cv2.Rodrigues(R[:3, :3])[0]
        return rvec

    def estimate(self, faces, view_pose, camera):
        """Estimate the head pose of the given face (z forward for rendering)"""
        view_matrix = view_pose.transform()
        camera_matrix = camera.camera_matrix()
        dist_coeffs = camera.dist_coeffs
        for f in faces:
            if f.is_confirmed():
                if "facial_landmarks" in f.features:
                    points_2d = f.features["facial_landmarks"].data
                    if f.pose is not None:
                        world_transform = f.pose.transform()
                        sensor_pose = Vector6D().from_transform(np.dot(np.dot(np.linalg.inv(world_transform), world_transform), np.linalg.inv(self.offset)))
                        r = sensor_pose.rotation().to_array()
                        t = sensor_pose.position().to_array()
                        self.__add_offset(r, -RX_OFFSET, -RY_OFFSET, -RZ_OFFSET)
                        rvec = self.__euler2rodrigues(r)
                        success, rvec, tvec, _ = cv2.solvePnPRansac(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec, tvec=t)
                        success = self.__check_consistency(tvec, rvec)
                    else:
                        success, rvec, tvec, _ = cv2.solvePnPRansac(self.model_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                        success = self.__check_consistency(tvec, rvec)
                    if success:
                        r = self.__rodrigues2euler(rvec)
                        self.__add_offset(r, RX_OFFSET, RY_OFFSET, RZ_OFFSET)
                        sensor_pose = Vector6D(x=tvec[0][0], y=tvec[1][0], z=tvec[2][0],
                                               rx=r[0][0], ry=r[1][0], rz=.0)
                        world_pose = Vector6D().from_transform(np.dot(view_matrix, sensor_pose.transform()))
                        f.bbox.depth = tvec[2][0]
                        f.update_pose(world_pose.position(), rotation=world_pose.rotation())
