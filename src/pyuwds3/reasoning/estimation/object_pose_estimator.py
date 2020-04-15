import numpy as np
from ...types.vector.vector6d import Vector6D
from ...types.vector.vector3d import Vector3D


class ObjectPoseEstimator(object):
    def estimate(self, objects, view_pose, camera):
        """ """
        view_matrix = view_pose.transform()
        camera_matrix = camera.camera_matrix()
        for o in objects:
            if o.bbox.depth is not None:
                fx = camera_matrix[0][0]
                fy = camera_matrix[1][1]
                cx = camera_matrix[0][2]
                cy = camera_matrix[1][2]
                c = o.bbox.center()
                z = o.bbox.depth
                x = (c.x - cx) * z / fx
                y = (c.y - cy) * z / fy
                sensor_transform = Vector6D(x=x, y=y, z=z).transform()
                world_pose = Vector6D().from_transform(np.dot(view_matrix, sensor_transform))
                position = world_pose.position()
                rotation = Vector3D()
                o.update_pose(position, rotation)
