import math
import numpy as np
from ...types.vector.vector3d import Vector3D
from ...types.vector.vector6d import Vector6D


class GridSampler(object):
    """
    """
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax, xdim, ydim, zdim):
        """ Grid sampler constructor
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.xdim = xdim
        self.ydim = ydim
        self.zdim = zdim
        self.zmin = zmin
        self.zmax = zmax
        self.xstep = (self.xmax - self.xmin) / float(self.xdim)
        self.ystep = (self.ymax - self.ymin) / float(self.ydim)
        self.zstep = (self.zmax - self.zmin) / float(self.zdim)
        self.data = np.zeros((xdim, ydim, zdim), dtype=np.float32)

    def cell_position(self, x, y, z):
        """
        """
        x = self.xmin+(x*self.xstep)
        y = self.ymin+(y*self.ystep)
        z = self.zmin+(z*self.zstep)
        return Vector3D(x=x, y=y, z=z)

    def random_cell_indices(self):
        x = int(np.random.random_sample()*self.xdim)
        y = int(np.random.random_sample()*self.ydim)
        z = int(np.random.random_sample()*self.zdim)
        return x, y, z

    def cell_random_pose(self, x, y, z):
        """
        """
        x = self.xmin+(x*self.xstep)
        y = self.ymin+(y*self.ystep)
        z = self.zmin+(z*self.zstep)
        rx = np.random.random_sample()*2.0*math.pi
        ry = np.random.random_sample()*2.0*math.pi
        rz = np.random.random_sample()*2.0*math.pi
        return Vector6D(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)

    def update(self, x, y, z, value):
        """
        """
        self.data[x, y, z] = value
