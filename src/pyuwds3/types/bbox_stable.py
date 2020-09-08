import numpy as np
from .vector.vector2d_stable import Vector2DStable
from .vector.scalar_stable import ScalarStable
from .bbox import BoundingBox


class BoundingBoxStable(BoundingBox):
    """ """
    def __init__(self, xmin=0, ymin=0, xmax=0, ymax=0, depth=None,
                 p_cov_c=0.85, m_cov_c=0.003,
                 p_cov_a=0.85, m_cov_a=1e-9,
                 p_cov_h=0.85, m_cov_h=1e-9, time=None):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)
        w = xmax - xmin
        h = ymax - ymin
        center = self.center()
        x = center.x
        y = center.y
        if h == 0:
            a = 1.0
        else:
            a = w/float(h)
        self.center_filter = Vector2DStable(x=x, y=y, p_cov=p_cov_c, m_cov=m_cov_c, time=time)
        self.aspect_filter = ScalarStable(x=a, p_cov=p_cov_a, m_cov=m_cov_a, time=time)
        self.height_filter = ScalarStable(x=h, p_cov=p_cov_h, m_cov=m_cov_h, time=time)
        if depth is not None:
            self.depth = float(depth)
        else:
            self.depth = None

    def update(self, xmin, ymin, xmax, ymax, depth=None, time=None):
        w = xmax - xmin
        h = ymax - ymin
        self.center_filter.update(xmin + w/2.0, ymin + h/2.0, time=time)
        self.aspect_filter.update(w/float(h), time=time)
        self.height_filter.update(h, time=time)
        if depth is not None:
            self.depth = depth
        h = self.height_filter.x
        w = self.height_filter.x * self.aspect_filter.x
        x = self.center_filter.x
        y = self.center_filter.y
        self.xmin = x - w/2.0
        self.ymin = y - h/2.0
        self.xmax = x + w/2.0
        self.ymax = y + h/2.0

    def predict(self, time=None):
        self.center_filter.predict(time=time)
        self.aspect_filter.predict(time=time)
        self.height_filter.predict(time=time)
        h = self.height_filter.x
        w = self.height_filter.x * self.aspect_filter.x
        x = self.center_filter.x
        y = self.center_filter.y
        self.xmin = x - w/2.0
        self.ymin = y - h/2.0
        self.xmax = x + w/2.0
        self.ymax = y + h/2.0

    def update_cov(self, p_cov, m_cov):
        self.center_filter.update_cov(p_cov, m_cov)
        self.aspect_filter.update_cov(p_cov, m_cov)
        self.height_filter.update_cov(p_cov, m_cov)
        if self.depth_filter is not None:
            self.depth_filter.update_cov(p_cov, m_cov)
