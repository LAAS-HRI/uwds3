import cv2
import math
from ...types.detection import Detection


class MedianFlowTracker(object):
    """Single object tracker based on opencv medianflow tracker"""
    def __init__(self, track):
        """ """
        self.tracker = None
        self.track = track

    def update(self, rgb_image, detection, depth_image=None):
        """ """
        self.tracker = cv2.TrackerMedianFlow_create()
        xmin = detection.bbox.xmin
        ymin = detection.bbox.ymin
        w = detection.bbox.width()
        h = detection.bbox.height()
        bbox = (xmin, ymin, w, h)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.tracker.init(bgr_image, bbox)

    def predict(self, rgb_image, depth_image=None):
        """ """
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        if self.tracker is not None:
            success, bbox = self.tracker.update(bgr_image)
            xmin, ymin, w, h = bbox
            xmin = 0 if xmin < 0 else xmin
            ymin = 0 if ymin < 0 else ymin
            x = int(xmin + w/2.0)
            y = int(ymin + h/2.0)
            if depth_image is not None:
                x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                depth = depth_image[int(y-1)][int(x)]/1000.0
                if math.isnan(depth) or depth == 0.0:
                    depth = None
            else:
                depth = None
            prediction = Detection(xmin, ymin, xmin+w, ymin+h, self.track.label, 0.8, depth=depth)
            return success, prediction
        else:
            return False, None
