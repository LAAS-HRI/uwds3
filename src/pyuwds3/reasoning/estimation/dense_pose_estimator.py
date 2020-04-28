import numpy as np
import cv2


class DensePoseEstimator(object):
    def __init__(self):
        pass

    def estimate(self, depth_image, person_tracks, view_pose):
        if depth_image is not None:
            for p in person_tracks:
                if p.is_located() and p.is_confirmed():
                    xmin = int(p.bbox.xmin)
                    ymin = int(p.bbox.ymin)
                    h = int(p.bbox.height())
                    w = int(p.bbox.width())
                    depth_image_cropped = depth_image[ymin:ymin+h, xmin:xmin+w]
                    mask = depth_image_cropped.copy()/1000.0
                    mask[mask > p.bbox.depth + 0.4] = 0
                    mask[mask < p.bbox.depth - 0.4] = 0
                    mask[mask != 0] = 255
                    p.mask = mask
