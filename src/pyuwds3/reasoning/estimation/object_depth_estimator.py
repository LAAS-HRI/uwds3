import numpy as np
import cv2


class ObjectDepthEstimator(object):
    """
    """
    def __init__(self, mode="center"):
        modes = ["center", "median", "mean"]
        self.mode = mode
        if self.mode not in modes:
            raise ValueError("Invalid mode provided should be one of: {}".format(modes))

    def estimate(self, depth_image, objects):
        """
        """
        if depth_image is not None:
            for o in objects:
                if self.mode == "center":
                    x = int(o.bbox.xmin + o.bbox.width()/2.0)
                    y = int(o.bbox.ymin + o.bbox.height()/2.0)
                    x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                    y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                    depth = depth_image[int(y)][int(x)]/1000.0
                elif self.mode == "median" or self.mode == "mean":
                    xmin = int(o.bbox.xmin)
                    ymin = int(o.bbox.ymin)
                    h = int(o.bbox.height())
                    w = int(o.bbox.width())
                    if o.has_mask():
                        mask_resized = cv2.resize(o.mask, (w, h))
                        mask_normalized = mask_resized/255
                        masked_depth = (depth_image[ymin:ymin+h, xmin:xmin+w])*mask_normalized
                    else:
                        masked_depth = (depth_image[ymin:ymin+h, xmin:xmin+w])
                    if self.mode == "median":
                        depth = np.median(np.where(masked_depth.flatten() > 0.0))/1000.0
                    else:
                        depth = np.mean(np.where(masked_depth.flatten() > 0.0))/1000.0
                else:
                    pass
                o.bbox.depth = depth
