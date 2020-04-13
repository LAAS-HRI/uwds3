import cv2
import uwds3_msgs.msg
from pyuwds3.types.shape.sphere import Sphere
from pyuwds3.types.shape.box import Box
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

K = 3


class ShapeEstimator(object):
    """ """
    def estimate(self, rgb_image, objects_tracks, camera):
        """ """
        camera_matrix = camera.camera_matrix()
        dist_coeffs = camera.dist_coeffs
        for o in objects_tracks:
            try:
                if o.is_confirmed() and o.bbox.height() > 0:
                    if o.bbox.depth is not None:
                        if o.label != "person":
                            if not o.has_shape():
                                shape = o.bbox.cylinder(camera_matrix, dist_coeffs)
                                if o.label == "face":
                                    shape = Sphere(shape.width()*2.0)
                                if o.label == "hand":
                                    shape = Sphere(shape.width())
                                if o.label == "table":
                                    shape = Box(shape.height(), shape.width(), .01)
                                shape.pose.pos.x = .0
                                shape.pose.pos.y = .0
                                shape.pose.pos.z = .0
                                shape.color = self.compute_dominant_color(rgb_image, o.bbox)
                                o.shapes.append(shape)
                        else:
                            shape = o.bbox.cylinder(camera_matrix, dist_coeffs)
                            z = o.pose.pos.z
                            shape.pose.pos.x = .0
                            shape.pose.pos.y = .0
                            shape.pose.pos.z = -(z - shape.h/2.0)/2.0
                            if not o.has_shape():
                                shape.color = self.compute_dominant_color(rgb_image, o.bbox)
                                shape.w = 0.50
                                shape.h = z + shape.h/2.0
                                o.shapes.append(shape)
                            else:
                                o.shapes[0].w = 0.50
                                shape.h = z + shape.h/2.0
                                o.shapes[0].h = shape.h
            except Exception as e:
                pass

    def compute_dominant_color(self, rgb_image, bbox):
        xmin = int(bbox.xmin)
        ymin = int(bbox.ymin)
        h = int(bbox.height())
        w = int(bbox.width())
        cropped_image = rgb_image[ymin:ymin+h, xmin:xmin+w].copy()
        cropped_image = cv2.resize(cropped_image, (68, 68))
        np_pixels = cropped_image.shape[0] * cropped_image.shape[1]
        cropped_image = cropped_image.reshape((np_pixels, 3))
        clt = KMeans(n_clusters=K)
        labels = clt.fit_predict(cropped_image)
        label_counts = Counter(labels)
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]/255.0
        color = np.ones(4)
        color[0] = dominant_color[0]
        color[1] = dominant_color[1]
        color[2] = dominant_color[2]
        return color
