import cv2
import rospy
from ...types.shape.sphere import Sphere
from ...types.shape.box import Box
from ...types.shape.cylinder import Cylinder
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

K = 3


class ShapeEstimator(object):
    """ """
    def estimate(self, rgb_image, objects_tracks, camera):
        """ """
        for o in objects_tracks:
            try:
                if o.is_confirmed() and o.bbox.height() > 0:
                    if o.bbox.depth is not None:
                        if o.label != "person":
                            if not o.has_shape():
                                shape = self.compute_cylinder_from_bbox(o.bbox, camera)
                                if o.label == "face" or o.label == "hand":
                                    shape = self.compute_sphere_from_bbox(o.bbox, camera)
                                shape.pose.pos.x = .0
                                shape.pose.pos.y = .0
                                shape.pose.pos.z = .0
                                shape.color = self.compute_dominant_color(rgb_image, o.bbox)
                                o.shapes.append(shape)
                        else:
                            shape = self.compute_cylinder_from_bbox(o.bbox, camera)
                            z = o.pose.pos.z
                            shape.pose.pos.x = .0
                            shape.pose.pos.y = .0
                            shape.pose.pos.z = -(z - shape.h/2.0)/2.0
                            if not o.has_shape():
                                shape.color = [0, 200, 0, 1]
                                shape.w = 0.50
                                shape.h = z + shape.h/2.0
                                o.shapes.append(shape)
                            else:
                                o.shapes[0].w = 0.50
                                shape.h = z + shape.h/2.0
                                o.shapes[0].h = shape.h
            except Exception as e:
                rospy.logwarn(e)

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
        color[3] = 1.0
        return color

    def compute_cylinder_from_bbox(self, bbox, camera):
        camera_matrix = camera.camera_matrix()
        z = bbox.depth
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        w = bbox.width()
        h = bbox.height()
        w = w * z / fx
        h = h * z / fy
        return Cylinder(w, h)

    def compute_sphere_from_bbox(self, bbox, camera):
        camera_matrix = camera.camera_matrix()
        z = bbox.depth
        fx = camera_matrix[0][0]
        fy = camera_matrix[1][1]
        w = bbox.width()
        h = bbox.height()
        w = w * z / fx
        h = h * z / fy
        d = max(w, h)
        return Sphere(d)
