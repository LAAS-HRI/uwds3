from math import sqrt


def iou(bbox_a, bbox_b):
    """Returns the intersection over union metric"""
    xa = int(max(bbox_a.xmin, bbox_b.xmin))
    ya = int(max(bbox_a.ymin, bbox_b.ymin))
    xb = int(min(bbox_a.xmax, bbox_b.xmax))
    yb = int(min(bbox_a.ymax, bbox_b.ymax))
    intersection_area = max(0.0, xb - xa + 1.0) * max(0.0, yb - ya + 1.0)
    union_area = bbox_a.area() + bbox_b.area()
    return intersection_area / float(union_area - intersection_area)


def centroid(bbox_a, bbox_b):
    """Returns the euler distance between centroids"""
    xa = bbox_a.center().x
    xb = bbox_b.center().x
    ya = bbox_a.center().y
    yb = bbox_b.center().y
    return sqrt(pow(xa-xb, 2)+pow(ya-yb, 2))


def manhattan_centroid(bbox_a, bbox_b):
    """Returns the manhattan distance between centroids"""
    xa = bbox_a.center().x
    xb = bbox_b.center().x
    ya = bbox_a.center().y
    yb = bbox_b.center().y
    return sqrt(pow(xa-xb, 2))+sqrt(pow(ya-yb, 2))


def overlap(bbox_a, bbox_b):
    """Returns the overlap ratio"""
    xa = int(max(bbox_a.xmin, bbox_b.xmin))
    ya = int(max(bbox_a.ymin, bbox_b.ymin))
    xb = int(min(bbox_a.xmax, bbox_b.xmax))
    yb = int(min(bbox_a.ymax, bbox_b.ymax))
    intersection_area = (max(0, xb-xa+1)*max(0, yb-ya+1))
    return intersection_area / bbox_a.area()
