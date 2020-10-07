
import math
from scipy.spatial.distance import euclidean
from ..types.bbox import BoundingBox


def characteristic_dimension(bb):
    xmin, ymin, xmax, ymax, _ = bb
    w = xmax - xmin
    h = ymax - ymin
    return math.sqrt(pow(w, 2) + pow(h, 2))


def is_left_of(bb1, bb2):
    _, _, bb1_max, _, _ = bb1
    bb2_min, _, _, _, _ = bb2
    margin = characteristic_dimension(bb1)
    return bb1_max + margin < bb2_min


def is_right_of(bb1, bb2):
    bb1_min, _, _, _, _ = bb1
    _, _, bb2_max, _, _ = bb2
    margin = characteristic_dimension(bb1)
    return bb1_min - margin > bb2_max


def is_behind(bb1, bb2):
    _, _, _, _, bb1_depth = bb1
    _, _, _, _, bb2_depth = bb2
    return bb1_depth > bb2_depth
