
import math
from scipy.spatial.distance import euclidean
from ..types.bbox import BoundingBox


def is_left_of(bb1, bb2):
    _, _, bb1_max, _, _ = bb1
    bb2_min, _, _, _, _ = bb2
    return bb1_max < bb2_min


def is_right_of(bb1, bb2):
    bb1_min, _, _, _, _ = bb1
    _, _, bb2_max, _, _ = bb2
    return bb1_min > bb2_max


def is_behind(bb1, bb2):
    _, _, _, _, bb1_depth = bb1
    _, _, _, _, bb2_depth = bb2
    return bb1_depth > bb2_depth
