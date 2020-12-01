#!/usr/bin/env python
#-*- coding: UTF-8 -*-

# Original code from Séverin Lemaignan

import math

INSIDE_EPSILON = 0.025 # 2.5mm
ONTOP_EPSILON = 0.02 # 2cm
ISABOVE_EPSILON = 0.06 #6cm

def bb_center(bb):

    x1,y1,z1 = bb[0]
    x2,y2,z2 = bb[1]

    return x1+x2/2, y1+y2/2, z1+z2/2


def bb_footprint(bb):
    """ Returns a rectangle that defines the bottom face of a bounding box
    """
    x1,y1,z1 = bb[0]
    x2,y2,z2 = bb[1]

    return (x1,y1), (x2,y2)


def bb_frontprint(bb):
    """ Returns a rectangle that defines the front face of a bounding box.
    """

    x1,y1,z1 = bb[0]
    x2,y2,z2 = bb[1]

    return (x1,z1), (x2,z2)


def bb_sideprint(bb):
    """ Returns a rectangle that defines the side face of a bounding box
    """
    x1,y1,z1 = bb[0]
    x2,y2,z2 = bb[1]

    return (y1,z1), (y2,z2)


def characteristic_dimension(bb):
    """ Returns the length of the bounding box diagonal
    """

    x1,y1,z1 = bb[0]
    x2,y2,z2 = bb[1]

    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))


def distance(bb1, bb2):
    """ Returns the distance between the bounding boxes centers.
    """
    x1,y1,z1 = bb_center(bb1)
    x2,y2,z2 = bb_center(bb2)

    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))


def overlap(rect1, rect2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    (l1,b1), (r1,t1) = rect1
    (l2,b2), (r2,t2) = rect2
    return range_overlap(l1, r1, l2, r2) and \
            range_overlap(b1, t1, b2, t2)


def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other

    http://codereview.stackexchange.com/questions/31352/overlapping-rectangles
    '''
    return (a_min <= b_max) and (b_min <= a_max)


def weakly_cont(rect1, rect2):
    '''Obj1 is weakly contained if the base of the object is surrounded
    by Obj2
    '''
    (l1,b1), (r1,t1) = rect1
    (l2,b2), (r2,t2) = rect2

    return (l1 >= l2) and (b1 >= b2) and (r1 <= r2) and (t1 <= t2)


def is_wkly_cont(bb1, bb2):
    '''Takes two bounding boxes and then return the value of weakly_cont
    '''
    return weakly_cont(bb_footprint(bb1), bb_footprint(bb2))


def is_lower(bb1, bb2):
    """ Returns true if obj 1 is lower than obj2.

        For obj 1 to be lower than obj 2:
         - The the top of its bounding box must be lower than the bottom
           of obj 2's bounding box
    """

    _, bb1_max = bb1
    bb2_min, _ = bb2

    x1,y1,z1 = bb1_max
    x2,y2,z2 = bb2_min

    return z1 < z2


def is_above(bb1, bb2):
    """ For obj 1 to be above obj 2:
         - the bottom of its bounding box must be higher that
           the top of obj 2's bounding box
         - the bounding box footprint of both objects must overlap
    """

    bb1_min, _ = bb1
    _, bb2_max = bb2

    x1,y1,z1 = bb1_min
    x2,y2,z2 = bb2_max
    if z1 < z2 - ISABOVE_EPSILON:
        return False

    return overlap(bb_footprint(bb1),
                   bb_footprint(bb2))


def is_below(bb1, bb2):
    """ Returns true if ob1 is below obj 2.

        For obj 1 to be below obj 2:
         - obj 1 is lower than obj 2
         - the bounding box footbrint of both objects must overlap
    """
    if is_lower(bb1, bb2):
        return overlap(bb_footprint(bb1), bb_footprint(bb2))

    return False


def is_on_top(bb1, bb2):
    """ For obj 1 to be on top of obj 2:
         - obj1 must be above obj 2
         - the bottom of obj 1 must be close to the top of obj 2
    """


    bb1_min, _ = bb1
    _, bb2_max = bb2

    x1,y1,z1 = bb1_min
    x2,y2,z2 = bb2_max
    # print bb1
    # print bb2
    return z1 < z2 + ONTOP_EPSILON and is_above(bb1, bb2)


def is_close(bb1, bb2):
    """ Returns True if the first object is close to the second.

    More precisely, returns True if the first bounding box is within a radius R
    (R = 2 X second bounding box dimension) of the second bounding box.

    Note that in general, isclose(bb1, bb2) != isclose(bb2, bb1)
    """

    dist = distance(bb1, bb2)
    dim2 = characteristic_dimension(bb2)

    return dist < 2 * dim2


def is_in(bb1, bb2):
    """ Returns True if bb1 is in bb2.

    To be 'in' bb1 is weakly contained by bb2 and the bottom of bb1 is lower
    than the top of bb2 and higher than the bottom of bb2.
    """
    bb1_min, _ = bb1
    bb2_min, bb2_max = bb2

    x1,y1,z1 = bb1_min
    x2,y2,z2 = bb2_max
    x3,y3,z3 = bb2_min
    # print z1
    # print z2
    # print z2 - INSIDE_EPSILON
    if z1 > z2 - INSIDE_EPSILON:
        return False

    if z1 < z3 + INSIDE_EPSILON:
        return False

    return weakly_cont(bb_footprint(bb1),
                       bb_footprint(bb2))

def is_included(bb1, bb2):
    """ Returns True if bb1 is included in bb2.

    To be 'in' bb1 is weakly contained by bb2 and the top of bb1 is lower
    than the top of bb2 and the bottom of bb1 is  higher than the bottom of bb2.
    """
    bb1_min, bb1_max = bb1
    bb2_min, bb2_max = bb2

    x1,y1,z1 = bb1_min
    x2,y2,z2 = bb1_max
    x3,y3,z3 = bb2_min
    x4,y4,z4 = bb2_max


    if z2 > z4+ INSIDE_EPSILON:
        return False

    if z1 < z3 - INSIDE_EPSILON:
        return False


    return weakly_cont(bb_footprint(bb1),
                       bb_footprint(bb2))
