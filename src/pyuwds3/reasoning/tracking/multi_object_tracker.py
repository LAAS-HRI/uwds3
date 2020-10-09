import numpy as np
import rospy
from ...utils.bbox_metrics import iou, overlap, centroid
from ..assignment.linear_assignment import LinearAssignment
from scipy.spatial.distance import euclidean, cosine
from ...types.scene_node import SceneNode


def iou_cost(detection, track):
    """Returns the iou cost"""
    return 1 - iou(detection.bbox, track.bbox)


def overlap_cost(detection, track):
    """Returns the overlap cost"""
    return 1 - overlap(detection.bbox, track.bbox)


def centroid_cost(detection, track):
    """Returns the centroid cost"""
    return centroid(detection.bbox, track.bbox)


def color_cost(detection, track):
    """Returns the color cost"""
    return cosine(detection.features["color"].data,
                     track.features["color"].data)


def face_cost(detection, track):
    """Returns the face cost"""
    return euclidean(detection.features["facial_description"].data,
                     track.features["facial_description"].data)


class MultiObjectTracker(object):
    """Represents the multi object tracker"""
    def __init__(self,
                 geometric_metric,
                 features_metric,
                 max_distance_geom,
                 max_distance_feat,
                 n_init,
                 max_lost,
                 max_age,
                 p_cov_c=0.85, m_cov_c=0.003,
                 p_cov_a=1.0, m_cov_a=1e-37,
                 p_cov_h=1.0, m_cov_h=1e-37,
                 p_cov_p=0.08, m_cov_p=3,
                 p_cov_r=0.06, m_cov_r=0.001,
                 use_tracker=True):

        self.p_cov_p = p_cov_p
        self.m_cov_p = m_cov_p
        self.p_cov_r = p_cov_r
        self.m_cov_r = m_cov_r

        self.p_cov_c = p_cov_c
        self.m_cov_c = m_cov_c
        self.p_cov_a = p_cov_a
        self.m_cov_a = m_cov_a
        self.p_cov_h = p_cov_h
        self.m_cov_h = m_cov_h

        self.n_init = n_init
        self.max_lost = max_lost
        self.max_age = max_age
        self.features_metric = features_metric
        self.use_tracker = use_tracker
        self.tracks = []
        self.geometric_assignment = LinearAssignment(geometric_metric, max_distance=max_distance_geom)
        self.features_assignment = LinearAssignment(features_metric, max_distance=max_distance_feat)

    def update(self, rgb_image, detections, depth_image=None, time=None):
        """Updates the tracker"""

        not_occluded_tracks = [t for t in self.tracks if not t.is_occluded()]

        # First we try to assign the detections to the tracks by using a geometric assignment (centroid or iou)
        if len(detections) > 0:
            first_matches, unmatched_detections, unmatched_tracks = self.geometric_assignment.match(not_occluded_tracks, detections)

            # Then we try to assign the detections to the tracks that didn't match based on the features
            if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
                trks = [self.tracks[t] for t in unmatched_tracks]
                dets = [detections[d] for d in unmatched_detections]

                second_matches, remaining_detections, remaining_tracks = self.features_assignment.match(trks, dets)
                matches = list(first_matches)+list(second_matches)
            else:
                matches = first_matches
                remaining_tracks = unmatched_tracks
                remaining_detections = unmatched_detections

            for detection_indice, track_indice in matches:
                self.tracks[track_indice].update_bbox(detections[detection_indice], time=time)
                if self.use_tracker is True:
                    self.tracks[track_indice].tracker.update(rgb_image, detections[detection_indice], depth_image=depth_image)
        else:
            remaining_tracks = np.arange(len(self.tracks))
            remaining_detections = []

        for track_indice in remaining_tracks:
            self.tracks[track_indice].mark_missed()
            if self.use_tracker is True:
                if self.tracks[track_indice].is_confirmed():
                    success, detection = self.tracks[track_indice].tracker.predict(rgb_image, depth_image=depth_image)
                    if success is True:
                        self.tracks[track_indice].update_bbox(detection, detected=False, time=time)

        for detection_indice in remaining_detections:
            self.start_track(rgb_image, detections[detection_indice], depth_image=depth_image, time=time)

        self.tracks = [t for t in self.tracks if not t.to_delete()]

        return self.tracks

    def start_track(self, rgb_image, detection, depth_image=None, time=None):
        """Start to track a detection"""
        self.tracks.append(SceneNode(detection=detection,
                                     n_init=self.n_init,
                                     max_lost=self.max_lost,
                                     max_age=self.max_age,
                                     p_cov_c=self.p_cov_c, m_cov_c=self.m_cov_c,
                                     p_cov_a=self.p_cov_a, m_cov_a=self.m_cov_a,
                                     p_cov_h=self.p_cov_h, m_cov_h=self.m_cov_h,
                                     p_cov_p=self.p_cov_p, m_cov_p=self.m_cov_p,
                                     p_cov_r=self.p_cov_r, m_cov_r=self.m_cov_r,
                                     time=time))
        track_indice = len(self.tracks)-1
        if self.use_tracker is True:
            self.tracks[track_indice].start_tracker()
            self.tracks[track_indice].tracker.update(rgb_image, detection)
        return len(self.tracks)-1
