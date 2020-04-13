import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


class LinearAssignment(object):
    """ """
    def __init__(self, cost_metric, max_distance=None):
        """ """
        self.cost_metric = cost_metric
        self.max_distance = max_distance

    def match(self, tracks, detections):
        """ """

        if(len(tracks) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        # Create the cost matrix
        C = np.zeros((len(detections), len(tracks)), dtype=np.float32)

        # Compute the cost matrix
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                C[d, t] = self.cost_metric(det, trk)

        # Run the optimization problem
        M = linear_assignment(C)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in M[:, 0]):
                unmatched_detections.append(d)
        unmatched_tracks = []
        for t, trk in enumerate(tracks):
            if(t not in M[:, 1]):
                unmatched_tracks.append(t)

        matches = []
        for m in M:
            if self.max_distance is None:
                matches.append(m.reshape(1, 2))
            else:
                if(C[m[0], m[1]] > self.max_distance):
                    unmatched_detections.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m.reshape(1, 2))

        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_tracks)
