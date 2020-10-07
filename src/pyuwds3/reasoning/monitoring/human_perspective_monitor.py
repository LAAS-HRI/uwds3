import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from ...utils.bbox_metrics import overlap
# from ...utils.egocentric_spatial_relations import is_right_of, is_left_of


def overlap_cost(track_a, track_b):
    """Returns the overlap cost"""
    return 1 - overlap(track_a.bbox, track_b.bbox)


class ObjectState(object):
    NOT_VISIBLE = 1
    VISIBLE = 0


class HumanPerspectiveMonitor(Monitor):
    """ """
    def __init__(self, internal_simulator, beliefs_base, rendering_ratio=(1/10.0)):
        """ """
        super(HumanPerspectiveMonitor, self).__init__(internal_simulator=internal_simulator,
                                                      beliefs_base=beliefs_base)
        self.rendering_ratio = rendering_ratio
        self.other_perspective = None

        self.previous_object_states = {}
        self.previous_previously_visible_tracks_map = {}

        self.monitored_face = None
        self.monitored_person = None

        self.overlap_assignement = LinearAssignment(overlap_cost, max_distance=0.9)

    def monitor(self, face_tracks, person_tracks, time):
        """ """
        self.cleanup_relations()

        next_object_states = {}
        previously_visible_tracks_map = {}

        visibles_tracks = []
        closest_face = None
        rgb_image = None
        min_depth = 1000.0
        for t in face_tracks:
            if t.is_confirmed():
                if t.has_camera() is True and t.is_located() is True:
                    if t.bbox.depth < min_depth:
                        min_depth = t.bbox.depth
                        closest_face = t

        if closest_face != self.monitored_face:
            # new face monitored
            if len(person_tracks) > 0:
                matches, unmatched_objects, unmatched_person = self.overlap_assignement.match(person_tracks, [closest_face])
                if len(matches > 0):
                    _, person_indice = matches[0]
                    person = person_tracks[person_indice]
                    self.monitored_person = person
                    self.monitored_face = closest_face
        else:
            if closest_face is not None:
                rgb_image, depth_image, mask_image, visibles_tracks = self.simulator.get_camera_view(closest_face.pose, closest_face.camera, rendering_ratio=self.rendering_ratio)

                for track in visibles_tracks:
                    next_object_states[track.id] = ObjectState.VISIBLE
                    previously_visible_tracks_map[track.id] = track
                    if track.id not in self.previous_object_states.keys():
                        self.start_fact(track, "visible_by", object=self.monitored_person, time=time)

        for track_id in self.previous_object_states.keys():
            if track_id in self.previous_previously_visible_tracks_map:
                track = self.previous_previously_visible_tracks_map[track_id]
                if track_id not in next_object_states:
                    self.end_fact(object, "visible_by", object=self.monitored_person, time=time)

        self.previous_object_states = next_object_states
        self.previous_previously_visible_tracks_map = previously_visible_tracks_map

        return visibles_tracks, self.relations
