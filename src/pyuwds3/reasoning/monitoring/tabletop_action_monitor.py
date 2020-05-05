import numpy as np
from ..assignment.linear_assignment import LinearAssignment
from ...utils.bbox_metrics import overlap, centroid
from .monitor import Monitor
from scipy.spatial.distance import euclidean


OCCLUSION_THRESHOLD = 0.8
VELOCITY_THRESHOLD = 0.008
POSITION_TOLERANCE = 0.06


class ActionStates(object):
    """ The object states
    """
    PLACED = 0
    HELD = 1


def centroid_cost(track_a, track_b):
    """Returns the centroid cost"""
    return centroid(track_a.bbox, track_b.bbox)


def overlap_cost(track_a, track_b):
    """Returns the overlap cost"""
    return overlap(track_a.bbox, track_b.bbox)


class TabletopActionMonitor(Monitor):
    """ Special monitor for tabletop scenario
    """
    def __init__(self, internal_simulator=None, beliefs_base=None):
        """ Tabletop monitor constructor
        """
        super(TabletopActionMonitor, self).__init__(internal_simulator=internal_simulator, beliefs_base=beliefs_base)
        self.object_states = {}
        self.centroid_assignement = LinearAssignment(centroid_cost, max_distance=None)

    def monitor(self, support_tracks, object_tracks, person_tracks, hand_tracks, time=None):
        """ Monitor the physical consistency of the objects and detect human tabletop actions
        """
        self.cleanup_relations()
        if len(support_tracks) > 0:

            next_object_states = {}
            object_tracks_map = {}
            corrected_object_tracks = []
            for support in support_tracks:
                if support.is_located() and support.has_shape():
                    if not self.simulator.is_entity_loaded(support.id):
                        self.simulator.load_node(support, static=True)

            for object in object_tracks:
                if object.is_located() and object.has_shape():
                    if object.is_confirmed():
                        if not self.simulator.is_entity_loaded(object.id):
                            self.simulator.load_node(object)
                            simulated_object = object
                        else:
                            simulated_object = self.simulator.get_entity(object.id)

                        object_tracks_map[object.id] = object
                        # compute scene node input
                        simulated_position = simulated_object.pose.position()
                        perceived_position = object.pose.position()
                        corrected_object_tracks.append(simulated_object)

                        distance = euclidean(simulated_position.to_array(), perceived_position.to_array())

                        is_consistent = distance < POSITION_TOLERANCE
                        if object.id in self.object_states:
                            if self.object_states[object.id] == ActionStates.HELD:
                                is_consistent = distance < 0.02

                        is_perceived_object_moving = not np.allclose(object.pose.linear_velocity().to_array(), np.zeros(3), atol=VELOCITY_THRESHOLD)

                        # compute next state
                        if is_consistent:
                            distance_to_support, support = self.simulator.is_on_support(object)
                            if distance_to_support > POSITION_TOLERANCE or is_perceived_object_moving:
                                next_object_states[object.id] = ActionStates.HELD
                            else:
                                next_object_states[object.id] = ActionStates.PLACED
                        else:
                            next_object_states[object.id] = ActionStates.HELD

            for object_id in self.object_states.keys():
                if object_id not in next_object_states:
                    self.assign_and_trigger_action(object, "release", person_tracks, time)
                elif self.object_states[object_id] == ActionStates.HELD and \
                        next_object_states[object_id] == ActionStates.PLACED:
                    self.assign_and_trigger_action(object, "place", person_tracks, time)
                elif self.object_states[object_id] == ActionStates.PLACED and \
                        next_object_states[object_id] == ActionStates.HELD:
                    self.assign_and_trigger_action(object, "pick", person_tracks, time)
                else:
                    pass

                if self.object_states[object_id] == ActionStates.PLACED:
                    self.simulator.remove_constraint(object_id)

                if self.object_states[object_id] == ActionStates.HELD:
                    if object_id in object_tracks_map:
                        object = object_tracks_map[object_id]
                        self.simulator.update_constraint(object_id, object.pose)

            self.object_states = next_object_states

        return corrected_object_tracks, self.relations

    def assign_and_trigger_action(self, object, action, person_tracks, time):
        """
        """
        matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])
        if len(matches > 0):
            _, person_indice = matches[0]
            person = person_tracks[person_indice]
        else:
            return False

        self.trigger_event(person, action, object, time)

    def test_occlusion(self, object, tracks):
        """ Test occlusion with 2D bbox overlap
        """
        overlap = np.zeros(len(tracks))
        for idx, track in enumerate(tracks):
            overlap[idx] = overlap_cost(object, track)
        idx = np.argmax(overlap)
        object = tracks[idx]
        score = overlap[idx]
        if score > OCCLUSION_THRESHOLD:
            return True, object
        else:
            return False, None
