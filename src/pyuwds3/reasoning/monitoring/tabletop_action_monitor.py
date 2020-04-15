import numpy as np
from ..assignment.linear_assignment import LinearAssignment
from ...bbox_metrics import overlap, centroid
from ...types.temporal_relation import TemporalRelation, Event
from .monitor import Monitor

OCCLUSION_TRESHOLD = 0.2


class ActionStates(object):
    PLACED = 0
    HELD = 1
    RELEASED = 2


def centroid_cost(track_a, track_b):
    """Returns the centroid cost"""
    return centroid(track_a.bbox, track_b.bbox)


def overlap_cost(track_a, track_b):
    """Returns the overlap cost"""
    return overlap(track_a.bbox, track_b.bbox)


class TabletopActionMonitor(Monitor):
    def __init__(self, simulator=None, beliefs_base=None, moving_velocity_tolerance=0.01):
        self.relations = []
        self.object_states = {}
        self.relations_index = {}
        self.moving_velocity_tolerance = moving_velocity_tolerance
        self.centroid_assignement = LinearAssignment(centroid_cost, max_distance=None)

    def monitor(self, support_tracks, object_tracks, person_tracks, hand_tracks):
        moving_objects = []
        disappeared_objects = []

        support = support_tracks[0]

        cleaned_relations = []
        cleaned_index = {}
        for relation in self.relations:
            if not relation.to_delete():
                cleaned_relations.append(relation)
                if relation.object is not None:
                    cleaned_index[relation.subject.id+str(relation)+relation.object.id] = len(cleaned_relations) - 1
                else:
                    cleaned_index[relation.subject.id+str(relation)] = len(cleaned_relations) - 1
        self.relations = cleaned_relations
        self.relations_index = cleaned_index

        for object in object_tracks:
            if object.is_confirmed():
                if object.is_located():
                    if np.allclose(object.pose.linear_velocity().to_array(), np.zeros(3), atol=self.moving_velocity_tolerance) is False:
                        moving_objects.append(object)
                else:
                    if np.allclose(object.bbox.center_filter.velocity().to_array(), np.zeros(2), atol=self.moving_velocity_tolerance) is False:
                        moving_objects.append(object)
            elif object.is_occluded():
                disappeared_objects.append(object)

        if len(moving_objects) > 0 and len(person_tracks) > 0:
            matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, moving_objects)

            for person_indice, moving_object_indice in matches:
                object = moving_objects[moving_object_indice]
                person = person_tracks[person_indice]
                if object.is_located():
                    if object.linear_velocity().z > self.moving_velocity_tolerance:
                        self.mark_picked(object, person, support)
                    else:
                        self.mark_pushed(object, person, support)
                else:
                    if object.bbox.center_filter.velocity().x < - self.moving_velocity_tolerance:
                        self.mark_picked(object, person, support)
                    else:
                        self.mark_pushed(object, person, support)

        if len(disappeared_objects) > 0:
            matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, disappeared_objects)

            for person_indice, disappeared_object_indice in matches:
                object = moving_objects[disappeared_object_indice]
                person = person_tracks[person_indice]
                ok, object_in_front = self.test_occlusion(object, object_tracks + person_tracks + hand_tracks)
                if ok:
                    self.start_situation(object, "behind", object_in_front)
                    # start relation behind
                else:
                    # the object is not here anymore, picked ?
                    if len(person_tracks) > 0:
                        action_matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])

                        for object_indice, person_indice in matches:
                            object = moving_objects[moving_object_indice]
                            person = person_tracks[person_indice]
                            self.mark_picked(object, person, support)
                    else:
                        # object disapeared by himself
                        self.trigger_event(object, "disapeared")

        return self.relations

    def test_occlusion(self, object, tracks):
        overlap = np.zeros(len(tracks))
        for idx, track in enumerate(tracks):
            overlap[idx] = overlap_cost(object, track)
        idx = np.argmax(overlap)
        object = tracks[idx]
        score = overlap[idx]
        if score > OCCLUSION_TRESHOLD:
            return True, object
        else:
            return False, None

    def mark_placed(self, object, person, support):
        if object.id not in self.object_states:
            # Initialize the machine state
            self.object_states[object.id] = ActionStates.PLACED
            self.trigger_event(person, "place", object)
        current_state = self.object_states[object.id]
        if current_state == ActionStates.HELD:
            self.object_states[object.id] = ActionStates.PLACED
            self.end_situation(person, "hold", object)
            self.trigger_event(person, "place", object)
            self.start_situation(object, "on", support)

    def mark_pushed(self, object, person, support):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.PLACED:
                self.object_states[object.id] = ActionStates.HELD
                self.start_situation(person, "hold", object)
                self.trigger_event(person, "push", object)

    def mark_picked(self, object, person, support):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.PLACED:
                self.object_states[object.id] = ActionStates.HELD
                self.start_situation(person, "hold", object)
                self.trigger_event(person, "pick", object)
                self.end_situation(object, "on", support)
            elif current_state == ActionStates.HELD:
                self.end_situation(person, "push", object)
                self.start_situation(person, "hold", object)
                self.trigger_event(person, "pick", object)
                self.end_situation(object, "on", support)

    def mark_released(self, object, person):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.HELD:
                self.object_states[object.id] = ActionStates.RELEASED
                self.end_situation(person, "hold", object)
