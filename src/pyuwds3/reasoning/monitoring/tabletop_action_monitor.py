import numpy as np
from ..assignment.linear_assignment import LinearAssignment
from ...utils.bbox_metrics import overlap, centroid
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
    def __init__(self, internal_simulator=None, beliefs_base=None, moving_velocity_tolerance=0.0001):
        """
        """
        super(TabletopActionMonitor, self).__init__(internal_simulator=internal_simulator, beliefs_base=beliefs_base)
        self.object_states = {}
        self.moving_velocity_tolerance = moving_velocity_tolerance
        self.centroid_assignement = LinearAssignment(centroid_cost, max_distance=None)

    def monitor(self, support_tracks, object_tracks, person_tracks, hand_tracks, time=None):
        """
        """
        moving_objects = []
        lost_objects = []
        moving_objects = []
        not_moving_objects = []

        for support in support_tracks:
            if support.is_located() and support.has_shape():
                if self.simulator.is_entity_loaded(support.id) is False:
                    self.simulator.load_node(support)
                else:
                    self.simulator.update_constraint(support.id, support.pose)
        support = support_tracks[0]

        for object in object_tracks:
            if object.is_located():
                if self.simulator.is_entity_loaded(object.id) is False:
                    self.simulator.load_node(object)
                    matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])
                    for object_indice, person_indice in matches:
                        self.mark_placed(object_tracks[object_indice], person_tracks[person_indice], support, time=time)
                else:
                    if np.allclose(object.pose.linear_velocity().to_array(), np.zeros(3), atol=self.moving_velocity_tolerance) is False:
                        moving_objects.append(object)
                    else:
                        not_moving_objects.append(object)
            if object.is_lost():
                lost_objects.append(object)

        if len(moving_objects) > 0:
            matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, moving_objects)
            for object_indice, person_indice in matches:
                self.mark_pushed(object_tracks[object_indice], person_tracks[person_indice], support, time=time)

        if len(not_moving_objects) > 0:
            matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, not_moving_objects)
            for object_indice, person_indice in matches:
                self.mark_placed(object_tracks[object_indice], person_tracks[person_indice], support, time=time)

        if len(lost_objects) > 0:
            matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, lost_objects)
            for object_indice, person_indice in matches:
                self.mark_picked(object_tracks[object_indice], person_tracks[person_indice], support, time=time)
        #
        # cleaned_relations = []
        # cleaned_index = {}
        # for relation in self.relations:
        #     if not relation.to_delete():
        #         cleaned_relations.append(relation)
        #         if relation.object is not None:
        #             cleaned_index[relation.subject.id+str(relation)+relation.object.id] = len(cleaned_relations) - 1
        #         else:
        #             cleaned_index[relation.subject.id+str(relation)] = len(cleaned_relations) - 1
        #
        # self.relations = cleaned_relations
        # self.relations_index = cleaned_index
        #
        # for object in object_tracks:
        #     if self.simulator.is_entity_loaded(object.id) is False:
        #         self.simulator.load_node(object)
        #     if object.is_confirmed():
        #         if object.is_located():
        #             if np.allclose(object.pose.linear_velocity().to_array(), np.zeros(3), atol=self.moving_velocity_tolerance) is False:
        #                 moving_objects.append(object)
        #         else:
        #             if np.allclose(object.bbox.center_filter.velocity().to_array(), np.zeros(2), atol=self.moving_velocity_tolerance) is False:
        #                 moving_objects.append(object)
        #     elif object.is_occluded():
        #         disappeared_objects.append(object)
        #
        # if len(moving_objects) > 0 and len(person_tracks) > 0:
        #     matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, moving_objects)
        #
        #     for moving_object_indice, person_indice in matches:
        #         object = moving_objects[moving_object_indice]
        #         person = person_tracks[person_indice]
        #         if object.is_located():
        #             if object.pose.linear_velocity().z > self.moving_velocity_tolerance:
        #                 self.mark_picked(object, person, support)
        #             else:
        #                 self.mark_pushed(object, person, support)
        #         else:
        #             if object.bbox.center_filter.velocity().x < - self.moving_velocity_tolerance:
        #                 self.mark_picked(object, person, support)
        #             else:
        #                 self.mark_pushed(object, person, support)
        #
        # if len(disappeared_objects) > 0:
        #     matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, disappeared_objects)
        #
        #     for disappeared_object_indice, person_indice in matches:
        #         object = moving_objects[disappeared_object_indice]
        #         person = person_tracks[person_indice]
        #         ok, object_in_front = self.test_occlusion(object, object_tracks + person_tracks + hand_tracks)
        #         if ok:
        #             self.start_predicate(object, "behind", object_in_front)
        #             # start relation behind
        #         else:
        #             # the object is not here anymore, picked ?
        #             if len(person_tracks) > 0:
        #                 action_matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])
        #
        #                 for object_indice, person_indice in matches:
        #                     object = moving_objects[moving_object_indice]
        #                     person = person_tracks[person_indice]
        #                     self.mark_picked(object, person, support)
        #             else:
        #                 # object disapeared by himself
        #                 self.trigger_event(object, "disapeared")
        #print len(self.relations)
        self.relations = [r for r in self.relations if not r.to_delete()]
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

    def mark_placed(self, object, person, support, time):
        if object.id not in self.object_states:
            # Initialize the machine state
            self.object_states[object.id] = ActionStates.PLACED
            self.trigger_event(person, "place", object, time)
            self.start_predicate(object, "on", support, time)
        current_state = self.object_states[object.id]
        if current_state == ActionStates.HELD:
            self.object_states[object.id] = ActionStates.PLACED
            self.end_predicate(person, "holding", object, time)
            self.trigger_event(person, "place", object, time)
            self.start_predicate(object, "on", support, time)

    def mark_pushed(self, object, person, support, time):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.PLACED:
                current_state = ActionStates.HELD
                self.object_states[object.id] = ActionStates.HELD
                self.start_predicate(person, "holding", object, time=time)
                self.trigger_event(person, "push", object, time=time)

    def mark_picked(self, object, person, support):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.PLACED:
                self.object_states[object.id] = ActionStates.HELD
                self.start_predicate(person, "holding", object)
                self.trigger_event(person, "pick", object)
                self.end_predicate(object, "on", support)
            elif current_state == ActionStates.HELD:
                self.end_predicate(person, "push", object)
                self.start_predicate(person, "holding", object)
                self.trigger_event(person, "pick", object)
                self.end_predicate(object, "on", support)

    def mark_released(self, object, person):
        if object.id in self.object_states:
            current_state = self.object_states[object.id]
            if current_state == ActionStates.HELD:
                self.object_states[object.id] = ActionStates.RELEASED
                self.end_predicate(person, "holding", object)
