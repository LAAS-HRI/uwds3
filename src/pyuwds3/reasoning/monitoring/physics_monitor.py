import numpy as np
import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from scipy.spatial.distance import euclidean


INF = 10e3


class ActionStates(object):
    """ The object states
    """
    PLACED = 0
    HELD = 1
    RELEASED = 2


def centroid_cost(track_a, track_b):
    """Returns the centroid cost"""
    try:
        return euclidean(track_a.pose.position().to_array(), track_b.pose.position().to_array())
    except Exception:
        return INF


class PhysicsMonitor(Monitor):
    """ Special monitor for tabletop scenario
    """
    def __init__(self, internal_simulator=None, beliefs_base=None, placement_tolerance=0.1, holding_tolerance=0.08):
        """ Tabletop monitor constructor
        """
        super(PhysicsMonitor, self).__init__(internal_simulator=internal_simulator, beliefs_base=beliefs_base)
        self.previous_object_states = {}
        self.previous_object_tracks_map = {}
        self.placement_tolerance = placement_tolerance
        self.holding_tolerance = holding_tolerance
        self.centroid_assignement = LinearAssignment(centroid_cost, max_distance=None)
        self.sim_last_update = None

    def monitor(self, object_tracks, person_tracks, time=None):
        """ Monitor the physical consistency of the objects and detect human tabletop actions
        """
        self.cleanup_relations()

        next_object_states = {}
        object_tracks_map = {}
        corrected_object_tracks = []

        for object in object_tracks:
            if not self.simulator.is_entity_loaded(object.id):
                self.simulator.load_node(object)
        # perform prediction
        self.generate_prediction()

        for object in object_tracks:
            if object.is_located() and object.has_shape():
                if object.is_confirmed():
                    simulated_object = self.simulator.get_entity(object.id)

                    object_tracks_map[object.id] = object
                    # compute scene node input
                    simulated_position = simulated_object.pose.position()
                    perceived_position = object.pose.position()

                    distance = euclidean(simulated_position.to_array(), perceived_position.to_array())

                    is_physically_plausible = distance < self.placement_tolerance
                    if object.id in self.previous_object_states:
                        if self.previous_object_states[object.id] == ActionStates.HELD:
                            is_physically_plausible = distance < self.holding_tolerance

                    # compute next state
                    if is_physically_plausible:
                        next_object_states[object.id] = ActionStates.PLACED
                    else:
                        next_object_states[object.id] = ActionStates.HELD
                        self.simulator.reset_entity_pose(object.id, object.pose)

        for object_id in self.previous_object_states.keys():
            if object_id in self.previous_object_tracks_map:
                object = self.previous_object_tracks_map[object_id]
                if object_id not in next_object_states:
                    if self.previous_object_states[object_id] == ActionStates.HELD:
                        next_object_states[object_id] = ActionStates.RELEASED
                        self.assign_and_trigger_action(object, "release", person_tracks, time)
                    else:
                        next_object_states[object_id] = self.previous_object_states[object_id]
                elif self.previous_object_states[object_id] == ActionStates.HELD and \
                        next_object_states[object_id] == ActionStates.PLACED:
                    self.assign_and_trigger_action(object, "place", person_tracks, time)
                elif self.previous_object_states[object_id] == ActionStates.RELEASED and \
                        next_object_states[object_id] == ActionStates.PLACED:
                    self.assign_and_trigger_action(object, "place", person_tracks, time)
                elif self.previous_object_states[object_id] == ActionStates.PLACED and \
                        next_object_states[object_id] == ActionStates.HELD:
                    self.assign_and_trigger_action(object, "pick", person_tracks, time)
                elif self.previous_object_states[object_id] == ActionStates.RELEASED and \
                        next_object_states[object_id] == ActionStates.HELD:
                    self.assign_and_trigger_action(object, "pick", person_tracks, time)
                else:
                    pass

        corrected_object_tracks = self.simulator.get_not_static_entities()
        self.previous_object_states = next_object_states
        self.previous_object_tracks_map = object_tracks_map

        return corrected_object_tracks, self.relations

    def generate_prediction(self):
        """ Perform physics prediction"""
        if self.sim_last_update is None:
            self.simulator.step_simulation()
            self.sim_last_update = cv2.getTickCount()
        else:
            now = cv2.getTickCount()
            elapsed_time = (now-self.sim_last_update) / cv2.getTickFrequency()
            nb_step = int(elapsed_time/(1/240.0))
            for i in range(0, nb_step):
                self.simulator.step_simulation()
            self.sim_last_update = now

    def assign_and_trigger_action(self, object, action, person_tracks, time):
        """ Assign the action to the closest person of the given object and trigger it """
        matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])
        if len(matches > 0):
            _, person_indice = matches[0]
            person = person_tracks[person_indice]
            self.trigger_event(person, action, object, time)
        # else:
        #     self.trigger_event(object, "moved by himself")
    #
    # def test_occlusion(self, object, tracks):
    #     """ Test occlusion with 2D bbox overlap
    #     """
    #     overlap_score = np.zeros(len(tracks))
    #     for idx, track in enumerate(tracks):
    #         overlap_score[idx] = overlap(object, track)
    #     idx = np.argmax(overlap_score)
    #     object = tracks[idx]
    #     score = overlap[idx]
    #     if score > OCCLUSION_THRESHOLD:
    #         return True, object
    #     else:
    #         return False, None
    #
    # def test_support(self, object):
    #     """ Test if an object lie on a support using raycasting
    #     """
    #     if object.has_shape() and object.is_located():
    #         shape = object.shapes[0]
    #         if shape.is_box():
    #             z_offset = shape.height()/2.0
    #         elif shape.is_sphere():
    #             z_offset = shape.radius()
    #         elif shape.is_cylinder():
    #             z_offset = shape.height()/2.0
    #         else:
    #             return None, None, None
    #
    #         ray_start = object.pose.position()
    #         ray_start.z -= z_offset
    #         ray_end = object.pose.position()
    #         ray_end.z = 0.0 # set to ground
    #         hited, dist, hit_object = self.simulator.test_raycast(ray_start, ray_end)
    #         if hited is True:
    #             return True, dist, hit_object
    #         else:
    #             dist = ray_start.z
    #         return True, dist, None
    #     return False, None, None
    #
    # def test_containment(self, object):
    #     """ """
    #     if object.has_shape() and object.is_located():
    #         shape = object.shapes[0]
    #         if shape.is_box():
    #             z_offset = shape.height()/2.0
    #         elif shape.is_sphere():
    #             z_offset = shape.radius()
    #         elif shape.is_cylinder():
    #             z_offset = shape.height()/2.0
    #         else:
    #             return None, None
    #
    #     return False, None
