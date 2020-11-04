import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from ...utils.bbox_metrics import overlap
from ...utils.egocentric_spatial_relations import is_right_of, is_left_of, is_behind


def overlap_cost(track_a, track_b):
    """Returns the overlap cost"""
    return 1 - overlap(track_a.bbox, track_b.bbox)


class ObjectState(object):
    NOT_VISIBLE = 1
    VISIBLE = 0


class HumanPerspectiveMonitor(Monitor):
    """ """
    def __init__(self, internal_simulator, rendering_ratio=(1/4.0)):
        """ """
        super(HumanPerspectiveMonitor, self).__init__(internal_simulator=internal_simulator)
        self.rendering_ratio = rendering_ratio
        self.other_perspective = None

        self.previous_object_states = {}
        self.visible_tracks_map = {}
        self.previously_visible_tracks_map = {}

        self.monitored_face = None
        self.monitored_person = None

        self.overlap_assignement = LinearAssignment(overlap_cost, max_distance=0.9)

    def monitor(self, face_tracks, person_tracks, time):
        """ """
        self.cleanup_relations()

        next_object_states = {}
        visible_tracks_map = {}
        success = False
        assign_body = False

        closest_face = None
        rgb_image = None
        min_depth = 1000.0
        if len(face_tracks) > 1:
            for t in face_tracks:
                if t.is_confirmed():
                    if t.has_camera() is True and t.is_located() is True:
                        if t.bbox.depth < min_depth:
                            min_depth = t.bbox.depth
                            closest_face = t
        else:
            if len(face_tracks) == 1:
                closest_face = face_tracks[0]

        if closest_face is not None:
            if self.monitored_face is None:
                # new face monitored
                assign_body = True
            else:
                if closest_face.id != self.monitored_face.id:
                    assign_body = True

            if assign_body is True:
                #rospy.logwarn("new face {} monitored".format(closest_face.id))
                if len(person_tracks) > 0:
                    matches, unmatched_objects, unmatched_person = self.overlap_assignement.match(person_tracks, [closest_face])
                    if len(matches > 0):
                        #rospy.logwarn("assign body to face")
                        _, person_indice = matches[0]
                        person = person_tracks[person_indice]
                        self.monitored_person = person
                    else:
                        #rospy.logwarn("cannot assign body to face")
                        closest_face = None

            if closest_face is not None:
                visible_tracks = []
                visible_tracks_map = {}
                rgb_image, depth_image, mask_image, visible_tracks = self.simulator.get_camera_view(closest_face.pose, closest_face.camera, rendering_ratio=self.rendering_ratio)
                success = True
                for track in visible_tracks:
                    next_object_states[track.id] = ObjectState.VISIBLE
                    visible_tracks_map[track.id] = track
                    if track.id not in self.previous_object_states.keys():
                        #rospy.logwarn("start {} is visible by {}".format(track.description, self.monitored_person.description))
                        self.start_fact(track, "visible_by", object=self.monitored_person, time=time)
                self.visible_tracks_map = visible_tracks_map

        for track_id in self.previous_object_states.keys():
            if track_id in self.previously_visible_tracks_map:
                track = self.previously_visible_tracks_map[track_id]
                if track_id not in next_object_states:
                    self.end_fact(track, "visible_by", object=self.monitored_person, time=time)
                    #rospy.logwarn("end {} is visible by {}".format(track.description, self.monitored_person.description))

        self.compute_egocentric_relations(self.visible_tracks_map.values(), time)

        self.monitored_face = closest_face
        self.previous_object_states = next_object_states

        self.previously_visible_tracks_map = visible_tracks_map

        return success, rgb_image, self.visible_tracks_map.values(), self.relations

    def compute_egocentric_relations(self, object_tracks, time):
        """ """
        for obj1 in object_tracks:
            for obj2 in object_tracks:
                if obj1 != obj2:
                    if obj1.is_perceived() and obj2.is_perceived():
                        if is_right_of(obj1.bbox.to_array(), obj2.bbox.to_array()):
                            self.start_fact(obj1, "right_of", object=obj2, time=time)
                        else:
                            self.end_fact(obj1, "right_of", object=obj2, time=time)

                        if is_left_of(obj1.bbox.to_array(), obj2.bbox.to_array()):
                            self.start_fact(obj1, "left_of", object=obj2, time=time)
                        else:
                            self.end_fact(obj1, "left_of", object=obj2, time=time)

                        if is_behind(obj1.bbox.to_array(), obj2.bbox.to_array()):
                            self.start_fact(obj1, "behind", object=obj2, time=time)
                        else:
                            self.end_fact(obj1, "behind", object=obj2, time=time)

        for obj1 in self.previously_visible_tracks_map.values():
            for obj2 in self.previously_visible_tracks_map.values():
                if obj1 != obj2:
                    if obj1.id not in self.visible_tracks_map or obj2.id not in self.visible_tracks_map:
                        self.end_fact(obj1, "right_of", object=obj2, time=time)
                        self.end_fact(obj1, "left_of", object=obj2, time=time)
                        self.end_fact(obj1, "behind", object=obj2, time=time)
