import rospy
import cv2
from .monitor import Monitor


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
        self.previous_object_tracks_map = {}
    #
    # def monitor_myself(self, all_tracks, view_pose, camera):
    #     """ """
    #     rgb_image, _, _, visible_tracks = self.simulator.get_camera_view(view_pose, camera, prior_tracks=tracks, rendering_ratio=self.rendering_ratio)
    #     return rgb_image, visible_tracks, self.relations

    def monitor(self, face_tracks, person_tracks):
        """ """
        self.cleanup_relations()

        next_object_states = {}
        object_tracks_map = {}

        visibles_tracks = []
        closest_person_camera = None
        rgb_image = None
        min_depth = 1000.0
        for t in face_tracks:
            if t.is_confirmed():
                if t.has_camera() is True and t.is_located() is True:
                    if t.bbox.depth < min_depth:
                        min_depth = t.bbox.depth
                        closest_person_camera = t

        if closest_person_camera is not None:
            rgb_image, depth_image, mask_image, visibles_tracks = self.simulator.get_camera_view(person_camera.pose, person_camera.camera, rendering_ratio=self.rendering_ratio)

            for track in visibles_tracks:
                next_object_states[track.id] = ObjectState.VISIBLE

        for track_id in self.previous_object_states.keys():
            if track_id in self.previous_object_tracks_map:
                track = self.previous_object_tracks_map[track_id]
                if track_id not in next_object_states:
                    if self.previous_object_states == ObjectState.VISIBLE:
                        next_object_states[track_id] = ObjectState.NOT_VISIBLE
                    else:
                        next_object_states[]

        self.previous_object_states = next_object_states
        self.previous_object_tracks_map = object_tracks_map


        return rgb_image, visibles_tracks, self.relations
