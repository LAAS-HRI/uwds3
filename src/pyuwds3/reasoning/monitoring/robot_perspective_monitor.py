import rospy
from .monitor import Monitor
from ...utils.egocentric_spatial_relations import is_right_of, is_left_of, is_behind

class RobotPerspectiveMonitor(Monitor):
    def __init__(self, internal_simulator):
        super(RobotPerspectiveMonitor, self).__init__(internal_simulator=internal_simulator)

        self.previously_visible_tracks_map = {}

    def monitor(self, object_tracks, person_tracks, robot_camera, view_pose, time):
        """ """
        self.cleanup_relations()

        visibles_tracks_map = {}

        for obj1 in object_tracks + person_tracks:
            visibles_tracks_map[obj1.id] = obj1
            for obj2 in object_tracks + person_tracks:
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

        for obj1 in self.previously_visible_tracks_map:
            for obj2 in self.previously_visible_tracks_map:
                if obj1 != obj2:
                    if obj1 not in visibles_tracks_map or obj2 not in visibles_tracks_map:
                        self.end_fact(obj1, "right_of", object=obj2, time=time)
                        self.end_fact(obj1, "left_of", object=obj2, time=time)
                        self.end_fact(obj1, "behind", object=obj2, time=time)

        self.previous_previously_visible_tracks_map = visibles_tracks_map

        return object_tracks + person_tracks, self.relations
