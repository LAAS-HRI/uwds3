import rospy
from .monitor import Monitor
from ...utils.egocentric_spatial_relations import is_right_of, is_left_of, is_behind

class RobotPerspectiveMonitor(Monitor):
    def __init__(self):
        super(RobotPerspectiveMonitor, self).__init__(internal_simulator=internal_simulator)

    def monitor(self, object_tracks, robot_camera, view_pose):
        """ """
        self.cleanup_relations()

        #TODO 
