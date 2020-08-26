import rospy
import cv2
from .monitor import Monitor


class PerspectiveMonitor(Monitor):
    """
    """
    def __init__(self, internal_simulator, beliefs_base, rendering_ratio=(1/10.0)):
        """
        """
        super(PerspectiveMonitor, self).__init__(internal_simulator=internal_simulator,
                                                 beliefs_base=beliefs_base)
        self.rendering_ratio = rendering_ratio
        self.other_perspective = None

    def monitor_myself(self, tracks, view_pose, camera):
        """
        """
        rgb_image, _, _, visible_tracks = self.simulator.get_camera_view(view_pose, camera, prior_tracks=tracks, rendering_ratio=self.rendering_ratio)
        return rgb_image, visible_tracks, self.relations

    def monitor_others(self, face_tracks):
        """
        """
        visibles_tracks = []
        person_camera = None
        rgb_image = None
        min_depth = 1000.0
        for t in face_tracks:
            if t.is_confirmed():
                if t.has_camera() is True and t.is_located() is True:
                    if t.bbox.depth < min_depth:
                        min_depth = t.bbox.depth
                        person_camera = t

        if person_camera is not None:
            rgb_image, depth_image, mask_image, visibles_tracks = self.simulator.get_camera_view(person_camera.pose, person_camera.camera, rendering_ratio=self.rendering_ratio)
            success = True
        else:
            success = False


        return success, rgb_image, visibles_tracks, self.relations
