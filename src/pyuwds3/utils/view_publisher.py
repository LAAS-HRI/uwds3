import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading


class ViewPublisher(object):
    def __init__(self, topic_name, queue_size=1):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(topic_name, Image, queue_size=queue_size)

    def publish(self,
                rgb_image,
                tracks,
                time,
                view_pose=None,
                camera=None,
                events=[],
                overlay_image=None,
                fps=None):
        myThread(self.bridge,
                 self.publisher,
                 rgb_image,
                 tracks,
                 time,
                 view_pose=view_pose,
                 camera=camera,
                 events=events,
                 overlay_image=overlay_image,
                 fps=fps).start()


class myThread(threading.Thread):
    def __init__(self,
                 bridge,
                 publisher,
                 rgb_image,
                 tracks,
                 time,
                 view_pose=None,
                 camera=None,
                 events=[],
                 overlay_image=None,
                 fps=None):
        threading.Thread.__init__(self)
        self.bridge = bridge
        self.publisher = publisher
        self.rgb_image = rgb_image
        self.tracks = tracks
        self.time = time
        self.view_pose = view_pose
        self.camera = camera
        self.events = events
        self.overlay_image = overlay_image
        self.fps = fps

    def run(self):
        bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        if self.overlay_image is not None:
            bgr_overlay = cv2.cvtColor(self.overlay_image, cv2.COLOR_RGB2BGR)
            bgr_image = cv2.addWeighted(bgr_image, 1.0, bgr_overlay, 0.3, 0)
        cv2.rectangle(bgr_image, (0, 0), (250, 40), (200, 200, 200), -1)
        fps_str = "FPS: {:0.1f}hz".format(self.fps) if self.fps is not None else "FPS: Unknown"
        cv2.putText(bgr_image, "Nb tracks : {}".format(len(self.tracks)), (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(bgr_image, fps_str, (5, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for track in self.tracks:
            track.draw(bgr_image, (230, 0, 120, 125), 1, view_pose=self.view_pose, camera=self.camera)
        for i, event in enumerate(self.events):
            cv2.putText(bgr_image, "{}".format(event.description), (20, 60+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,200), 2)
        msg = self.bridge.cv2_to_imgmsg(bgr_image, "bgr8")
        msg.header.stamp = self.time
        self.publisher.publish(msg)
