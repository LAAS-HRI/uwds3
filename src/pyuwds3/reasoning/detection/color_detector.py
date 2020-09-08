import cv2
import math
import numpy as np
import rospy
from pyuwds3.types.detection import Detection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ColorDetector(object):
    def __init__(self, debug_topics=True, color="red"):
        """ """
        color_list = ["red", "blue", "green"]

        self.bridge = CvBridge()

        if color not in color_list:
            raise ValueError("color parameter should be one of: "+str(color_list))

        if color == "red":
            self.lower = np.array([136, 87, 111], np.uint8)
            self.upper = np.array([180, 255, 255], np.uint8)
        elif color == "blue":
            self.lower = np.array([107, 140, 100], np.uint8)
            self.upper = np.array([130, 255, 255], np.uint8)
        elif color == "green":
            self.lower = np.array([25, 100, 100], np.uint8)
            self.upper = np.array([102, 255, 255], np.uint8)
        self.color = color
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

        self.debug_topics = debug_topics
        if self.debug_topics is True:
            self.pub = rospy.Publisher("color_mask", Image, queue_size=1)

    def detect(self, rgb_image, depth_image=None):
        """ Detect an object based on the color """
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

        lower = np.array([136, 87, 111], np.uint8)
        upper = np.array([180, 255, 255], np.uint8)

        color_mask = cv2.inRange(hsv_image, self.lower, self.upper)

        color_mask = cv2.dilate(color_mask, self.kernel)

        if self.debug_topics is True:
            self.pub.publish(self.bridge.cv2_to_imgmsg(color_mask))

        # find the bbox
        x, y, w, h = cv2.boundingRect(color_mask)
        xmin = x
        ymin = y
        xmax = xmin+w
        ymax = ymin+h

        if depth_image is not None:
            x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
            y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
            depth = depth_image[int(y)][int(x)]/1000.0
            if math.isnan(depth) or depth == 0.0:
                depth = None
        else:
            depth = None
        mask = color_mask[int(ymin):int(ymax), int(xmin):int(xmax)]

        object_detection = Detection(int(xmin),
                                     int(ymin),
                                     int(xmin+w),
                                     int(ymin+h),
                                     self.color+"_thing",
                                     1.0,
                                     mask=mask,
                                     depth=depth)

        if object_detection.bbox.area() > 100:
            return [object_detection]
        else:
            return []
