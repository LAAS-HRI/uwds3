import cv2
import math
import numpy as np
import rospy
from pyuwds3.types.detection import Detection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


LONGTERM_LEARNING_RATE = 1e-7
SHORTTERM_LEARNING_RATE = 0.02


class ForegroundDetector(object):
    """ Foreground detector for unknown object in interactive tabletop scenario with fixed camera
    """
    def __init__(self, max_overlap=0.3):
        """ Detector constructor
        """
        self.initialize()
        self.max_overlap = max_overlap
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("foreground_mask", Image, queue_size=1)

    def initialize(self):
        """ Initialize the detector (reset the background)
        """
        self.long_term_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=130, detectShadows=True)
        self.short_term_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)

    def detect(self, rgb_image, depth_image=None):
        """ Detect unknown objects
        """
        filtered_bbox = []
        output_dets = []

        h, w, _ = rgb_image.shape
        foreground_mask_full = np.zeros((h, w), dtype=np.uint8)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        bgr_image_cropped = bgr_image[int(h/2.0):h, 0:int(w)]
        cropped_height, cropped_width, _ = bgr_image_cropped.shape

        foreground_mask = self.long_term_detector.apply(bgr_image_cropped, learningRate=LONGTERM_LEARNING_RATE)
        foreground_mask[foreground_mask != 255] = 0 # shadows suppression
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, self.kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, self.kernel)

        motion_mask = self.short_term_detector.apply(bgr_image_cropped, learningRate=SHORTTERM_LEARNING_RATE)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel)
        motion_mask = cv2.dilate(motion_mask, self.kernel)

        not_moving_mask = cv2.bitwise_not(motion_mask)
        foreground_mask = cv2.bitwise_and(foreground_mask, not_moving_mask)
        foreground_mask_full[int(h/2.0):h, 0:int(w)] = foreground_mask

        self.pub.publish(self.bridge.cv2_to_imgmsg(foreground_mask))

        # find the contours
        contours, hierarchy = cv2.findContours(foreground_mask_full, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 10e-3 * peri, True)
            xmin, ymin, w, h = cv2.boundingRect(approx)
            if w > 20 and h > 20 and w < rgb_image.shape[1] and h < rgb_image.shape[0]:
                filtered_bbox.append(np.array([xmin, ymin, xmin+w, ymin+h]))

        filtered_bbox = self.non_max_suppression(np.array(filtered_bbox), self.max_overlap)

        for bbox in filtered_bbox:
            xmin, ymin, xmax, ymax = bbox
            w = int(xmax - xmin)
            h = int(ymax - ymin)
            x = int(xmin + w/2.0)
            y = int(ymin + h/2.0)
            if depth_image is not None:
                x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                depth = depth_image[int(y)][int(x)]/1000.0
                if math.isnan(depth) or depth == 0.0:
                    depth = None
            else:
                depth = None
            mask = foreground_mask_full[int(ymin):int(ymax), int(xmin):int(xmax)]
            output_dets.append(Detection(int(xmin), int(ymin), int(xmax), int(ymax), "thing", 1.0, mask=mask, depth=depth))

        return output_dets

    def non_max_suppression(self, boxes, max_bbox_overlap):
        """ Perform non maximum suppression
        Original code from pyimagesearch
        """
        if len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = np.argsort(area)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs, np.concatenate(
                    ([last], np.where(overlap > max_bbox_overlap)[0])))

        return boxes[pick]
