import cv2
import math
import numpy as np
import rospy
from pyuwds3.types.detection import Detection
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DetectorState(object):
    INIT = 0
    WAITING = 1
    READY = 2
    RUNNING = 3

    state = {0: "INIT", 1: "WAITING", 2: "READY", 3: "RUNNING"}


class ForegroundDetector(object):
    def __init__(self, interactive_mode=True):
        self.interactive_mode = interactive_mode
        self.roi_points = []
        self.state = DetectorState.INIT
        self.background_substraction = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=150, detectShadows=True)

        if self.interactive_mode is True:
            cv2.namedWindow("select_roi")
            cv2.setMouseCallback("select_roi", self.click_and_select)
        else:
            self.state = DetectorState.RUNNING

        self.bridge = CvBridge()
        self.pub = rospy.Publisher("test", Image, queue_size=1)

    def detect(self, rgb_image, depth_image=None, roi_points=[], prior_detections=[]):
        filtered_bbox = []
        output_dets = []

        h, w, _ = rgb_image.shape
        foreground_mask = np.zeros((h, w), dtype=np.uint8)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        #bgr_image_resized = cv2.resize(bgr_image, (w/2.0, h/2.0))

        foreground_mask[int(h/2.0):h, 0:int(w)] = self.background_substraction.apply(bgr_image[int(h/2.0):h, 0:int(w)], learningRate=10e-7)
        foreground_mask[foreground_mask != 255] = 0 # shadows suppression

        self.pub.publish(self.bridge.cv2_to_imgmsg(foreground_mask))

        for d in prior_detections:
            x = int(d.bbox.xmin)
            x = x - 5 if x > 5 else x
            y = int(d.bbox.ymin)
            y = y - 5 if y > 5 else y
            w = int(d.bbox.width())
            w = w + 5 if w + 5 < rgb_image.shape[1] else w
            h = int(d.bbox.height())
            h = h + 5 if h + 5 < rgb_image.shape[0] else h
            foreground_mask[y:y+h, x:x+w] = 0
        # remove the noise of the mask
        kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel_big)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_big)

        if len(self.roi_points) == 2:
            roi_mask = np.full(foreground_mask.shape, 255, dtype="uint8")
            roi_mask[self.roi_points[0][1]:self.roi_points[1][1], self.roi_points[0][0]:self.roi_points[1][0]] = 0
            opening -= roi_mask

        opening[opening != 255] = 0
        # find the contours
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 10e-3 * peri, True)
            xmin, ymin, w, h = cv2.boundingRect(approx)
            if w > 10 and h > 10 and w < rgb_image.shape[1] and h < rgb_image.shape[0]:
                filtered_bbox.append(np.array([xmin, ymin, xmin+w, ymin+h]))

        if self.interactive_mode is True:
            debug_image = cv2.cvtColor(opening.copy(), cv2.COLOR_GRAY2BGR)
            if len(self.roi_points) == 1:
                opening = cv2.rectangle(debug_image, self.roi_points[0], self.roi_points[0], (0, 255, 0), 3)
            elif len(self.roi_points) == 2:
                opening = cv2.rectangle(debug_image, self.roi_points[0], self.roi_points[1], (0, 255, 0), 3)
            cv2.rectangle(debug_image, (0, 0), (300, 40), (200, 200, 200), -1)
            cv2.putText(debug_image, "Detector state : {}".format(DetectorState.state[self.state]), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(debug_image, "Select ROI & press 'r' to start", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow("select_roi", debug_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.background_substraction = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
                self.state = DetectorState.RUNNING

        if self.state == DetectorState.RUNNING:
            filtered_bbox = self.non_max_suppression(np.array(filtered_bbox), 0.5)

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
                mask = opening[int(ymin):int(ymax), int(xmin):int(xmax)]
                output_dets.append(Detection(int(xmin), int(ymin), int(xmax), int(ymax), "thing", 0.4, mask=mask, depth=depth))

            return output_dets
        else:
            return []

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

    def click_and_select(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points = [(x, y)]
            self.state = DetectorState.WAITING
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_points.append((x, y))
            self.state = DetectorState.READY


if __name__ == '__main__':
    from uwds3_perception.tracking.multi_object_tracker import MultiObjectTracker, iou_cost, color_cost
    from uwds3_perception.estimation.color_features_estimator import ColorFeaturesEstimator
    capture = cv2.VideoCapture(0)
    detector = ForegroundDetector()
    color_extractor = ColorFeaturesEstimator()
    tracker = MultiObjectTracker(iou_cost, color_cost, 0.1, 0.2, 15, 2, 3, use_appearance_tracker=False)
    tracks = []
    while True:
        ok, frame = capture.read()
        if ok:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #confirmed_tracks = [t for t in tracks if t.is_confirmed()]
            detections = detector.detect(frame)#, prior_detections=confirmed_tracks)
            color_extractor.estimate(rgb_image, detections=detections)
            tracks = tracker.update(rgb_image, detections)
            for t in tracks:
                if t.is_confirmed():
                    t.draw(frame, (36, 255, 12))
            cv2.imshow("result", frame)
            cv2.waitKey(1)
    capture.release()
