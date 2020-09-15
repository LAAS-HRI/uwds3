import numpy as np
import cv2
import math
import yaml
from pyuwds3.types.detection import Detection


class SSDDetector(object):
    """
    """
    def __init__(self, weights, model, config_file_path, input_size=(300, 300), max_overlap_ratio=0.3, swapRB=False, enable_cuda=True):
        """
        """
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = cv2.dnn.readNetFromTensorflow(weights, model)
        if enable_cuda is True:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.input_size = input_size
        self.max_overlap_ratio = max_overlap_ratio
        self.swapRB = swapRB

    def detect(self, rgb_image, depth_image=None):
        """ """
        frame_resized = cv2.resize(rgb_image, self.input_size, interpolation=cv2.INTER_AREA)

        self.model.setInput(cv2.dnn.blobFromImage(frame_resized, swapRB=self.swapRB))

        detections = self.model.forward()
        filtered_detections = []

        detection_per_class = {}
        score_per_class = {}

        rows = frame_resized.shape[0]
        cols = frame_resized.shape[1]

        height_factor = rgb_image.shape[0]/float(self.input_size[0])
        width_factor = rgb_image.shape[1]/float(self.input_size[1])

        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]
            if class_id in self.config:
                if self.config[class_id]["activated"] is True:
                    if confidence > self.config[class_id]["confidence_threshold"]:

                        class_label = self.config[class_id]["label"]

                        xmin = int(detections[0, 0, i, 3] * cols * width_factor)
                        ymin = int(detections[0, 0, i, 4] * rows * height_factor)
                        xmax = int(detections[0, 0, i, 5] * cols * width_factor)
                        ymax = int(detections[0, 0, i, 6] * rows * height_factor)

                        xmin = 0 if xmin < 0 else xmin
                        ymin = 0 if ymin < 0 else ymin
                        xmax = rgb_image.shape[1]-1 if xmax > rgb_image.shape[1]-1 else xmax
                        ymax = rgb_image.shape[0]-1 if ymax > rgb_image.shape[0]-1 else ymax

                        if (xmax - xmin) > 20 and (ymax - ymin) > 20:
                            bbox = [xmin, ymin, xmax, ymax, confidence]
                            if class_label not in detection_per_class:
                                detection_per_class[class_label] = []
                            if class_label not in score_per_class:
                                score_per_class[class_label] = []
                            detection_per_class[class_label].append(bbox)

        for class_label, dets in detection_per_class.items():
            filtered_dets = self.non_max_suppression(np.array(dets), self.max_overlap_ratio)
            for d in filtered_dets:
                if depth_image is not None:
                    w = d[2] - d[0]
                    h = d[3] - d[1]
                    x = d[0] + w/2.0
                    y = d[1] + h/2.0
                    x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                    y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                    depth = depth_image[int(y)][int(x)]/1000.0
                    if math.isnan(depth) or depth == 0.0:
                        depth = None
                else:
                    depth = None
                filtered_detections.append(Detection(d[0], d[1], d[2], d[3], class_label, d[4], depth=depth))

        return filtered_detections

    def non_max_suppression(self, boxes, max_bbox_overlap):
        """ Perform non maximum suppression
        Original code from pyimagesearch modified to works with detection that have a confidence
        """
        if len(boxes) == 0:
            return []

        boxes = boxes.astype(np.float)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        idxs = np.argsort(scores)

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
