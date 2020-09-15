import numpy as np
import cv2
import math
import yaml
from pyuwds3.types.detection import Detection


class MaskRCNNDetector(object):
    def __init__(self, weights, model, config_file_path, input_size=(800, 1365), max_overlap_ratio=0.3, mask_treshold=0.8, swapRB=False, enable_cuda=True):
        """ """
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.model = cv2.dnn.readNetFromTensorflow(weights, model)
        if enable_cuda is True:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.input_size = input_size
        self.max_overlap_ratio = max_overlap_ratio
        self.mask_treshold = mask_treshold
        self.swapRB = swapRB

    def detect(self, rgb_image, depth_image=None):
        """ """
        frame_resized = cv2.resize(rgb_image, self.input_size, interpolation=cv2.INTER_AREA)

        self.model.setInput(cv2.dnn.blobFromImage(frame_resized, swapRB=self.swapRB))

        (detections, masks) = self.model.forward(["detection_out_final", "detection_masks"])
        filtered_detections = []

        rows = frame_resized.shape[0]
        cols = frame_resized.shape[1]

        height_factor = rgb_image.shape[0]/float(self.input_size[0])
        width_factor = rgb_image.shape[1]/float(self.input_size[1])

        for i in range(0, detections.shape[2]):
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

                        w = xmax - xmin
                        h = ymax - ymin

                        mask = masks[i, class_id]
                        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask = (mask > self.mask_treshold)

                        if depth_image is not None:
                            x = xmin + w/2.0
                            y = ymin + h/2.0
                            x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                            y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                            depth = depth_image[int(y)][int(x)]/1000.0
                            if math.isnan(depth) or depth == 0.0:
                                depth = None
                        else:
                            depth = None

                        filtered_detections.append(Detection(xmin, ymin, xmax, ymax, class_label, confidence, mask=mask, depth=depth))

        return filtered_detections
