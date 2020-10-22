import cv2
import numpy as np
from ...types.features import Features


class AppearanceFeaturesEstimator(object):
    def __init__(self, weights, model, output_dim=2048):
        self.model = cv2.dnn.readNetFromTensorflow(weights)#, model)
        self.name = "appearance"
        self.dimensions = (output_dim, 1)

    def estimate(self, rgb_image, detections):
        """ """
        cropped_imgs = []
        detections_to_process = []

        for det in detections:
            if self.name not in det.features:
                xmin = int(det.bbox.xmin)
                ymin = int(det.bbox.ymin)
                w = int(det.bbox.width())
                h = int(det.bbox.height())
                detections_to_process.append(det)
                cropped_imgs.append(rgb_image[ymin:ymin+h, xmin:xmin+w])

        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          size=(224, 224),
                                          scalefactor=1/255.0,
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for det, features in zip(detections_to_process, self.model.forward()):
                confidence = max(1.0, rgb_image.shape[0]/(2.0*float(h)))
                if self.name not in det.features:
                    det.features[self.name] = Features(self.name, self.dimensions, np.array(features).flatten(), confidence)
                else:
                    det.features[self.name].update(np.array(features), confidence)
