import cv2
from .bbox import BoundingBox


class Detection(object):
    """Represents a 2D detection associated with a label and a confidence"""

    def __init__(self, xmin, ymin, xmax, ymax, label, confidence, depth=None, mask=None):
        """Detection constructor"""
        self.label = label
        self.confidence = confidence
        self.bbox = BoundingBox(xmin, ymin, xmax, ymax, depth=depth)
        self.mask = None
        if mask is not None:
            if mask.shape[0] != self.bbox.height() or mask.shape[1] != self.bbox.width():
                try:
                    self.mask = cv2.resize(mask.astype("uint8"), (self.bbox.width(), self.bbox.height()))
                except Exception:
                    self.mask = None
            else:
                self.mask = mask
        self.features = {}

    def draw(self, image, color):
        """Draws the detection"""
        text_color = (0, 0, 0)
        cv2.rectangle(image, (self.bbox.xmin, self.bbox.ymax-20),
                      (self.bbox.xmax, self.bbox.ymax), (200, 200, 200), -1)
        self.bbox.draw(image, color, 2)
        self.bbox.draw(image, text_color, 1)
        cv2.putText(image, "{:0.2f}".format(self.confidence), (self.bbox.xmax-40, self.bbox.ymax-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        cv2.putText(image, self.label, (self.bbox.xmin+5, self.bbox.ymax-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    def __str__(self):
        return "{}:{} with {} confidence".format(self.bbox, self.label, self.confidence)
