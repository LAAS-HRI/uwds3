import math
import dlib
from pyuwds3.types.detection import Detection


class DlibFaceDetector(object):
    def __init__(self):
        self.model = dlib.get_frontal_face_detector()

    def detect(self, rgb_image, depth_image=None):
        """ Detect frontal faces """
        face_detections = []

        faces = self.model(rgb_image, 0)

        for face in faces:
            if depth_image is not None:
                w = face.width()
                h = face.height()
                x = face.left() + w/2.0
                y = face.top() + h/2.0
                x = depth_image.shape[1]-1 if x > depth_image.shape[1] else x
                y = depth_image.shape[0]-1 if y > depth_image.shape[0] else y
                depth = depth_image[int(y)][int(x)]/1000.0
                if math.isnan(depth) or depth == 0.0:
                    depth = None
            else:
                depth = None
            face_detections.append(Detection(face.left(), face.top(), face.right(), face.bottom(), "face", 1.0, depth=depth))
        return face_detections
