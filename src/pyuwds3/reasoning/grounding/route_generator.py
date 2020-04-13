import cv2


class RouteGenerator(object):
    """ """
    def __init__(self, model, weights, config_file, max_depth=25):
        self.model = cv2.readNetFromTensorflow(model, weights)


    def generate(self, depth_image, goal_track):
        pass
