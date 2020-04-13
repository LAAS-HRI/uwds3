import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


class AppearanceFeaturesExtractor(object):
    """
    """

    def __init__(self, model_type="ResNet50", weights="imagenet", input_shape=(128,128)):
        """
        """
        model_types = ["MobileNet", "MobileNetV2", "VGG16", "ResNet50"]
        weights_types = ["imagenet", "random"]
        if weights not in weights_types:
            raise ValueError("Invalid weights. Should be one of: {}".format(weights_types))
        if model_type not in model_types:
            raise ValueError("Invalid model type. Should be one of: {}".format(model_types))
        assert input_shape[0] > 32
        assert input_shape[1] > 32

        self.input_shape = input_shape

        if model_type == "MobileNet":
            self.model = MobileNet(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)
        if model_type == "MobileNetV2":
            self.model = MobileNetV2(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)
        if model_type == "VGG16":
            self.model = VGG16(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)
        if model_type == "ResNet50":
            self.model = ResNet50(weights=weights, include_top=False, pooling='avg', input_shape=input_shape)

    def extract(self, rgb_image, track=None):
        """
        """
        if track is not None:
            xmin = track.bbox.xmin
            ymin = track.bbox.ymin
            w = track.bbox.width()
            h = track.bbox.height()
            crop_image = rgb_image[ymin:ymin+h, xmin:xmin+w]
        else:
            crop_image = rgb_image
        image_resized = cv2.resize(crop_image, (self.input_shape[0], self.input_shape[1]))
        x = image.img_to_array(image_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return np.array(features).flatten()
