from word_embedding import create_embedding_layer
from siamese import siamese_network, train_siamese_network
from keras.models import Model
from keras.layers import Input, Dropout, Dense, Flatten, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2


def base_model(input_shape, vector_dim, base_model="ResNet50", freeze_layers=30):
    """Create the base model of the siamese network
    """
    input = Input(input_shape)
    if base_model == "ResNet50":
        base = ResNet50(include_top=False, weights="imagenet", input_tensor=input, input_shape=input_shape, pooling='avg', classes=1000)
    elif base_model == "VGG16":
        base = VGG16(include_top=False, weights="imagenet", input_tensor=input, input_shape=input_shape, pooling='avg', classes=1000)
    elif base_model == "MobileNet":
        base = MobileNet(include_top=False, weights="imagenet", input_tensor=input, input_shape=input_shape, pooling='avg', classes=1000)
    elif base_model == "MobileNetV2":
        base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=input, input_shape=input_shape, pooling='avg', classes=1000)
    else:
        raise ValueError("Invalid base model")
    x = base.output
    x = Dropout(0.5)(x)
    preds = Dense(vector_dim, activation="sigmoid")(x)
    model = Model(input, preds)
    if freeze_layers is not None:
        for layer in model.layers[0:freeze_layers]:
            layer.trainable = False
    print "Siamese CNN features extractor:"
    model.summary()
    return model


def siamese_cnn(input_shape, vector_dim):
    """Create siamese network
    """
    base = base_model(input_shape, vector_dim)
    return siamese_network(input_shape, base, metric="euclidean")


if __name__ == '__main__':
    input_shape = (128, 128, 3)
    num_classes = 10
    batch_size = 10
    epochs = 20
    model = siamese_cnn(input_shape, 2048)
