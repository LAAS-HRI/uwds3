from keras.layers import Input, Dropout, Dense, Concatenate, Add, LSTM, Embedding
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from word_embeddings import create_embedding_layer


def image_model(input_shape, base_model="ResNet50", freeze_layers=30, vector_dim=300):
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
    print("Image features extractor:")
    model.summary()
    return model


def sequence_model(pre_trained_embedding_file, max_length, vector_dim):
    input = Input([max_length])
    embedding = create_embedding_layer(pre_trained_embedding_file)(input)
    x = Dropout(0.5)(embedding)
    preds = LSTM(vector_dim)(x)
    model = Model(input, preds)
    print("Sentence features extractor:")
    model.summary()
    return model


def merger_model(image_model,
                 sequence_model,
                 input_image_shape,
                 max_length,
                 vocab_size,
                 vector_dim):
    input_image = Input(input_image_shape)
    input_seq = Input([max_length])
    # Merger model => Merge (here by addition) the two representation
    image_embedding = image_model(input_image)
    seq_embedding = sequence_model(input_seq)
    merger = Add()([image_embedding, seq_embedding])
    decoder = Dense(vector_dim, activation='relu')(merger)
    preds = Dense(vocab_size, activation='softmax')(decoder)
    model = Model([input_image, input_seq], preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("Caption merger network:")
    model.summary()
    return model


def captioner_model(pre_trained_embedding_file,
                    caption_max_length=15,
                    input_image_shape=(128, 128, 3),
                    base_image_model="ResNet50",
                    freeze_base_layers=42,
                    vocab_size=40000,
                    vector_dim=300):
    seq_model = sequence_model(pre_trained_embedding_file, caption_max_length, vector_dim)
    img_model = image_model(input_image_shape, base_image_model, freeze_base_layers, vector_dim)
    model = merger_model(img_model, seq_model, input_image_shape, caption_max_length, vocab_size, vector_dim)

if __name__ == '__main__':
    glove_embedding_file = "../../models/features/glove/glove.6B.300d.txt"
    model = captioner_model(glove_embedding_file)
