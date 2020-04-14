"""Modified from keras team siamese network example to be more modular"""
import random
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import numpy as np


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def siamese_network(input_shape, base_network, metric="euclidean", optimizer="adadelta"):
    input_a = Input(input_shape)
    input_b = Input(input_shape)
    # Siamese network => same network for embedding is used to share the weights
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)
    if metric == "euclidean":
        distance = Lambda(euclidean_distance,
                          output_shape=eucl_dist_output_shape)([embedding_a, embedding_b])
    elif metric == "cosine":
        distance = Lambda(cosine_distance,
                          output_shape=cos_dist_output_shape)([embedding_a, embedding_b])
    else:
        raise ValueError("Invalid metric provided: Should be 'euclidean' or 'cosine'")
    model = Model([input_a, input_b], distance)
    print "Siamese network:"
    model.summary()
    model.compile(optimizer=optimizer, loss=contrastive_loss)
    return model


def create_pairs(x, digit_indices, num_classes):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def train_siamese_network(model,
                          input_shape,
                          x_train, y_train,
                          x_test, y_test,
                          num_classes,
                          epochs=20,
                          batch_size=128):
    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices, num_classes)

    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    te_pairs, te_y = create_pairs(x_test, digit_indices, num_classes)

    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=128,
              epochs=epochs,
              validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)
    print("Training finished")
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    return model


def save_model(self, model, model_file_path):
    model_json = model.to_json()
    print "Saving model..."
    with open(model_file_path+".json", "w") as json_file:
        json_file.write(model_json)
    print "Saving weights..."
    model.save_weights(model_file_path+".h5")
