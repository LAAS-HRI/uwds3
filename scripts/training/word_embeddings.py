import numpy as np
from keras.layers import Embedding
from tqdm import tqdm


def load_pretrained_embedding(pretrained_embedding_file):
    """Load the pre-trained embedding
    """
    word_to_vector = {}
    index_to_vector = {}
    word_to_index = {}
    index = 1
    with open(pretrained_embedding_file, "r") as file:
        for line in tqdm(file):
            tokens = line.split()
            vector = np.array([float(i) for i in tokens[2:]]).astype(np.float32)
            word_to_vector[tokens[0]] = vector
            index_to_vector[index] = vector
            word_to_index[tokens[0]] = index
            index += 1
    vector_dim = len(vector)
    vocab_size = len(word_to_vector)
    return word_to_vector, index_to_vector, word_to_index, vector_dim, vocab_size


def compute_embedding_matrix(word_to_vector, word_to_index, vector_dim, vocab_size):
    """Compute the embedding matrix to create the keras layer
    """
    embedding_matrix = np.zeros((vocab_size, vector_dim))
    for word, index in word_to_index.items():
        embedding_vector = word_to_vector[word]
        if embedding_vector is not None:
            embedding_matrix[index-1] = embedding_vector
    return embedding_matrix


def create_embedding_layer(pre_trained_embedding_file, trainable=False):
    """Create the keras embedding layer from a pre-trained embedding
    """
    word_to_vector, _, word_to_index, vector_dim, vocab_size = load_pretrained_embedding(pre_trained_embedding_file)
    embedding_matrix = compute_embedding_matrix(word_to_vector, word_to_index, vector_dim, vocab_size)
    return Embedding(vocab_size, vector_dim, weights=[embedding_matrix], trainable=trainable)
