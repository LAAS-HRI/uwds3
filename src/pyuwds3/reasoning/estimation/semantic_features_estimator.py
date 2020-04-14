import numpy as np
from pyuwds3.types.features import Features


class SemanticFeaturesEstimator(object):
    def __init__(self, pre_trained_embedding_file):
        self.name = "semantic"
        self.word_to_vector = {}
        self.index_to_vector = {}
        index = 0
        with open(pre_trained_embedding_file, "r") as file:
            for line in file:
                tokens = line.split()
                vector = np.array([float(i) for i in tokens[2:]]).astype(np.float32)
                self.word_to_vector[tokens[0]] = vector
                self.index_to_vector[index] = vector
                index += 1
        vector_dim = len(vector)
        self.dimensions = (vector_dim, 0)

    def estimate(self, tracks):
        word_vector_sequence = []
        for t in tracks:
            if self.name not in t.features:
                if t.description != "":
                    for word in t.description.split(" "):
                        if word in self.word_to_vector:
                            word_vector_sequence.append(self.word_to_vector[word])
                    sentence_vector = np.average(np.array(word_vector_sequence), axis=0)
                    t.features[self.name] = Features(self.name, self.dimensions, sentence_vector, 1.0)
