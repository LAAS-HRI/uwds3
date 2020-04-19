import cv2
import numpy as np


class DescriptionMatcher(object):
    def __init__(self, pre_trained_embedding_file,
                 word_vector_dim,
                 siamese_model, siamese_weight,
                 similarity_threshold=0.6):
        self.model = cv2.readNetFromTensorflow(siamese_model, siamese_weight)
        self.word_to_vector = {}
        self.index_to_vector = {}
        self.word_vector_dim = word_vector_dim
        self.dimensions = (word_vector_dim, 0)
        self.dictionary = set()
        index = 1 # keras tokens start at 1
        with open(pre_trained_embedding_file, "r") as file:
            for line in file:
                tokens = line.split()
                vector = np.array([float(i) for i in tokens[2:]]).astype(np.float32)
                self.word_to_vector[tokens[0]] = vector
                self.index_to_vector[index] = vector
                index += 1

    def match(self, description, tracks, events, max_dist=0.5):
        pass

    def compute_fast_sentence_vector(self, description):
        word_vector_sequence = []
        for word in description.split():
            if word in self.word_to_vector:
                word_vector_sequence.append(self.word_to_vector[word])
        return np.average(np.array(word_vector_sequence), axis=0)

    # def compute_sentence_vector(self, description):
    #     pass
    #
    # def compute_fast_subword_vector(self, word):
    #     subword_vector =
    #     fragments = set(word[i:j] for i in range(len(word)) for j in range(i+3, len(word)+1))
    #     subwords = self.dictionary.intersection(fragments)
    #     #TODO compute subword vector
    #     return subword_vector
