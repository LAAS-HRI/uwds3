import numpy as np
from ...types.features import Features


class SemanticFeaturesEstimator(object):
    """ Semantic features estimator that convert the description to word vector representation
    """
    def __init__(self, static_word_embeddings, vector_dim=300):
        """ SemanticFeaturesEstimator constructor
        """
        self.name = "semantic"
        self.dimensions = (vector_dim, 0)
        self.static_word_embeddings = static_word_embeddings

    def estimate(self, tracks):
        """ Estimate the semantic features of the given tracks
        """
        for t in tracks:
            if self.name not in t.features:
                if t.description != "":
                    sentence_vector = self.static_word_embeddings.compute_sentence_vector(t.description)
                    t.features[self.name] = Features(self.name, self.dimensions, sentence_vector, 1.0)
