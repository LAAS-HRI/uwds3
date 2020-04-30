import cv2
import numpy as np


class DescriptionMatcher(object):
    def __init__(self, static_word_embeddings, similarity_threshold=0.6):
        self.model = static_word_embeddings
        self.similarity_threshold = similarity_threshold

    def match(self, description, tracks, events):
        pass
