import numpy as np
import rospy
import fasttext


class StaticWordEmbeddings(object):
    """ Utility class to work with fasttext static word embeddings
    """
    def __init__(self, pretrained_embeddings_file):
        """ Static word embedding model contructor
        """
        self.model = fasttext.load_model(pretrained_embeddings_file)

    def compute_sentence_vector(self, sentence):
        """ Compute fast sentence vector
        """
        return np.average(np.array([self.model.get_word_vector(w) for w in sentence.split()]), axis=0)

    def get_similar_words(self, word):
        """ Return the most similar words
        """
        return self.model.get_nearest_neighbors(word)

    def get_most_similar_word(self, word):
        """ Return the most similar known word
        """
        score, word = self.model.get_nearest_neighbors(word)[0]
        return word

    def get_unknwon_words(self, sentence):
        """ Return the unknown words of the given sentence
        """
        return [w for w in sentence.split() if w not in self.model]
