#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle


class KNearestNeighborsAssignment(object):
    """   """

    def __init__(self, feature_name, max_distance, data_directory="", n_neighbors=1, algorithm="ball_tree", weights="distance"):
        """
        """
        self.feature_dim = None
        self.feature_name = feature_name
        self.trained = False
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.nb_samples = 0
        self.algorithm = algorithm
        self.max_distance = max_distance
        if data_directory == "":
            self.data_directory = "/tmp"
        else:
            self.data_directory = data_directory
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors,
                                      algorithm=self.algorithm)
        try:
            data = np.load(self.data_directory+"/"+self.feature_name+"_knn_classif.npz")
            self.X = list(data["x"])
            self.Y = list(data["y"])
            self.nb_samples = len(self.X)
            self.feature_dim = len(self.X[0])
            if self.n_neighbors is None:
                self.n_neighbors = math.sqrt(len(self.X[0]))
            self.train()
        except Exception:
            self.X = []
            self.Y = []

    def train(self):
        """
        """
        self.model.fit(np.array(self.X))

    def update(self, feature, label):
        """
        """
        if self.feature_dim is None:
            self.feature_dim = len(feature)
        self.X.append(feature)
        self.Y.append(label)
        self.nb_samples += 1

    def predict(self, feature):
        """
        """
        distances, matchs = self.model.kneighbors([feature])
        distance = distances[0]
        if distance > self.max_distance:
            return False, "unknown", 0.0
        indice = matchs[0][0]
        label = self.Y[indice]
        return True, label, distance

    def save(self, data_file_path):
        """
        """
        file = open(file, 'w')
        pickle.dump(self.knn, file)
        file.close()

    def load(self, data_file_path):
        """
        """
        file = open(file, 'r')
        self.knn = pickle.load(file)
        file.close()


class KNNLoader(object):
    def load(self, data_file_path):
        file = open(file, 'r')
        knn = pickle.load(file)
        file.close()
        return knn


if __name__ == '__main__':
    knn = KNearestNeighborsAssignment("test_face", 0.53)
    print("Add samples to KNN")
    knn.update([0.0, 0.0, 0.0], "bob")
    knn.update([0.0, 0.5, 0.0], "alice")
    knn.update([1.0, 1.0, 0.5], "max")
    knn.train()
    print("Training complete")
    success, label, distance = knn.predict([1.0, 1.0, 1.0])
    if success is True:
        print("Classification result: this is {}".format(label))
    else:
        print("Classification failed: unknown person")
