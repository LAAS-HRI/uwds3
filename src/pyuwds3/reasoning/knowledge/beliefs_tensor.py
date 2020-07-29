"""
"""

import numpy as np
import math
import time
import sys

MYSELF = 0
OTHER = 1
OBJECT = 2

MASK = 0
FACT = 1
ACTION = 2

RAW = 0
FILTERED = 1


class BeliefsTensor(object):
    """ This class represent the beliefs of an agent as a 3D tensor of shape (nb_relation, nb_entities, nb_entities)"""

    def __init__(self, owner, save_dir="", unknown_confidence=0.5, unknown_delta=0.25, t_max=3, alpha=None, f=30.0):
        """ Base constructor """

        if alpha is not None:
            if alpha <= 0.0 and alpha > 1.0:
                raise AssertionError("Invalid smooting constant alpha value, should be between ]0.0,1.0] range")

        self.entities = np.array([])
        self.relations = np.array([])

        self.entity_types = np.array([], dtype=int)
        self.relation_types = np.array([], dtype=int)

        self.unknown_confidence = unknown_confidence
        self.unknown_delta = unknown_delta

        self.dt = 1.0/f

        self.origin = time.time()
        self.current = time.time()

        self.t_max = t_max

        self.alpha = alpha

        self.save_dir = save_dir

        self.owner = owner
        self.entities = np.append(self.entities, self.owner)
        self.entity_types = np.append(self.entity_types, MYSELF)

        self.E = {}

        self.T = np.full((t_max, len(self.relations),
                             len(self.entities),
                             len(self.entities)), self.unknown_confidence, dtype=np.float32)

    def get_entities(self, type=None):
        """ This method return the list of the entities present in the beliefs base """
        if type is None:
            return self.entities

    def get_entity_index(self, entity):
        """ This method return the index of the given entity """

        if entity not in list(self.entities):
            raise ValueError("Entity '{}' not registered.".format(entity))
        return np.where(self.entities == entity)[0][0]

    def get_relations(self, type=None):
        """ This method return list of the relations predicates """
        if type is None:
            return self.relations
        else:
            return [entity for entity in self.my_beliefs.get_entities() if self.my_beliefs.get_entity_type(entity) == type]

    def get_relation_index(self, relation):
        """ This method return the index of the given relation """

        if relation not in list(self.relations):
            raise ValueError("Relation '{}' not registered.".format(relation))
        return np.where(self.relations == relation)[0][0]

    def get_owner(self):
        """ This method return the owner of the beliefs """
        return self.owner

    def get_sub_tensor(self, relations_of_interest, entities_of_interest, t=0):
        raise NotImplementedError()

    def add_entity(self, entity, type, description=None):
        """ This method add an entity to the beliefs base (+1 in dim 1 and 2) """

        if type < 1:
            raise ValueError("Invalid entity type. Only stricly positive integers are accepted.")
        if entity in self.entities:
            raise ValueError("Entity '{}' already registered.".format(entity))

        Ttemp = np.full((self.T.shape[0], self.T.shape[1], self.T.shape[2] + 1, self.T.shape[3] + 1), 0.5)
        Ttemp[:, :, :-1, :-1] = self.T

        self.T = Ttemp

        self.entities = np.append(self.entities, entity)
        self.entity_types = np.append(self.entity_types, type)

        if description is not None:
            self.E[entity] = description

        return len(self.entities)-1

    def remove_entity(self, entity):
        """ This method remove an entity from the beliefs base (-1 in dim 2 and 3) """

        index = self.get_entity_index(entity)

        Ttemp = np.zeros((self.T.shape[0], self.T.shape[1], self.T.shape[2] - 1, self.T.shape[3] - 1))

        if index == 0:
            raise ValueError("Entity '{}' not removable.".format(entity))
        elif index == self.T.shape[1]:
            Ttemp = self.T[:, :, :-1, :-1]
        else:
            Ttemp[:, :, :index-1, :index-1] = self.T[:, :, :index-1, :index-1]
            Ttemp[:, :, index:, :index-1] = self.T[:, :, index+1:, :index-1]
            Ttemp[:, :, :index-1, index:] = self.T[:, :, :index-1, index+1:]
            Ttemp[:, :, index:, index:] = self.T[:, :, index+1:, index+1:]

        self.T = Ttemp

        self.entities = np.delete(self.entities, index)
        self.entity_types = np.delete(self.entity_types, index)

        if entity in self.E:
            del self.E[entity]

    def add_relation(self, relation, type):
        """ This method add a relation to the beliefs base (+1 in dim 1) """

        if relation in list(self.relations):
            raise ValueError("Relation '{}' already registered.".format(relation))

        self.relations = np.append(self.relations, relation)
        self.relation_types = np.append(self.relation_types, type)

        Ttemp = np.full((self.T.shape[0], self.T.shape[1] + 1, self.T.shape[2], self.T.shape[3]), self.unknown_confidence)
        Ttemp[:, :-1, :, :] = self.T
        self.T = Ttemp

        return len(self.relations)-1

    def remove_relation(self, relation):
        """ This method remove a relation from the beliefs base (-1 in dim 1) """

        if relation not in self.relations:
            raise ValueError("Trying to remove '{}' relation that is not registered.".format(relation))

        index = self.get_relation_index(relation)

        Ttemp = np.zeros((self.T.shape[0], self.T.shape[1]-1, self.T.shape[2], self.T.shape[3]))

        if index == 0:
            Ttemp = self.T[:, 1:, :, :]
        elif index == self.T.shape[0]:
            Ttemp = self.T[:, :-1, :, :]
        else:
            Ttemp[:, :index-1, :, :] = self.T[:, :index-1, :, :]
            Ttemp[:, index:, :, :] = self.T[:, index+1:, :, :]

        self.T = Ttemp

        self.relations = np.delete(self.relations, index)
        self.relation_types = np.delete(self.relation_types, index)

    def get_relation_matrix(self, relation, t=0):
        """ This method return the relation matrix of the given relation """

        return self.T[t, self.get_relation_index(relation), :, :]

    def update_description(self, entity, description):
        """ This method update the description of an entity or create it if new """

        self.E[entity] = description

    def descriptions(self):
        """ This method return the descriptions """

        return self.E

    def get_description(self, entity):
        """ """
        if entity not in self.E:
            raise ValueError("Description not available for '{}'".format(entity))
        return self.E[entity]

    def update_triplet(self, subject, relation, object, confidence, t=0):
        """ This method update the triplet probability """

        assert confidence >= 0.0 and confidence <= 1.0

        if relation not in self.get_relations():
            raise ValueError("Relation '{}' not registered in {}'s beliefs.".format(relation, self.get_owner()))
        if subject not in self.get_entities():
            raise ValueError("Subject '{}' not registered in {}'s beliefs.".format(subject, self.get_owner()))
        if object not in self.get_entities():
            raise ValueError("Object '{}' not registered in {}'s beliefs.".format(object, self.get_owner()))

        self.T[t][self.get_relation_index(relation)][self.get_entity_index(subject)][self.get_entity_index(object)] = confidence

    def get_triplet(self, subject, relation, object, threshold=None, t=0):
        """ This method return the triplet probability """

        if relation not in self.get_relations():
            raise ValueError("Relation '{}' not registered.".format(relation))
        if subject not in self.get_entities():
            raise ValueError("Subject '{}' not registered.".format(subject))
        if object not in self.get_entities():
            raise ValueError("Object '{}' not registered.".format(object))
        if threshold is None:
            return self.T[0, self.get_relation_index(relation)][self.get_entity_index(subject)][self.get_entity_index(object)]
        else:
            if threshold > 1.0 or threshold < 0.0:
                raise AssertionError("Invalid threshold value, should be between 1.0 and 0.0")
            return self.T[t, self.get_relation_index(relation)][self.get_entity_index(subject)][self.get_entity_index(object)] > threshold

    def get_entity_type(self, entity):
        """ This method return the type of entity """

        return self.entity_types[self.get_entity_index(entity)]

    def get_relation_type(self, relation):
        """ This method return the type of the given relation """

        return self.relation_types[self.get_relation_index(relation)]

    def get_feature(self, subject, object, sequence_lenght=1, t=0):
        """ This method return the relationnal feature of a given (subject, object) pair """

        return self.T[t, :, self.get_entity_index(subject), self.get_entity_index(object)]

    def project_into(self, other, t=0):
        """ This method return the owner's projected beliefs tensor """

        if self.get_entity_type(other) < OTHER:
            raise ValueError("Invalid entity provided, projection not available for objects or himself")

        frm = self.get_entity_index(self.get_owner())
        to = self.get_entity_index(other)

        if t is not None:
            Tout = np.copy(self.T[t])
            Tout[:, :, [frm, to]] = Tout[:, :, [to, frm]] # col
            Tout[:, [frm, to], :] = Tout[:, [to, frm], :] # row
        else:
            Tout = np.copy(self.T)
            Tout[:, :, :, [frm, to]] = Tout[:, :, :, [to, frm]] # col
            Tout[:, :, [frm, to], :] = Tout[:, :, [to, frm], :] # row
        return Tout

    def get_tensor(self, relations_of_interest=None, entities_of_interest=None, t=0, copy=True):
        """ This method return a copy of the beliefs tensor """

        if copy is True:
            return np.copy(self.T[t])
        else:
            return self.T[t]

    def __str__(self):
        return "{}'s tesseract : \r\n{}".format(self.owner, self.T[RAW])

    def commit(self):
        """ This method commit the changes into the timeline """

        self.current = time.time()
        index = self.T.shape[0]

        self.T[1:, :, :, :] = self.T[:index-1, :, :, :]

        if self.alpha is not None:
            self.T[1, :, :, :] = self.T[1, :, :, :] + (self.alpha*(self.T[0, :, :, :] - self.T[1, :, :, :]))

    def save(self, save_dir="", t=0):
        """ This method save in a file the owner's beliefs """

        if t is not None:
            np.save(save_dir+str(self.current)+".beliefs.npz", self.T[t], allow_pickle=False)
        else:
            np.save(save_dir+str(self.current)+".beliefs.npz", self.T, allow_pickle=False)
        np.save(save_dir+str(self.current)+".entities.npz", self.E, allow_pickle=False)

    def __sizeof__(self):
        """ """
        memory_usage_bytes = sys.getsizeof(self.T)
        memory_usage_bytes += sys.getsizeof(self.entities)
        memory_usage_bytes += sys.getsizeof(self.entity_types)
        memory_usage_bytes += sys.getsizeof(self.relations)
        memory_usage_bytes += sys.getsizeof(self.relation_types)
        return memory_usage_bytes
