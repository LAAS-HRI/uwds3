#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
from collections import OrderedDict
from .beliefs_tensor import BeliefsTensor, FACT, MASK, ACTION
from .beliefs_explainer import BeliefsExplainer
from scipy.spatial.distance import cosine

AGENT = 1
OBJECT = 2


class BeliefsBase(object):
    """This knowledge base is a perspective-aware beliefs base that maintain himself and others minds as beliefs tensors"""

    def __init__(self, owner="myself", t_max=3, alpha=0.25):
        """ The default constructor """
        self.t_max = t_max
        self.alpha = alpha
        self.my_beliefs = BeliefsTensor(owner, t_max=self.t_max, alpha=self.alpha)
        self.my_beliefs.add_relation("see", MASK)
        self.other_beliefs = OrderedDict()

        self.explain = BeliefsExplainer(self)

    def get_description(self, entity, perspective=None):
        """ """
        if perspective is None:
            return self.my_beliefs.get_description(entity)
        else:
            return self.other_beliefs[perspective].get_description(entity)

    def get_entities(self):
        """ This method return the entities that are in the scene """
        return self.my_beliefs.get_entities()

    def get_agents(self):
        """ This method return the agents that are in the scene """

        return [entity for entity in self.my_beliefs.get_entities() if self.my_beliefs.get_entity_type(entity) < OBJECT]

    def get_objects(self):
        """ This method return the objects that are in the scene """

        return [entity for entity in self.my_beliefs.get_entities() if self.my_beliefs.get_entity_type(entity) == OBJECT]

    def get_relations(self):
        """ This method return the relations that are in the scene """

        return self.my_beliefs.get_relations()

    def add_entity(self, entity, type, description=None):
        """ This method add an entity to the scene """
        self.my_beliefs.add_entity(entity, type, description)

        for other, other_beliefs in self.other_beliefs.items():
            other_beliefs.add_entity(entity, type, description)

        if type == AGENT:
            self.other_beliefs[entity] = BeliefsTensor(entity, t_max=self.t_max, alpha=self.alpha)
            self.other_beliefs[entity].relations = self.my_beliefs.get_relations().copy()
            self.other_beliefs[entity].entities = self.my_beliefs.get_entities().copy()
            self.other_beliefs[entity].entities[0] = entity
            self.other_beliefs[entity].entities[self.my_beliefs.get_entity_index(entity)] = self.my_beliefs.get_owner()
            self.other_beliefs[entity].T = (np.copy(self.my_beliefs.T)*0.0)+0.5

            # Lvl 2 ToM really usefull ?!
            # self.other_beliefs[entity] = self.copy()
            # self.other_beliefs[entity].entities[0] = entity
            # self.other_beliefs[entity].entities[self.my_beliefs.get_entity_index(entity)] = self.my_beliefs.get_owner()
            # self.other_beliefs[entity].my_beliefs.B = (self.other_beliefs[entity].my_beliefs.B*0.0)+0.5

    def add_relation(self, relation, type):
        """ """
        self.my_beliefs.add_relation(relation, type)
        for other_belief in self.other_beliefs.values():
            other_belief.add_relation(relation, type)

    def update_description(self, entity, description):
        """ This method update the description of an entity """

        self.my_beliefs.update_description(entity, description)

    def update_triplet(self, subject, relation, object, confidence):
        """  """

        self.my_beliefs.update_triplet(subject, relation, object, confidence)
        self.propagate_triplet_to_others(subject, relation, object, confidence)

    def propagate_triplet_to_others(self, subject, relation, object, confidence):
        """ """

        for other, other_beliefs in self.other_beliefs.items():
            if subject != other:# and relation != "see":
                object_visibility_confidence = self.my_beliefs.get_triplet(other, "see", object)
                subject_visibility_confidence = self.my_beliefs.get_triplet(other, "see", subject)
                if (object_visibility_confidence*subject_visibility_confidence) > 0.5:
                    self.other_beliefs[other].update_triplet(subject, relation, object, confidence)
            else:
                self.other_beliefs[other].update_triplet(subject, relation, object, confidence)

    def get_triplet(self, subject, relation, object, perspective=None):
        """  """

        if perspective is None:
            confidence = self.my_beliefs.get_triplet(subject, relation, object)
        else:
            confidence = self.other_beliefs[perspective].get_triplet(subject, relation, object)
        return confidence


    def find(self, type, description, method="cosine", mask=None, perspective=None):
        """  """

        raise NotImplementedError()

    def relate(self, subject, relation, method="cosine", mask=None, perspective=None):
        """  """

        if perspective is None:
            pass
        else:
            pass
        pass

    def get_relational_feature(self, subject, object, mask=None, perspective=None):
        """  """

        if perspective is None:
            feature = self.my_beliefs.get_feature(subject, object)
        else:
            feature = self.other_beliefs[perspective].get_feature(subject, object)
        if mask is not None:
            return feature*mask.T
        else:
            return feature

    def commit(self):
        """ """

        self.my_beliefs.commit()
        for other, other_beliefs in self.other_beliefs.items():
            other_beliefs.commit()

    def evaluate_beliefs_divergeance_with(self, other, subject=None, relations_of_interest=None, entities_of_interest=None):
        """  """

        # TODO reduce tensor dim by using relations and entities of interest

        if other == self.my_beliefs.get_owner():
            raise ValueError("Invalid other '{}' provided, cannot compute ground truth with himself'.".format(other))
        Rprojected = self.my_beliefs.project_into(other)
        Restimated = self.other_beliefs[other].get_tensor()
        if subject is not None:
            Rprojected = Rprojected[:, self.my_beliefs.get_entity_index(subject), :]
            Restimated = Restimated[:, self.other_beliefs[other].get_entity_index(subject), :]

        assert Rprojected.shape == Restimated.shape

        return max(Rprojected.flatten() - Restimated.flatten())
        #return 1 - cosine(Rprojected.flatten(), Restimated.flatten())

    def __str__(self):
        return str(self.my_beliefs.get_tensor())

    def __sizeof__(self):
        memory_usage_bytes = sys.getsizeof(self.my_beliefs)
        for other, other_beliefs in self.other_beliefs.items():
            memory_usage_bytes += sys.getsizeof(other_beliefs)
        return memory_usage_bytes

    def get_memory_usage(self):
        """ """
        memory_usage_bytes = sys.getsizeof(self)
        memory_usage_gbytes = memory_usage_bytes/1073741824.0
        return "{} bytes ({} Go)".format(memory_usage_bytes, memory_usage_gbytes)
