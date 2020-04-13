import numpy as np


class BeliefsExplainer(object):
    """ This reasoner is in charge of generating human friendly explaination about beliefs """

    def __init__(self, tesseractor):
        """ Constructor of the reasoner """
        self.tsr = tesseractor

    def beliefs(self, owner, subject=None, relations_of_interest=None, entities_of_interest=None):
        """ This method explain the beliefs of the given owner """

        explaination = ""

        if owner == self.tsr.my_beliefs.get_owner():
            beliefs = self.tsr.my_beliefs
            owner_str = "I"
        else:
            beliefs = self.tsr.other_beliefs[owner]
            owner_str = "I think that "+str(owner)

        if relations_of_interest is None:
            relations_of_interest = beliefs.get_relations()
        if entities_of_interest is None:
            entities_of_interest = beliefs.get_entities()

        if subject is None:
            subjects_of_interest = beliefs.get_entities()
        else:
            subjects_of_interest = [subject]

        first = True

        for relation_index in range(0, len(beliefs.get_relations())):
            relation = beliefs.get_relations()[relation_index]
            if relation in relations_of_interest:

                for subject_index in range(0, len(beliefs.get_entities())):
                    subject = beliefs.get_entities()[subject_index]
                    if subject in subjects_of_interest:

                        for object_index in range(0, len(beliefs.get_entities())):
                            object = beliefs.get_entities()[object_index]
                            if object in entities_of_interest:

                                confidence = beliefs.get_triplet(subject, relation, object)

                                if first is False and not (confidence < 0.75 and confidence > 0.35):
                                    explaination += "\r\n"

                                if confidence < 0.75 and confidence > 0.35:
                                    pass
                                elif confidence >= 0.75:
                                    explaination += owner_str + " believe that {} {} {} ({} confidence)".format(subject, relation, object, confidence)
                                    first = False if first is True else False
                                else:
                                    explaination += owner_str + " believe that {} not {} {} ({} confidence)".format(subject, relation, object, confidence)
                                    first = False if first is True else False
        return explaination

    def my_beliefs(self, subject=None, relations_of_interest=None, entities_of_interest=None):
        """ This method explain my own beliefs """

        return self.beliefs(self.tsr.my_beliefs.get_owner(), subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)

    def others_beliefs(self, subject=None, person_of_interest=None, relations_of_interest=None, entities_of_interest=None):
        """ This method explain others beliefs """

        explaination = ""
        if person_of_interest is None:
            person_of_interest = self.tsr.get_agents()

        first = True
        for other, other_beliefs in self.tsr.other_beliefs.items():
            if other in person_of_interest:
                if first is False:
                    explaination += "\r\n"
                first = False if first is True else False
                explaination += self.beliefs(other, subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)
        return explaination

    def beliefs_difference_with(self, other, subject=None, relations_of_interest=None, entities_of_interest=None):
        """ This method explain the difference of beliefs between myself and other """

        explaination = ""

        if other == self.tsr.my_beliefs.get_owner():
            raise ValueError("Invalid other provided, cannot explain difference between I and myself.")

        if relations_of_interest is None:
            relations_of_interest = self.tsr.other_beliefs[other].get_relations()
        if entities_of_interest is None:
            entities_of_interest = self.tsr.other_beliefs[other].get_entities()

        beliefs = self.tsr.other_beliefs[other]

        if subject is None:
            subjects_of_interest = self.tsr.other_beliefs[other].get_entities()
        else:
            subjects_of_interest = [subject]

        divergeance = self.tsr.evaluate_beliefs_divergeance_with(other, subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)
        if divergeance >= 0.98:
            return "I think that "+str(other)+" and I believe the same things ({} confidence)".format(divergeance)

        Bdiff = np.absolute(self.tsr.other_beliefs[other].get_tensor() - self.tsr.my_beliefs.project_into(other))

        first = True

        for relation_index in range(0, len(beliefs.get_relations())):
            relation = beliefs.get_relations()[relation_index]
            if relation in relations_of_interest:

                for subject_index in range(0, len(beliefs.get_entities())):
                    subject = beliefs.get_entities()[subject_index]
                    if subject in subjects_of_interest:

                        for object_index in range(0, len(beliefs.get_entities())):
                            object = beliefs.get_entities()[object_index]
                            if object in entities_of_interest:

                                if Bdiff[relation_index, subject_index, object_index] > 0.25:
                                    confidence = beliefs.get_triplet(subject, relation, object)
                                    #print("{} {} {} ({} confidence) diff : {}".format(subject, relation, object, confidence, Bdiff[relation_index, subject_index, object_index]))
                                    if first is False:
                                        explaination += "\r\n"
                                    if confidence < 0.75 and confidence > 0.25:
                                        truth = self.tsr.my_beliefs.project_into(other)[relation_index, subject_index, object_index]
                                        if truth >= 0.75:
                                            explaination += "I think that " + other + " don't known that {} {} {} ({} confidence)".format(subject, relation, object, truth)
                                            first = False if first is True else False
                                        elif truth < 0.25:
                                            explaination += "I think that " + other + " don't known that {} not {} {} ({} confidence)".format(subject, relation, object, truth)
                                            first = False if first is True else False
                                    elif confidence > 0.75:
                                        explaination += "I think that " + other + " believe that {} {} {} ({} confidence)".format(subject, relation, object, confidence)
                                        first = False if first is True else False
                                    else:
                                        explaination += "I think that " + other + " believe that {} not {} {} ({} confidence)".format(subject, relation, object, confidence)
                                        first = False if first is True else False
        return explaination

    def false_beliefs(self, subject=None, person_of_interest=None, relations_of_interest=None, entities_of_interest=None, perspective=None):
        """ This method explain false beliefs relatively to my own beliefs """

        explaination = ""

        if person_of_interest is None:
            person_of_interest = self.tsr.get_agents()

        first = True

        for other, other_beliefs in self.tsr.other_beliefs.items():
            if other in person_of_interest:

                current_explaination = self.beliefs_difference_with(other, subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)

                if current_explaination != "":
                    if first is False:
                        explaination += "\r\n" + current_explaination
                    else:
                        explaination += current_explaination
                    first = False if first is True else False

        return explaination

    def repair_beliefs(self, other, subject=None, relations_of_interest=None, entities_of_interest=None):
        """ """
        explaination = ""

        if other == self.tsr.my_beliefs.get_owner():
            raise ValueError("Invalid other provided, cannot repair with myself.")

        if relations_of_interest is None:
            relations_of_interest = self.tsr.other_beliefs[other].get_relations()
        if entities_of_interest is None:
            entities_of_interest = self.tsr.other_beliefs[other].get_entities()

        divergeance = self.tsr.evaluate_beliefs_divergeance_with(other, subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)
        if divergeance >= 0.98:
            return ""
        else:
            explaination += "@{} :\r\n".format(other)

        Bdiff = np.absolute(self.tsr.other_beliefs[other].get_tensor() - self.tsr.my_beliefs.project_into(other))

        first = True

        beliefs = self.tsr.other_beliefs[other]

        if subject is None:
            subjects_of_interest = self.tsr.other_beliefs[other].get_entities()
        else:
            subjects_of_interest = [subject]

        for relation_index in range(0, len(beliefs.get_relations())):
            relation = beliefs.get_relations()[relation_index]
            if relation in relations_of_interest:

                for subject_index in range(0, len(beliefs.get_entities())):
                    subject = beliefs.get_entities()[subject_index]
                    if subject in subjects_of_interest:

                        for object_index in range(0, len(beliefs.get_entities())):
                            object = beliefs.get_entities()[object_index]
                            if object in entities_of_interest:

                                if Bdiff[relation_index, subject_index, object_index] > 0.3:
                                    confidence = beliefs.get_triplet(subject, relation, object)

                                    if first is False:
                                        explaination += "\r\n"
                                    if confidence < 0.75 and confidence > 0.35:
                                        truth = self.tsr.my_beliefs.project_into(other)[relation_index, subject_index, object_index]
                                        if truth >= 0.75:
                                            explaination += "{} is not {} {} ({} confidence)".format(subject, relation, object, truth)
                                            first = False if first is True else False
                                        elif truth < 0.25:
                                            explaination += "{} is {} {} ({} confidence)".format(subject, relation, object, truth)
                                            first = False if first is True else False
                                    elif confidence > 0.75:
                                        explaination += "{} is {} {} ({} confidence)".format(subject, relation, object, confidence)
                                        first = False if first is True else False
                                    else:
                                        explaination += "{} is not {} {} ({} confidence)".format(subject, relation, object, confidence)
                                        first = False if first is True else False
        return explaination

    def repair_others_beliefs(self, subject=None, person_of_interest=None, relations_of_interest=None, entities_of_interest=None):

        explaination = ""

        if person_of_interest is None:
            person_of_interest = self.tsr.get_agents()

        first = True

        for other, other_beliefs in self.tsr.other_beliefs.items():
            if other in person_of_interest:

                current_explaination = self.repair_beliefs(other, subject=subject, relations_of_interest=relations_of_interest, entities_of_interest=entities_of_interest)

                if current_explaination != "":
                    if first is False:
                        explaination += "\r\n" + current_explaination
                    else:
                        explaination += current_explaination
                    first = False if first is True else False

        return explaination
