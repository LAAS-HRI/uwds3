from ...types.situations import SituationType, Fact, Event


class Monitor(object):
    def __init__(self, internal_simulator=None):
        """ Monitor constructor
        """
        self.relations = []
        self.relations_index = {}
        self.simulator = internal_simulator

    def cleanup_relations(self,time=None):
        """ Cleanup the relations buffer
        """
        to_keep = []
        index = {}
        for r in self.relations:
            if not r.to_delete(time=time):
                to_keep.append(r)
                index[r.subject+r.predicate+r.object] = len(to_keep) - 1
        self.relations = to_keep
        self.relations_index = index

    def trigger_event(self, subject, event, object=None, time=None):
        """ Trigger an event
        """
        if object is not None:
            description = subject.label+"("+subject.id[:6]+") "+event+" "+object.label+"("+object.id[:6]+")"
            e = Event(subject.id, description, predicate=event, object=object.id, time=time)
            self.relations_index[subject.id+str(event)+object.id] = len(self.relations)-1
        else:
            description = subject.label+"-"+subject.id[:6]+" "+event
            e = Event(subject.id, description, time=time)
            self.relations_index[subject.id+str(event)] = len(self.relations)-1
        self.relations.append(e)

    def start_fact(self, subject, predicate, object=None, time=None):
        """ Start a temporal predictate
        """
        if object is not None:
            if subject.id+str(predicate)+object.id not in self.relations_index:
                description = subject.description+"("+subject.id[:6]+") is "+predicate+" "+object.description+"("+object.id[:6]+")"
                relation = Fact(subject.id, description, predicate=predicate, object=object.id,expiration=5)
                relation.start(time=time)
                self.relations.append(relation)
                self.relations_index[subject.id+str(predicate)+object.id] = len(self.relations)-1
        else:
            if subject.id+str(predicate) not in self.relations_index:
                description = subject.description+"("+subject.id[:6]+") is "+predicate
                relation = Fact(subject.id, description, predicate=predicate)
                self.relations.append(relation.start(time=time))
                self.relations_index[subject.id+str(predicate)] = len(self.relations)-1

    def end_fact(self, subject, predicate, object=None, time=None):
        """ End a temporal predicate
        """
        if object is not None:
            if subject.id+str(predicate)+object.id in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)+object.id]]
                relation.description.replace("is", "was")
                relation.end(time=time)
                del self.relations_index[subject.id+str(predicate)+object.id]
        else:
            if subject.id+str(predicate) in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)]].end(time=time)
                relation.description.replace("is", "was")
                relation.end(time=time)
                del self.relations_index[subject.id+str(predicate)]
