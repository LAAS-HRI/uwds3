from ...types.temporal_situation import TemporalSituationType, TemporalPredicate, Event


class Monitor(object):
    def __init__(self, internal_simulator=None, beliefs_base=None):
        self.relations = []
        self.relations_index = {}
        self.simulator = internal_simulator
        self.beliefs_base = beliefs_base

    def trigger_event(self, subject, event, object=None, time=None):
        if object is not None:
            description = subject.label+"-"+subject.id[:6]+" "+event+" "+object.label+"-"+object.id[:6]
            e = Event(subject.id, description, predicate=event, object=object.id, time=time)
            self.relations_index[subject.id+str(event)+object.id] = len(self.relations)-1
        else:
            description = subject.label+"-"+subject.id[:6]+" "+event
            e = Event(subject.id, description, time=time)
            self.relations_index[subject.id+str(event)] = len(self.relations)-1
        print("Evt: "+description)
        self.relations.append(e)

    def start_predicate(self, subject, predicate, object=None, time=None):
        if object is not None:
            description = subject.label+"-"+subject.id[:6]+" is "+predicate+" "+object.label+"-"+object.id[:6]
            relation = TemporalPredicate(subject.id, description, predicate=predicate, object=object.id)
            relation.start(time=time)
            self.relations.append(relation)
            self.relations_index[subject.id+str(predicate)+object.id] = len(self.relations)-1
        else:
            description = subject.label+"-"+subject.id[:6]+" is "+predicate
            relation = TemporalPredicate(subject.id, description, predicate=predicate)
            self.relations.append(relation.start(time=time))
            self.relations_index[subject.id+str(predicate)] = len(self.relations)-1
        print("Str: "+description)


    def end_predicate(self, subject, predicate, object=None, time=None):
        if object is not None:
            if subject.id+str(predicate)+object.id in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)+object.id]]
                print("End: "+subject.label+"-"+subject.id[:6]+" was "+predicate+" "+object.label+"-"+object.id[:6])
                relation.end(time=time)
        else:
            if subject.id+str(predicate) in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)]].end(time=time)
                relation.end(time=time)
                print("End: "+subject.label+"-"+subject.id[:6]+" was "+predicate)
