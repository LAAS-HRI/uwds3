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
            e = Event(subject.id, description, object=object.id, time=time)
        else:
            description = subject.label+"-"+subject.id[:6]+" "+event
            e = Event(subject.id, description, time=time)
        print(description)
        self.relations.append(e)

    def start_predicate(self, subject, predicate, object=None, time=None):
        if object is not None:
            description = subject.label+"-"+subject.id[:6]+" is "+predicate+" "+object.label+"-"+object.id[:6]
            self.relations.append(TemporalPredicate(subject.id, description, predicate=predicate, object=object.id, time=time))
        else:
            description = subject.label+"-"+subject.id[:6]+" is "+predicate
            self.relations.append(TemporalPredicate(subject.id, description, predicate=predicate, time=time))
        print("start: "+description)
        self.relations_index[subject.id+str(predicate)+object.id] = len(self.relations)-1

    def end_predicate(self, subject, predicate, object=None, time=None):
        if object is not None:
            if subject.id+str(predicate)+object.id in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)+object.id]]
                relation.end(time=time)
                print("end: "+subject.label+"-"+subject.id[:6]+" "+predicate+" "+object.label+"-"+object.id[:6])
        else:
            if subject.id+str(predicate) in self.relations_index:
                relation = self.relations[self.relations_index[subject.id+str(predicate)]].end(time=time)
                relation.end(time=time)
                print("end: "+subject.label+"-"+subject.id[:6]+" "+predicate)
