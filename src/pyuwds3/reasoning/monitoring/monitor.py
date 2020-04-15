from ...types.temporal_situation import TemporalSituation, Event


class Monitor(object):
    def __init__(self, internal_simulator=None, beliefs_base=None):
        self.relations = []
        self.relations_index = {}
        self.simulator = internal_simulator
        self.beliefs_base = beliefs_base

    def trigger_event(self, subject, event, object=None):
        if object is not None:
            e = Event(subject.id, event, object=object.id)
            print(subject.id[:6] + " " + event + " " + object.id[:6])
        else:
            e = Event(subject.id, event)
            print(subject.id[:6] + " " + event)
        self.relations.append(e)

    def start_situation(self, subject, situation, object=None):
        if object is not None:
            r = TemporalSituation(subject.id, situation, object=object.id)
        else:
            r = TemporalSituation(subject.id, situation)
        r.start()
        self.relations.append(r)
        self.relations_index[subject.id+str(situation)+object.id] = len(self.relations)-1

    def end_situation(self, subject, situation, object=None):
        if object is not None:
            if subject.id+str(situation)+object.id in self.relations_index:
                self.relations[self.relations_index[subject.id+str(situation)+object.id]].end()
        else:
            if subject.id+str(situation) in self.relations_index:
                self.relations[self.relations_index[subject.id+str(situation)]].end()
