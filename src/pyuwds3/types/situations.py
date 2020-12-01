import cv2
import rospy
import uuid
import uwds3_msgs.msg


class SituationType(object):
    """ Situation types
    """
    CAPTION = uwds3_msgs.msg.TemporalSituation.CAPTION
    FACT = uwds3_msgs.msg.TemporalSituation.FACT
    ACTION = uwds3_msgs.msg.TemporalSituation.ACTION


class Situation(object):
    """ Represent a situation
    """
    def __init__(self,
                 type=SituationType.CAPTION,
                 subject="",
                 description="",
                 predicate="",
                 object="",
                 confidence=1.0,
                 expiration=5.0,
                 point=None):

        self.id = str(uuid.uuid4()).replace("-", "")
        self.type = type
        self.description = description
        self.predicate = predicate
        self.subject = subject
        self.object = object
        self.confidence = confidence
        self.expiration = expiration
        self.start_time = None
        self.end_time = None
        self.point = point

    def is_fact(self):
        """ Returns True if is a predicate
        """
        return self.type == SituationType.FACT

    def is_caption(self):
        """ Returns True if is a caption
        """
        return self.type == SituationType.CAPTION

    def is_action(self):
        """ """
        return self.type == SituationType.ACTION

    def is_event(self):
        """ Returns True if is an event
        """
        return self.end_time == self.start_time

    def is_finished(self):
        """ Returns True if finished
        """
        return self.end_time is not None

    def start(self, time=None):
        """ Start the temporal predicate
        """
        if time is None:
            self.start_time = rospy.Time.now()
        else:
            self.start_time = time
        return self

    def end(self, time=None):
        """ End the temporal predicate
        """
        if time is None:
            self.end_time = rospy.Time.now()
        else:
            self.end_time = time
        return self

    def is_located(self):
        """ Returns True if is located
        """
        return self.point is not None

    def to_delete(self, time=None):
        """ Returns True is to delete
        """
        if self.end_time is None:
            return False
        if time is None:
            now = rospy.Time().now()
        else:
            now = time
        return now > self.end_time + rospy.Duration(self.expiration)

    def from_msg(self, msg):
        """ Convert from ROS message
        """
        self.id = msg.id
        self.type = msg.type
        self.description = msg.description
        self.predicate = msg.predicate
        self.subject = msg.subject_id
        self.object = msg.object_id
        if msg.start == rospy.Time(0):
            self.start_time = None
        else:
            self.start_time = msg.start
        if msg.end == rospy.Time(0):
            self.end_time = None
        else:
            self.end_time = msg.end
        if msg.is_located is True:
            self.point = msg.point
        else:
            self.point = None
        return self

    def to_msg(self, header):
        """ Convert to ROS message
        """
        msg = uwds3_msgs.msg.TemporalSituation()
        msg.id = self.id
        msg.type = self.type
        msg.description = self.description
        msg.predicate = self.predicate
        msg.subject_id = self.subject
        msg.object_id = self.object
        if self.start_time is not None:
            msg.start = self.start_time
        if self.end_time is not None:
            msg.end = self.end_time
        if self.is_located():
            msg.point.header = header
            msg.point = self.point.to_msg()
        return msg

    def __eq__(self, other):
        return self.id == other.id

    def __str(self):
        return self.description

    def __repr__(self):
        return self.description


class Fact(Situation):
    def __init__(self,
                 subject="",
                 description="",
                 predicate="",
                 object="",
                 confidence=1.0,
                 expiration=5.0,
                 point=None,
                 action=False):
        super(Fact, self).__init__(type=SituationType.FACT,
                                   subject=subject,
                                   description=description,
                                   object=object,
                                   predicate=predicate,
                                   confidence=confidence,
                                   expiration=expiration,
                                   point=point)
        if action is True:
            self.type = SituationType.ACTION


class Event(Fact):
    def __init__(self,
                 subject="",
                 description="",
                 predicate="",
                 object="",
                 confidence=1.0,
                 expiration=5.0,
                 point=None,
                 time=None,
                 action=False):
        super(Event, self).__init__(subject=subject,
                                    description=description,
                                    object=object,
                                    predicate=predicate,
                                    confidence=confidence,
                                    expiration=expiration,
                                    point=point)

        if action is True:
            self.type = SituationType.ACTION
        if time is None:
            self.start = rospy.Time.now()
            self.end_time = rospy.Time.now()
        else:
            self.start = time
            self.end_time = time
        self.expiration_duration = rospy.Duration(expiration)
        self.point = point
