import cv2
import rospy
import uwds3_msgs.msg


class TemporalSituationType(object):
    PREDICATE = uwds3_msgs.msg.TemporalSituation.PREDICATE
    CAPTION = uwds3_msgs.msg.TemporalSituation.CAPTION


class TemporalSituation(object):
    def __init__(self, subject, description,
                 object="",
                 confidence=1.0,
                 expiration=2.0,
                 point=None):
        self.description = description
        self.subject = subject
        self.object = object
        self.confidence = confidence
        self.start = None
        self.end = None
        self.last_update = rospy.Time.now()
        self.expiration_duration = rospy.Duration(expiration)
        self.default_value = 0.0
        self.point = point

    def start(self, time=None):
        if time is None:
            self.start = rospy.Time.now()
        else:
            self.start = time

    def update(self, time=None):
        if time is None:
            self.last_update = rospy.Time.now()
        else:
            self.last_update = time

    def end(self, time=None):
        if time is None:
            self.end = rospy.Time.now()
        else:
            self.end = time

    def is_located(self):
        return self.point is not None

    def to_delete(self):
        return self.last_update + self.expiration_duration < rospy.Time.now()

    def to_msg(self, header):
        msg = uwds3_msgs.msg.TemporalSituation()
        msg.type = self.type
        msg.description = self.description
        msg.subject = self.subject
        msg.object = self.object
        msg.confidence = self.confidence
        if self.start is not None:
            msg.start = self.start
        if self.end is not None:
            msg.end = self.end
        if self.is_located():
            msg.point.header = header
            msg.point = self.point.to_msg()
        msg.last_update = self.last_update
        msg.expiration_duration = self.expiration_duration
        msg.default_value = self.default_value
        return msg


class Event(TemporalSituation):
    def __init__(self, subject, description,
                 object="",
                 confidence=1.0,
                 expiration=2.0,
                 time=None):
        self.description = description
        self.subject = subject
        self.object = object
        self.confidence = confidence
        self.start = rospy.Time.now()
        self.end = rospy.Time.now()
        self.last_update = rospy.Time.now()
        self.expiration_duration = rospy.Duration(expiration)
        self.default_value = 0.0
