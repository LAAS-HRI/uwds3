import rospy


class TimelinePublisher(object):
    def __init__(self, topic_name):
        self.publisher = rospy.Publisher(topic_name, queue_size=1)

    def publish(self, events):
        pass
