#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_msgs.msg import WorldStamped
from jsk_rviz_plugins.msg import OverlayText


class TimelineViewer(object):
    def __init__(self):
        world_input_topic = rospy.get_param("~world_input_topic", "")
        self.world_subscriber = rospy.Subscriber(world_input_topic, WorldStamped, self.callback, queue_size=1)
        self.overlay_publisher = rospy.Publisher("event_overlay_text", OverlayText, queue_size=1)

    def callback(self, world_msg):
        overlay_msg = OverlayText()
        overlay_msg.height = 1500
        overlay_msg.width = 350
        overlay_msg.text_size = 12
        overlay_msg.fg_color.a = 1.0
        overlay_msg.bg_color.a = 0.2

        for situation in world_msg.world.timeline:
            overlay_msg.text += "\r\n" + situation.description

        self.overlay_publisher.publish(overlay_msg)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    rospy.init_node("timeline_viewer", anonymous=True)
    recorder = TimelineViewer().run()
