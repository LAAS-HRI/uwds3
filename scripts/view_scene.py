#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_msgs.msg import WorldStamped
import pygraphviz as pgv


class ViewScene(object):
    def __init__(self):
        world_input_topic = rospy.get_param("~world_input_topic", "corrected_tracks")
        self.world_subscriber = rospy.Subscriber(world_input_topic, WorldStamped, self.callback, queue_size=1)
        self.finished = False

    def callback(self, world_msg):
        nodes_names_map = {}
        G=pgv.AGraph(strict=False,directed=True)

        for node in world_msg.world.scene:
            node_name = node.description+"("+node.id[:6]+")"
            nodes_names_map[node.id] = node_name
            G.add_node(node_name)

        for situation in world_msg.world.timeline:
            if situation.object_id != "":
                subject_name = nodes_names_map[situation.subject_id]
                object_name = nodes_names_map[situation.object_id]
                G.add_edge(subject_name, object_name, label=situation.predicate)

        G.layout(prog='dot')
        G.draw("underworlds_scene.png")
        self.finished = True

    def run(self):
        while not rospy.is_shutdown():
            if self.finished is True:
                break
            else:
                rospy.spin()


if __name__ == "__main__":
    rospy.init_node("timeline_viewer", anonymous=True)
    recorder = ViewScene().run()
