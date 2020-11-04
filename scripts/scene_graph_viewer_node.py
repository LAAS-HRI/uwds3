#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from uwds3_msgs.msg import WorldStamped
import pygraphviz as pgv


class SceneGraphViewerNode(object):
    def __init__(self):
        self.world_input_topic = rospy.get_param("~world_input_topic", "corrected_tracks")
        self.world_subscriber = rospy.Subscriber(self.world_input_topic, WorldStamped, self.callback, queue_size=1)
        self.scene_graph_pub = rospy.Publisher(self.world_input_topic+"_scene_graph", Image, queue_size=1)
        self.cv_bridge = CvBridge()

    def callback(self, world_msg):
        nodes_names_map = {}
        G = pgv.AGraph(strict=False, directed=True)

        for node in world_msg.world.scene:
            if node.label != "face":
                node_name = node.description+"("+node.id[:6]+")"
                nodes_names_map[node.id] = node_name
                G.add_node(node_name)

        for situation in world_msg.world.timeline:
                if situation.object_id != "":
                    if situation.subject_id in nodes_names_map and situation.object_id in nodes_names_map:
                        subject_name = nodes_names_map[situation.subject_id]
                        object_name = nodes_names_map[situation.object_id]
                        G.add_edge(subject_name, object_name, label=situation.predicate)

        try:
            G.layout(prog='dot')
            G.draw("/tmp/"+self.world_input_topic+".png")
            scene_graph_img = cv2.imread("/tmp/"+self.world_input_topic+".png", flags=cv2.IMREAD_COLOR)
            self.scene_graph_pub.publish(self.cv_bridge.cv2_to_imgmsg(scene_graph_img, "bgr8"))
        except Exception as e:
            rospy.logwarn(str(e))

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    rospy.init_node("scene_graph_viewer", anonymous=True)
    recorder = SceneGraphViewerNode().run()
