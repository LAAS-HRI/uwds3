#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from pyuwds3.navigation_pipeline import NavigationPipeline


class NavigationPipelineNode(object):
    def __init__(self):
        rospy.loginfo("[perception] Starting navigation perception...")
        self.pipeline = NavigationPipeline()
        rospy.loginfo("[perception] perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node("navigation_pipeline")
    core = NavigationPipelineNode().run()
