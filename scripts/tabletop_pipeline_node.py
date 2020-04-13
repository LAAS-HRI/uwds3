#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from pyuwds3.tabletop_pipeline import TabletopPipeline


class TabletopPipelineNode(object):
    def __init__(self):
        rospy.loginfo("[perception] Starting tabletop perception...")
        self.pipeline = TabletopPipeline()
        rospy.loginfo("[perception] perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node("dialogue_pipeline")
    core = TabletopPipelineNode().run()
