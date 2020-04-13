#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from pyuwds3.dialogue_pipeline import DialoguePipeline


class DialoguePipelineNode(object):
    def __init__(self):
        rospy.loginfo("[perception] Starting dialogue perception...")
        self.pipeline = DialoguePipeline()
        rospy.loginfo("[perception] perception ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()


if __name__ == '__main__':
    rospy.init_node("dialogue_pipeline")
    core = DialoguePipelineNode().run()
