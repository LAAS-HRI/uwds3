#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import rospy

if __name__ == '__main__':
    rospy.init_node("oro", anonymous=False)
    verbose = rospy.get_param("~verbose", True)
    robot_pkg_base = rospy.get_param("~robot_pkg_base", "~/openrobots")
    cmd = "cd "+robot_pkg_base+" && oro-server" if verbose is True else "cd "+robot_pkg_base+" && oro-server > null"
    os.system(cmd)
