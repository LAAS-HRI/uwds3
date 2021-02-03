#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import os
import rospy
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.utils.marker_publisher import MarkerPublisher
from pyuwds3.utils.world_publisher import WorldPublisher
from optitrack_ros.msg import or_pose_estimator_state
from geometry_msgs.msg import Pose
from pyuwds3.types.vector.vector6d_stable import Vector6DStable
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.shape.sphere import Sphere
from pyuwds3.types.shape.cylinder import Cylinder
from pyuwds3.types.shape.mesh import Mesh
import numpy as np
from pyuwds3.types.shape.shape import Shape, ShapeType
NODE_NAME = "mocap_human_localization"
HUMAN_SUB_PARAM_NAME = "optitrack_human_topic_names"
HUMAN_SUB_PREPEND = "/optitrack/bodies/"
TF_FRAME_PREPEND = "mocap_human-"
TIMER_CALLBACK = 0.2

# A human head localization need to be named : Helmet_smth
# A hman hand localization need to be name smth pick smth

class MocapHumanLocalization(object):
    """ Mocap Huamn localisation class"""
    def __init__(self):
        # self.opt_human_topic_name = rospy.get_param(HUMAN_SUB_PARAM_NAME,   "person_01_torso")
        self.opt_human_topic_name = "crane"
        self.tf_bridge = TfBridge()
        self.tfOptitrack2Humans_ ={}
        self.subscribedNodeNames =rospy.get_param("~optitrack_human_topic_names",None)
        # self.subscribedNodeNames={"Neophasia": 1}

        self.personSubs =[]
        self.world_publisher = WorldPublisher("mocap_tracks","map")
        self.marker_publisher = MarkerPublisher("mocap")
        self.world2map_x = rospy.get_param("~world2map_x",6.4868)
        self.world2map_y = rospy.get_param("~world2map_y",2.8506)
        self.world2map_z = rospy.get_param("~world2map_z",0)

        self.world2map_roll = rospy.get_param("~world2map_roll",0)
        self.world2map_pitch = rospy.get_param("~world2map_pitch",0)
        self.world2map_yaw = rospy.get_param("~world2map_yaw",0)

        if not self.subscribedNodeNames is None:
            print or_pose_estimator_state()
            for key in self.subscribedNodeNames.keys():
                fullTopicName = HUMAN_SUB_PREPEND + key

                self.personSubs.append(rospy.Subscriber(
                        fullTopicName,
                        or_pose_estimator_state,
                        self.updateMocapPersonPose_callback,
                        (self.subscribedNodeNames[key],key)  ))

                new_node =SceneNode(label="human")

                if "Helmet" in key:
                    shape = Mesh("package://uwds3/models/cad_models/human/human_head1.stl",
                             x=0, y=-0.4, z=0,
                             rx=0, ry=0, rz=0)
                    r,g,b=1,1,1
                    shape.color[0] = r
                    shape.color[1] = g
                    shape.color[2] = b
                    shape.color[3] = 1
                    new_node.id=key
                    new_node.shapes.append(shape)
                    self.tfOptitrack2Humans_[self.subscribedNodeNames[key]]=new_node

                    new_id=key+"_body"
                    new_body=SceneNode(pose = Vector6D(x=0,y=0,z=0,rx=0,ry=0,rz=0),label="human")
                    body_shape = Cylinder(0.3,1.7, "shp1",x=0,y=-0.4,z=-1.1,a=1,r=1,g=1,b=1)
                    new_body.id=new_id
                    new_body.shapes.append(body_shape)
                    self.tfOptitrack2Humans_[new_id]=new_body
                else:
                    new_node.id=key
                    self.tfOptitrack2Humans_[self.subscribedNodeNames[key]]=new_node

        self.timer = rospy.Timer(rospy.Duration(TIMER_CALLBACK), self.world_publisher_timer_callback)
        self.header = rospy.Header()
        self.header.frame_id ='map'
        self.world2map_transform = Vector6D(x=self.world2map_x,y=self.world2map_y,z=self.world2map_z,
        rx=self.world2map_roll,ry=self.world2map_pitch, rz=self.world2map_yaw).transform()


        # self.test_callback = rospy.Subscriber(
        #         "/optitrack/bodies/Helmet_3",
        #         or_pose_estimator_state,
        #         self.updateMocapPersonPose_callback
        #         ,
        #         self.subscribedNodeNames["Helmet_3"]
        #         )
        #print self.subscribedNodeNames
    def world_publisher_timer_callback(self,timer):
        self.world_publisher.publish(self.tfOptitrack2Humans_.values(),[],self.header)

        # self.tfOptitrack2Humans_.values()[0].shapes.append(Sphere(d=.1, r=.5))
        #print self.tfOptitrack2Humans_.values()[0].shapes[0].width()
        # #print self.header
        self.marker_publisher.publish(self.tfOptitrack2Humans_.values(),self.header)
        # #print self.tfOptitrack2Humans_.values()[0].pose

    def updateMocapPersonPose_callback(self,msg,args):
        humanId,key = args
        # print(len(msg.pos))
        if len(msg.pos) >0:
            pose_received = Vector6DStable( msg.pos[0].x,msg.pos[0].y,msg.pos[0].z)
            pose_received.from_quaternion(  msg.att[0].qx,
                                            msg.att[0].qy,
                                            msg.att[0].qz,
                                            msg.att[0].qw)
            # pose_received.rot.z +=1.57
            pose_received.from_transform(np.dot(self.world2map_transform,pose_received.transform()))
            # object.pose.from_transform(np.dot(pose.transform(),object.pose.transform()))
            # pose_received.pos.x += self.world2map_x
            # pose_received.pos.y += self.world2map_y
            # pose_received.pos.z += self.world2map_z

            self.tfOptitrack2Humans_[humanId].pose = pose_received
            self.header.stamp.secs=msg.ts.sec
            self.header.stamp.nsecs=msg.ts.nsec
            self.tfOptitrack2Humans_[humanId].time = self.header.stamp

            if str(key)+"_body" in self.tfOptitrack2Humans_ :
                self.tfOptitrack2Humans_[str(key)+"_body"].pose.pos.x=pose_received.pos.x
                self.tfOptitrack2Humans_[str(key)+"_body"].pose.pos.y=pose_received.pos.y
                self.tfOptitrack2Humans_[str(key)+"_body"].pose.pos.z=pose_received.pos.z
                self.tfOptitrack2Humans_[str(key)+"_body"].pose.rot.z=pose_received.rot.z

    def run(self):

        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == "__main__":
    rospy.init_node("mocap_human_localization", anonymous=False)
    perception = MocapHumanLocalization().run()
