import numpy as np
import rospy
from ..camera import *
from ..shape.mesh import Mesh
from uwds3_msgs.msg import WorldStamped
class Agent(SceneNode):
    def __init__(self,
                 detection=None,
                 label="agent",
                 description="",
                 pose=None,
                 n_init=1,
                 max_lost=10,
                 max_age=30,
                 camera = None,
                 shape_filename =None,
                 shape_color = None):
        super(Agent,self).__init__(detection,label,description,pose,n_init,max_lost,max_age)

        if camera is None:
            self.camera = camera
        else:
            self.camera = HumanCamera()

        if not shape_filename is None:
            if  pose is None:
                raise ValueError("Agent has a shape but no pose")
            else:
                shape = Mesh( shape_filename,
                             x=self.pos.x, y=self.pos.y, z=self.pos.z,
                             rx=self.rot.x, ry=self.rot.y, rz=self.rot.z)
                shape.color = shape_color
                node.shapes.append(shape)
        self.head_pose = None
