import numpy as np
from ..assignment.linear_assignment import LinearAssignment
from ...utils.bbox_metrics import overlap, centroid
from .monitor import Monitor
from scipy.spatial.distance import euclidean
from pyuwds3.utils.view_publisher import ViewPublisher
from pyuwds3.reasoning.monitoring.physics_monitor import PhysicsMonitor
import pybullet as p
MAX_VALUE = 666.


class Heatmap(Monitor):
    """ Special monitor for heatmap
    """
    def __init__(self, internal_simulator=None, beliefs_base=None):

        super(Heatmap, self).__init__(internal_simulator=internal_simulator, beliefs_base=beliefs_base)
        self.internal_simulator = internal_simulator
        self.heatm = {} # id -> heat value dictionay
        self.heatmax = {} # id -> max heat value ever reached dictionnary
        self.init_color = {} # id -> initial color dict (used to compute color)
        self.joint_index = {} #id -> joint index : used for color

        #self.max_obj = 0
        self.heatmap_publisher = ViewPublisher("heatmap") #output
      # def adj_obj(self,mx):
      #   for i in range(self.max_obj+1,mx+1):
      #       self.heatm[i]=0
      #   self.max_obj = mx

    def heat(self,nodes):
        node_list = []
        #creation of the list of id
        for node in nodes:
            node_list.append(node.id)
            if not (node.id in self.heatm): # add a new object to the heatmap
                self.heatm.setdefault(node.id,0)
                self.heatmax.setdefault(node.id,0)
                sim_id = self.internal_simulator.entity_id_map[node.id]
                visual_shapes = p.getVisualShapeData(sim_id)
                k=p.getVisualShapeData(sim_id)[0]
                # _,joint_id,_,_,_,_,_,color,_ =
                joint_id = k[1]
                color = k[7]
                self.init_color.setdefault(node.id,color) #INIT_COLOR_HERE
                self.joint_index.setdefault(node.id,joint_id)

        #computation of heatmap
        for key in self.heatm.keys():
            if key in node_list:
                self.heatm[key]=min(MAX_VALUE,self.heatm[key]+10)
            else:
                self.heatm[key]=max(0,self.heatm[key]-40)
            #compution of max heatmap
            if self.heatm[key]>self.heatmax[key]:
                self.heatmax[key] = self.heatm[key]

    def update_heatmaps(self,view_pose, camera):
        #update of heatmapp
        #nodes is the list of seen objects
        # n = self.max_obj
        _, _, _, nodes = self.simulator.get_camera_view(view_pose, camera)
        print (self.heatm)
        self.heat(nodes)
                # return 1
    def show_heatmap(self,view_pose,camera,stamp,fps):
        for id in self.heatm.keys():
            c = self.init_color[id]
            hm = self.heatm[id]
            joint_id = self.joint_index[id]
            r = c[0]+hm*(1-c[0])/MAX_VALUE
            g = c[1]*(1-hm/MAX_VALUE)
            b = c[2]*(1-hm/MAX_VALUE)
            a = c[3]
            # if id=="0":
            #     print ("heat=" + str(hm))
            #     print (r)
            #     print (g)
            #     print (b)
            #     print (a)
            #     print(self.init_color[id])
            sim_id= self.internal_simulator.entity_id_map[id]
            p.changeVisualShape(sim_id, joint_id, rgbaColor=[r, g, b, a], physicsClientId=self.internal_simulator.client_simulator_id)
        image,_,_,_ =  self.simulator.get_camera_view(view_pose, camera)
        self.heatmap_publisher.publish(image,[],stamp)
        # self.heatmap_publisher.publish(image,[],stamp,fps=fps)


#color :
# g = g_init*(1-hm/MAX_VALUE)
#b = b_init*(1-hm/MAX_VALUE)
#r = hm =0 -> ri  hm = MAX_VALUE -> 1
#r = hm(255-r_init)/(max_value)+r_init
