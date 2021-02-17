import numpy as np
import rospy
# from ..assignment.linear_assignment import LinearAssignment
# from ...utils.bbox_metrics import overlap, centroid
# from .monitor import Monitor
# from scipy.spatial.distance import euclidean
# from pyuwds3.utils.view_publisher import ViewPublisher
MAX_VALUE = 1000.
#I want the heating function to be  f(t)=MAX_VALUE(1-exp((t0-t)/3))
# It scale up to max_Value, and get at ~80% of it in 5sec
#at each step i will add f'(t)x delta_t:
# f'(t)= Max_Value/3 x exp(-t/3)
#rate of decrease is linear : -80*t :loosing 80%of value in 10sec
class Heatmap(object):
    """ Special monitor for heatmap
    """
    def __init__(self):

        self.heatm = {} # id -> heat value dictionay
        self.heatmax = {} # id -> max heat value ever reached dictionnary
        self.heat_time = {} #id -> time/last time it heat
        self.init_color = {} # id -> initial color dict (used to compute color)
        self.total_time = {} # id -> total time the user looked at the object
        self.last_time=0 #last time of update
        self.first_time=0 #time of the first update ~T0
        #self.max_obj = 0
        # self.heatmap_publisher = ViewPublisher("heatmap") #output
      # def adj_obj(self,mx):
      #   for i in range(self.max_obj+1,mx+1):
      #       self.heatm[i]=0
      #   self.max_obj = mx

    def heat(self,nodes,time):
        if self.first_time==0:
            self.first_time=time
            self.last_time=time
        node_list = []
        delta_t=time-self.last_time
        #creation of the list of id
        for node in nodes:
            node_list.append(node.id)
            if not (node.id in self.heatm): # add a new object to the heatmap
                self.heatm.setdefault(node.id,0)
                self.heatmax.setdefault(node.id,0)
                self.total_time.setdefault(node.id,0)
                self.heat_time.setdefault(node.id,self.last_time)
                # sim_id = self.internal_simulator.entity_id_map[node.id]
                # visual_shapes = p.getVisualShapeData(sim_id)
                # k=p.getVisualShapeData(sim_id)[0]
                # _,joint_id,_,_,_,_,_,color,_ =
                # joint_id = k[1]
                # color = k[7]
                self.init_color.setdefault(node.id,[0]*4) #INIT_COLOR_HERE

        #computation of heatmap
        for key in self.heatm.keys():
            if key in node_list:
                self.heat_time[key]=time
                add_value = ((MAX_VALUE/3.0)*np.exp((self.first_time-time)/3.))*delta_t
                self.heatm[key]=min(MAX_VALUE,self.heatm[key]+add_value)
                self.total_time[key]+=delta_t
            else:
                self.heatm[key]=max(0,self.heatm[key]-160*delta_t)
            #compution of max heatmap
            if self.heatm[key]>self.heatmax[key]:
                self.heatmax[key] = self.heatm[key]
        for node in nodes:
            self.color_node(node)


        self.last_time=time
    def color_node(self,node):
        if node.id in self.heatm:
            r=node.shapes[0].color[0]
            g=node.shapes[0].color[1]
            b=node.shapes[0].color[2]
            hm=self.heatm[node.id]
            node.shapes[0].color[0]=r+hm*(1-r)/MAX_VALUE
            node.shapes[0].color[1]= g*(1-hm/MAX_VALUE)
            node.shapes[0].color[2]=b*(1-hm/MAX_VALUE)
    # def show_heatmap(self,view_pose,camera,stamp,fps):
    #     for id in self.heatm.keys():
    #         c = self.init_color[id]
    #         hm = self.heatm[id]
    #         joint_id = self.joint_index[id]
    #         r = c[0]+hm*(1-c[0])/MAX_VALUE
    #         g = c[1]*(1-hm/MAX_VALUE)
    #         b = c[2]*(1-hm/MAX_VALUE)
    #         a = c[3]
    #         # if id=="0":
    #         #     print ("heat=" + str(hm))
    #         #     print (r)
    #         #     print (g)
    #         #     print (b)
    #         #     print (a)
    #         #     print(self.init_color[id])
    #         sim_id= self.internal_simulator.entity_id_map[id]
    #         p.changeVisualShape(sim_id, joint_id, rgbaColor=[r, g, b, a], physicsClientId=self.internal_simulator.client_simulator_id)
    #     image,_,_,_ =  self.simulator.get_camera_view(view_pose, camera)
    #     self.heatmap_publisher.publish(image,[],stamp)
    #     # self.heatmap_publisher.publish(image,[],stamp,fps=fps)
    #

#color :
# g = g_init*(1-hm/MAX_VALUE)
#b = b_init*(1-hm/MAX_VALUE)
#r = hm =0 -> ri  hm = MAX_VALUE -> 1
#r = hm(255-r_init)/(max_value)+r_init
