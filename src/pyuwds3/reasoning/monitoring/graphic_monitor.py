import numpy as np
import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from ...utils.bbox_metrics import overlap
from ...utils.allocentric_spatial_relations import is_on_top, is_included, is_close
from ...utils.egocentric_spatial_relations import is_right_of, is_left_of
from ...types.camera import Camera
from scipy.spatial.distance import euclidean
from pyuwds3.types.vector.vector6d_stable import Vector6DStable
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.shape.mesh import Mesh
from .physics_monitor import ActionStates
from pyuwds3.utils.view_publisher import ViewPublisher
from pyuwds3.utils.world_publisher import WorldPublisher
from pyuwds3.utils.marker_publisher import MarkerPublisher
from pyuwds3.utils.tf_bridge import TfBridge
from pyuwds3.utils.uwds3_ontologenius_bridge import OntologeniusReaderNode

import math

import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import tf2_ros
import tf2_geometry_msgs
from ...types.vector.vector6d import Vector6D
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from ontologenius import OntologiesManipulator
from ontologenius import OntologyManipulator

from  pr2_motion_tasks_msgs.msg import RobotAction

import pybullet as p



FILTERING_Y = 15
FILTERING_Z = 20
INF = 10e3
#Ray >1
N_RAY = 5
ALPHA_THRESHOLD = 0.6

def centroid_cost(track_a, track_b):
    """Returns the centroid cost"""
    try:
        return euclidean(track_a.pose.position().to_array(), track_b.pose.position().to_array())
    except Exception:
        return INF
class AgentType(object):
    HUMAN = "h"
    ROBOT = "r"

class GraphicMonitor(Monitor):
    """ Special monitor for agent management
    """
    def __init__(self,agent=None,agent_type =AgentType.ROBOT,
     handL = None,handR=None, head = "head_mount_kinect2_rgb_optical_frame", internal_simulator=None,
       position_tolerance=0.04,name="robot"): #beliefs_base=None,

        super(GraphicMonitor, self).__init__(internal_simulator=internal_simulator)#, beliefs_base=beliefs_base)

        self.tf_bridge = TfBridge()
        self.filtering_y_axis = rospy.get_param("~filtering_y_axis", FILTERING_Y)
        self.filtering_z_axis = rospy.get_param("~filtering_z_axis", FILTERING_Z)
        #init of the simulator and ontology
        self.internal_simulator = internal_simulator
        self.ontologies_manip = OntologiesManipulator()
        self.global_frame_id = rospy.get_param("~global_frame_id")
        self.world_publisher = WorldPublisher("corrected_tracks", self.global_frame_id)
        self.marker_publisher = MarkerPublisher("ar_perception_marker")

        self.onto_bridge = OntologeniusReaderNode(name)
        self.ontologies_manip.add("robot")
        self.onto=self.ontologies_manip.get("robot")
        self.onto.close()

        #init of the camera (kinect fov)
        self.camera = Camera(640)
        self.camera.setfov(84.1,53.8)

        #link between the physics simlatro and the reality
        #not used for now
        # self.position_tolerance = position_tolerance

        # dictionnary of the transparent object
        self.alpha_dic = {}

        #type of agent + limb name
        self.agent_type = agent_type
        self.agent = agent
        self.handL = handL
        self.handR = handR
        self.head  = head
        self.human_pose = None
        self.headpose = None

        #A map of graphic monitor class, one/agent
        self.agent_map={}


        #view of robot and human
        self._publisher = ViewPublisher(name+"_view")
        self._publisher2=ViewPublisher("human_view")

        #tf subscription



        # dictionnary of the picked objects
        self.pick_map = {}


        #mocap dicttionary, and object to publish dictionnary
        self.mocap_obj={}
        self.publish_dic={}

        #time, and fps limit a a
        self.time_monitor=rospy.Time().now().to_sec()
        self.time_view=rospy.Time().now().to_sec()
        self.n_frame_monitor=15
        self.n_frame_view = 15

        self.gone_map={}


    def pick_callback(self, msg):
        """
        #deal w/ robot pick
        """
        if msg.action == RobotAction.PICK:
            print msg.arm
            if msg.arm=="left_arm":
                self.pick_map[msg.objId]=self.handL

            if msg.arm=="right_arm":
                self.pick_map[msg.objId]=self.handR

        else:
            if msg.objId in self.pick_map:
                del self.pick_map[msg.objId]


    def get_head_pose(self,time):
        """
            #get the head pose in simulator, so in the global_frame_id (often map)
        """
        s,hpose=self.tf_bridge.get_pose_from_tf(self.simulator.global_frame_id,
                                                        self.head,time)
        return hpose
    def get_delta_head_pose(self,time,deltax=0,deltay=0,deltaz=0):
        s,hpose=self.tf_bridge.get_pose_from_tf(self.simulator.global_frame_id,
                                                        self.head,time)
        # s,pose_map =self.tf_bridge.get_pose_from_tf(self.global_frame_id, header.frame_id[1:],header.stamp)
        vect = Vector6DStable(deltax,deltay,deltaz)
        vect.from_transform(np.dot(hpose.transform(),vect.transform()))
        return vect



    def publish_view(self,tfm):
        """
        #publish the view of the agent,
        #create a map of all the object seen at a defnie time
        """
        time = rospy.Time.now().to_sec()
        header = rospy.Header()
        header.frame_id ='map'
        if time-self.time_view >1./(self.n_frame_view):
            #if needed : >7166666 work

            self.time_view=time
            if self.agent_type== AgentType.ROBOT:
                for obj_id in self.mocap_obj.keys():
                    if not obj_id in self.publish_dic:
                        self.publish_dic[obj_id]=WorldPublisher(str(obj_id)+"_tracks", self.global_frame_id)
                for obj_id in self.publish_dic.keys():
                    view_pose=self.mocap_obj[obj_id].pose + Vector6DStable(0.15,0,0,0,np.pi/2)

                    _, _, _, nodes = self.simulator.get_camera_view(view_pose, self.camera)
                    # for node in nodes:
                    #     node.last_update = self.mocap_obj[obj_id].last_update
                    header.stamp=self.mocap_obj[obj_id].last_update
                    self.publish_dic[obj_id].publish(nodes,[],header)



            # self.internal_simulator.step_simulation()
        # # print self.internal_simulator.entity_id_map
        # self.frame_count %= self.n_frame
        # if time-self.time > 7166666:
        #     # hpose=self.get_head_pose()
        #     # print hpose
        #     #
        #     # image,_,_,_ =  self.simulator.get_camera_view(hpose, self.camera)
        #     # self._publisher.publish(image,[],rospy.Time.now())
        #     # self.time=time
        #     # self.head.pose.rot.x+=0.03
        #     # print self.head.pose.rot.x
            # self.head.pose.rot.y+=0.06
            # print self.head.pose.rot.y
            # print self.head.pose.pos
            # print "   "



    def mocap(self,tracks,header):
        """
        mocap : add object in the simulator
        each human is present 2 time in the tracks : 1 for the head/1for the body
        """
        for object in tracks:
            if object.is_located() and object.has_shape():
                if not self.simulator.is_entity_loaded(object.id):
                    self.simulator.load_node(object)
                self.simulator.reset_entity_pose(object.id, object.pose)
                if not "_body" in object.id:
                    # if not object.id in self.agent_map:
                    #     self.agent_map[object.id]=GraphicMonitor(agent=None,agent_type =AgentType.ROBOT,
                    #      handL = None,handR=None, head = "head_mount_kinect2_rgb_optical_frame", internal_simulator=None,
                    #        position_tolerance=0.04,name="robot")
                    self.mocap_obj[object.id]=object
                    self.simulator.reset_entity_pose(object.id, object.pose)
                # self.human_pose=object.pose
                # self.simulator.change_joint(object.id,0,2)
                # print p.getVisualShapeData(self.simulator.entity_id_map[object.id])
                # p.resetJointState(self.simulator.entity_id_map[object.id],0,0.1)


    #main element :

    def monitor(self, object_tracks, pose, header):
        """ Monitor the physical consistency of the objects and compute fact

        """
        #place all object of the tracks in the simulator
        time = header.stamp
        self.cleanup_relations()

        # print "pbublish dixt" + str(len(self.publish_dic))
        # print "agent map " + str(len(self.agent_map))
        # print "pick map " + str(len(self.pick_map ))
        # print "mocap obj " + str(len(self.mocap_obj))
        # print "rel ind " + str(len(self.relations_index))
        # print "relations  " + str(len(self.relations))

        if pose != None:
            for object in object_tracks:
                if object.is_located() and object.has_shape():
                    print object.id
                    print object.last_update.to_sec()
                    # object.pose.from_transform(np.dot(pose.transform(),object.pose.transform()))
                    if not self.simulator.is_entity_loaded(object.id):
                        self.simulator.load_node(object)
                    self.disapearing_object(object,header)
                    print self.gone_map
                    if object.id in self.gone_map:
                        # print "=================="
                        # print object.id
                        # print object.last_update.to_sec()-header.stamp.to_sec()
                        if object.last_update.to_sec() +1< header.stamp.to_sec():
                            object.pose.pos.z -=42
                        else:
                            del self.gone_map[object.id]
                    base_link_sim_id = self.simulator.entity_id_map[object.id]
                    self.simulator.reset_entity_pose(object.id, object.pose)
            # self.marker_publisher.publish(object_tracks,header)

        #publish the head view

        if abs(time.to_sec()-self.time_monitor) >1./(self.n_frame_monitor):
            hpose=self.get_head_pose(time)
            # print hpose
            image,_,_,_ =  self.simulator.get_camera_view(hpose, self.camera)

            self._publisher.publish(image,[],time)

            self.time_monitor=time.to_sec()


        #compute the facts
        self.compute_allocentric_relations(object_tracks, time)
        # print ("robot")
        # print self.get_head_pose(time).pos.to_array()
        # self.compute_egocentric_relations(list(self.get_head_pose(time).pos.to_array()),object_tracks, time)
        # print ("_________________________")
        # if not self.headpose is None:
        #     print ("human")
        #     print list(self.headpose.pos.to_array())[:3]
        #     self.compute_egocentric_relations(list(self.headpose.pos.to_array())[:3],object_tracks, time)
        #     # print self.headpose
        #     print ("_________________________")


        self.world_publisher.publish([],self.relations,header)

        # print self.relations_index
        return object_tracks, self.relations

    #
    # def assign_and_trigger_action(self, object, action, person_tracks, time):
    #     """ Assign the action to the closest person of the given object and trigger it """
    #     matches, unmatched_objects, unmatched_person = self.centroid_assignement.match(person_tracks, [object])
    #     if len(matches > 0):
    #         _, person_indice = matches[0]
    #         person = person_tracks[person_indice]
    #         self.trigger_event(person, action, object, time)

    def test_occlusion(self, object, tracks, occlusion_threshold=0.8):
        """ Test occlusion with 2D bbox overlap
        """
        overlap_score = np.zeros(len(tracks))
        for idx, track in enumerate(tracks):
            overlap_score[idx] = overlap(object, track)
        idx = np.argmax(overlap_score)
        object = tracks[idx]
        score = overlap[idx]
        if score > occlusion_threshold:
            return True, object
        else:
            return False, None

#cansee
#getvisualshapedata [X][7] = color
    #
    def can_reach(self,start_pose,obj):
        """
        compute if obj can be reached from the start pos
        """
        # TODO: add some small movement
        end_id=self.simulator.entity_id_map[obj.id]
        if self.can_reach_rot(start_pose,end_id):
            return True

        return False


    def can_reach_rot(self,start_pose,end_id):
        """
        compute if obj can be reached from the start pos with no move
        """
        [xmin,ymin,zmin],[xmax,ymax,zmax] = p.getAABB(end_id)
        xlength = xmax - xmin
        ylength = ymax - ymin
        zlength = zmax - zmin
        pose_list = []
        for i in range(N_RAY):
            end_pose = [xmin +i*xlength/(N_RAY-1),
            ymin +i*xlength/(N_RAY-1),
            zmin +i*xlength/(N_RAY-1)]
            r = p.rayTest(start_pose,end_pose)
            if r[0][0]== end_id:
                return True
        return False

    def canSee(self,start_pose,obj):
        """ compute if obj can be seen from start pose"""
        end_id=self.simulator.entity_id_map[obj.id]
        [xmin,ymin,zmin],[xmax,ymax,zmax] = p.getAABB(end_id)
        xlength = xmax - xmin
        ylength = ymax - ymin
        zlength = zmax - zmin
        pose_list = []
        # print start_pose
        for i in range(N_RAY):
            for j in range (N_RAY):
                for k in range(N_RAY):
                    end_pose = [xmin +i*xlength/((N_RAY-1)*1.),
                    ymin +j*xlength/((N_RAY-1)*1.),
                    zmin +k*xlength/((N_RAY-1)*1.)]
                    if self.canSeeRec(start_pose,end_pose,end_id,0):
                        return True
        # pose_end=obj.pose.pos.to_array()
        # [pose_end[0],pose_end[1],pose_end[2]]
        start_pose=[start_pose[0][0],start_pose[1][0],start_pose[2][0]]
        if self.canSeeRec(start_pose,end_pose,end_id,0):
            return True
        return False

    def canSeeRec(self,start_pose,end_pose,end_id,hitnumber):
        """
        recursive function to achied the canSee goal
        ray cast btween strart pose and end pose
        we go through transparent object (the recursive part)
        """
        r=p.rayTestBatch([start_pose],[end_pose],reportHitNumber = hitnumber)

        # print r[0]
        # print end_id
        print r[0][0]
        if r[0][0] == end_id:
            return True
        if r[0][0]==-1:
            return False
        if not (r[0][0],r[0][1]) in self.alpha_dic:
            data = p.getVisualShapeData(r[0][0])
            if r[0][1] +1 >0 and r[0][1] +1 <len(data) and data[r[0][1] +1][1] == r[0][1]:
                self.alpha_dic[[0][0],r[0][1]] =data[ r[0][1] +1][7]
            else:
                for i in data:
                    if i[1]== r[0][1]:
                        self.alpha_dic[r[0][0],r[0][1]]=i[7]

        if self.alpha_dic[(r[0][0],r[0][1])] >ALPHA_THRESHOLD:
            return False
        return self.canSeeRec(r[0][3],end_pose,end_id,hitnumber+1)

    # def hasInView(self,start_pose,end_id,camera):

#     def compute_allocentric_relations(self, objects, time):
#         for obj1 in objects:
#             if obj1.is_located() and obj1.has_shape() and obj1.label!="no_fact":
#                 for obj2 in objects:
#                     if obj1.id != obj2.id:
#                         # evaluate allocentric relation
#                         if obj2.is_located() and obj2.has_shape() and obj2.label!="no_fact":
#                             # get 3d aabb
#                             success1, aabb1 = self.simulator.get_aabb(obj1)
#                             success2, aabb2 = self.simulator.get_aabb(obj2)
#
#                             if success1 is True and success2 is True:
#                                 # if (obj1.id == "cube_BBCG" and obj2.id == "box_B5"):
#                                 print obj1.id
#                                 print obj2.id
#                                 if is_on_top(aabb1, aabb2):
#                                     self.start_predicate(obj1, "on", object=obj2, time=time)
#                                     # self.onto.feeder.addObjectProperty(obj1.id,"isOnTopOf",obj2.id,time)
#                                     # print obj1.id
#                                     # print obj2.id
#                                     # print self.onto.individuals.getOn(obj1.id,"isOnTopOf")
#                                     # print self.onto.individuals.getOn(obj2.id,"isOnTopOf")
# #TODO COPY CA DANS SELF
#                                 else:
#                                     self.end_predicate(obj1, "on", object=obj2, time=time)
#
#                                 if is_included(aabb1, aabb2):
#                                     self.start_predicate(obj1, "in", object=obj2, time=time)
#                                     # self.onto.feeder.addObjectProperty(obj1.id,"isIn",obj2.id,time)
#                                 else:
#                                     self.end_predicate(obj1, "in", object=obj2, time=time)
        # print self.relations_index

    # def reset_allocentric_relation(self,objects,time):
    #     #TODO: PAS  FAIRE CA
    #     #DELETE LA VERSION PRECEDENDE
    #         for obj1 in objects:
    #             if obj1.is_located() and obj1.has_shape() and obj1.label!="no_fact":
    #                 for obj2 in objects:
    #                     if obj1.id != obj2.id:
    #                         # evaluate allocentric relation
    #                         if obj2.is_located() and obj2.has_shape() and obj2.label!="no_fact":
    #                             self.onto.feeder.removeObjectProperty(obj1.id,"isOnTopOf",obj2.id,time)
    #                             self.onto.feeder.removeObjectProperty(obj1.id,"isIn",obj2.id,time)

    #
    # def compute_allocentric_relations(self, objects, time):
    #     redo_onto = False
    #
    #     print self.relations_index
    #     for obj1 in objects:
    #         if obj1.is_located() and obj1.has_shape():
    #             success1, aabb1 = self.simulator.get_aabb(obj1)
    #             if success1:
    #                 is_on = True
    #                 is_in = True
                    # on = self.onto.individuals.getOn(obj1.id,"isOnTopOf")
                    # in_ = self.onto.individuals.getOn(obj1.id,"isIn")
    #
    #                 if len(on)>0:
    #                     if on[0].is_located() and on[0].has_shape():
    #                         success2, aabb2 = self.simulator.get_aabb(on[0])
    #                         if success2:
    #                             is_on = is_on_top(aabb1, aabb2)
    #                             # if is_on == False:
    #                                 #DELETE LINK IN ONTO
    #                 if len(in_)>0:
    #                     if in_[0].is_located() and in_[0].has_shape():
    #                         success2, aabb2 = self.simulator.get_aabb(in_[0])
    #                         if success2:
    #                             is_in = is_in(aabb1, aabb2)
    #                             # if is_on == False:
    #                                 #DELETE LINK IN ONTO
    #             if not (is_on and is_in):
    #                 for obj2 in objects:
    #                     if obj2 != obj1:
    #                         success2, aabb2 = self.simulator.get_aabb(obj2)
    #                         if success2:
    #                             if not is_on:
    #                                 is_on = is_on_top(aabb1, aabb2)
    #                                 if is_on:
    #                                     if redo_onto:
    #                                         redo_onto=True
    #                                         self.reset_allocentric_relation(object_tracks,time)
    #                                     self.onto.feeder.addObjectProperty(obj1.id,"isOnTopOf",obj2.id,time)
    #                             if not is_in:
    #                                 is_in = is_included(aabb1, aabb2)
    #                                 if is_in:
    #                                     if redo_onto:
    #                                         redo_onto=True
    #                                         self.reset_allocentric_relation(object_tracks,time)
    #                                     self.onto.feeder.addObjectProperty(obj1.id,"isIn",obj2.id,time)
    #             for obj2 in objects:
    #                 if obj1.id != obj2.id:
    #                     # evaluate allocentric relation
    #                     if obj2.is_located() and obj2.has_shape():
    #                         # get 3d aabb
    #                         success1, aabb1 = self.simulator.get_aabb(obj1)
    #                         success2, aabb2 = self.simulator.get_aabb(obj2)


    def disapearing_object(self,node,header):
        # if node.id=="table_1":
        #     print node.last_update.to_sec()
        #     print  header.stamp.to_sec()
        #     print node.last_update.to_sec() - header.stamp.to_sec()
        if "ube"in node.id:
            print "BEGIN"
            print node.id
            print node.last_update.to_sec()
            print  header.stamp.to_sec()
            print  header.stamp.to_sec() - node.last_update.to_sec()
            print self.pos_validityv2(node.pose,header)
            print "END"
        if node.last_update.to_sec() +1< header.stamp.to_sec():
            if self.pos_validityv2(node.pose,header):
                print "if2"
                start_pose = (self.get_head_pose(header.stamp)).pos.to_array()[:3]
                start_pose=[start_pose[0][0],start_pose[1][0],start_pose[2][0]]
                end_id=self.simulator.entity_id_map[node.id]
                # print node.last_seen_position.values()
                for end_pose_v6 in node.last_seen_position.values():
                    end_pose =end_pose_v6.pos.to_array()
                    # print end_pose

                    end_pose=[end_pose[0][0],end_pose[1][0],end_pose[2][0]]
                    # print "end_id"
                    # print node.id
                    if self.canSeeRec(start_pose,end_pose,end_id,0):
                        # print "here"
                        self.gone_map[node.id]=True
                        return

    def pos_validityv2(self,mvect,header):
        mpose = Vector6DStable(mvect.pos.x,mvect.pos.y,mvect.pos.z)
        frame_id = header.frame_id
        if frame_id[0]=='/':
            frame_id = frame_id[1:]
        bool_,head_pose = self.tf_bridge.get_pose_from_tf("head_mount_kinect2_rgb_link" ,
                                          frame_id,header.stamp)
        mpose.from_transform(np.dot(head_pose.transform(),mpose.transform()))
        #mpose is now in the head frame
        if mpose.pos.x==0:
            return False,None
        xy_angle = np.degrees(np.arctan(mpose.pos.y/mpose.pos.x))
        xz_angle = np.degrees(np.arctan(mpose.pos.z/mpose.pos.x))

        return (abs(xy_angle)<self.filtering_y_axis and
               abs(xz_angle)<self.filtering_z_axis),
    def compute_egocentric_relations(self,pose,objects,time):
        """ compute the egocentric relations (can see can reach)"""
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape() and obj1.label!="no_fact":
                if self.canSee(pose,obj1):
                    self.start_fact(obj1, "isSeen", time=time)
                else:
                    self.end_fact(obj1, "isSeen", time=time)

                if self.can_reach(pose,obj1):
                    self.start_fact(obj1, "isReachable", time=time)
                else:
                    self.end_fact(obj1, "isReachable", time=time)


    def compute_allocentric_relations(self, objects, time):
        included_map={}
        #included_map[a] = [b,c,d] <=> a is in b in c and in d
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape() and obj1.label!="no_fact":
                included_map[obj1.id]=[]
                for obj2 in objects:
                    if obj1.id != obj2.id:
                        # evaluate allocentric relation
                        if obj2.is_located() and obj2.has_shape() and obj2.label!="no_fact":
                            # get 3d aabb
                            success1, aabb1 = self.simulator.get_aabb(obj1)
                            success2, aabb2 = self.simulator.get_aabb(obj2)
                            if success1  and success2 :
                                if is_included(aabb1, aabb2) and (not (obj1.id in self.pick_map)) and (not (obj2.id in self.pick_map)):
                                    self.start_fact(obj1, "in", object=obj2, time=time)
                                    included_map[obj1.id].append(obj2.id)
                                else:
                                    self.end_fact(obj1, "in", object=obj2, time=time)
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape() and obj1.label!="no_fact":
                for obj2 in objects:
                    if obj1.id != obj2.id:
                        # evaluate allocentric relation
                        if obj2.is_located() and obj2.has_shape() and obj2.label!="no_fact":
                            # get 3d aabb
                            success1, aabb1 = self.simulator.get_aabb(obj1)
                            success2, aabb2 = self.simulator.get_aabb(obj2)
                            if success1  and success2 :
                                if  included_map[obj1.id]==included_map[obj2.id] and is_on_top(aabb1, aabb2) and (not (obj1.id in self.pick_map)) and (not (obj2.id in self.pick_map)):
                                    self.start_fact(obj1, "on", object=obj2, time=time)
                                else:
                                    self.end_fact(obj1, "on", object=obj2, time=time)
