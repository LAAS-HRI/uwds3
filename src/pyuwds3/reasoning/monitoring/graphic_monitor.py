import numpy as np
import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from ...utils.bbox_metrics import overlap
from ...utils.allocentric_spatial_relations import is_on_top, is_included, is_close,distance
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
from pyuwds3.utils.heatmap import Heatmap
from std_msgs.msg import Int8

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

from pyuwds3.reasoning.simulation.internal_simulator import InternalSimulator

from  pr2_motion_tasks_msgs.msg import RobotAction



import pybullet as p

INF = 10e3
#Ray >1
N_RAY = 8
ALPHA_THRESHOLD = 0.6

PICK_DIST=5
DELTA_TIME=0.5

FILTERING_Y = 15
FILTERING_Z = 20
MIN_VEL = 0.001
MIN_ANG = 0.001

def centroid_cost(track_a, track_b):
    """Returns the centroid cost"""
    try:
        return euclidean(track_a.pose.position().to_array(), track_b.pose.position().to_array())
    except Exception:
        return INF
class AgentType(object):
    HUMAN = "human"
    ROBOT = "robot"

class GraphicMonitor(Monitor):
    """ Special monitor for agent management
    """
    def __init__(self,agent=None,agent_type =AgentType.ROBOT,
     handL = None,handR=None, head = "head_mount_kinect2_rgb_optical_frame", internal_simulator=None,
       position_tolerance=0.04,name="robot" ): #beliefs_base=None,

        super(GraphicMonitor, self).__init__(internal_simulator=internal_simulator)#, beliefs_base=beliefs_base)

        # filteringvalue
        self.filtering_y_axis = rospy.get_param("~filtering_y_axis", FILTERING_Y)
        self.filtering_z_axis = rospy.get_param("~filtering_z_axis", FILTERING_Z)
        self.minimum_velocity = rospy.get_param("~minimum_velocity", MIN_VEL)
        self.minimum_angular_velocity = rospy.get_param("~minimum_angular_velocity", MIN_ANG)
        self.name = name
        self.myself = SceneNode(agent = True,label=agent_type)
        self.myself.id = name
        #init of the simulator and ontolog
        self.internal_simulator = internal_simulator
        self.simulator_id = self.internal_simulator.client_simulator_id
        self.ontologies_manip = OntologiesManipulator()
        self.global_frame_id = rospy.get_param("~global_frame_id")
        self.base_frame_id = rospy.get_param("~base_frame_id", "odom")
        self.world_publisher = WorldPublisher("corrected_tracks42_"+self.name, self.global_frame_id)
        self.marker_publisher = MarkerPublisher("ar_perception_marker")

        self.onto_bridge = OntologeniusReaderNode(name)
        self.ontologies_manip.add("robot")
        self.onto=self.ontologies_manip.get("robot")
        self.onto.close()

        #init of the camera (kinect fov)
        self.camera = Camera(640)
        self.camera.setfov(84.1,53.8)


        #init of the camera (fileterd fov)
        self.filtered_camera = Camera(640)
        self.filtered_camera.setfov(2*self.filtering_y_axis,2*self.filtering_y_axis)

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
        self.agent_monitor_map={}
        self.agent_map={}
        self.marker_pub={}
        self.movement_pub={}
        if self.agent_type==AgentType.ROBOT:
            self.marker_pub[name]=MarkerPublisher("ar_perception_markers"+str(name))
            self.movement_pub[name]=rospy.Publisher('movement_pub_robot', Int8, queue_size=10)


        #view of robot and human
        self._publisher = ViewPublisher(name+"_view")
        self.publisher_map={}
        #tf subscription

        #heatmap
        self.heatmap={}
        self.heatmap[self.name]=Heatmap()
        # dictionnary of the picked objects by the robot
        self.pick_map = {}
        self.last_head_pose = {}
        self.last_time_head_pose={}

        self.time_max=0
        self.number_iteration=0
        # dictionary of the frame that are abble to grasp/pick obj
        self.grasp_map = {}

        #mocap dicttionary, and object to publish dictionnary
        self.mocap_obj={}
        self.mocap_body={}
        self.publish_dic={}

        #time, and fps limit a a
        self.time_monitor=rospy.Time().now().to_sec()
        self.time_view=rospy.Time().now().to_sec()
        self.n_frame_monitor=15
        self.n_frame_view = 15

        #object being picke by a human:

        #   - hand_close-map : hand was close, checking if hand is going inside
        #   - may_be_picked_map : hand wa never inside, waiting to see if diseapear/moved
        #   - picked_map : obejct that were picked
        self.hand_close_map={}
        self.may_be_picked_map={}
        self.picked_map={}

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
                del self.pick_map[self.objId]


    def get_head_pose(self,time):
        """
            #get the head pose in simulator, so in the global_frame_id (often map)
        """
        if not self.simulator.is_entity_loaded(self.head):
            s,hpose=self.simulator.tf_bridge.get_pose_from_tf(self.simulator.global_frame_id,
                                                            self.head)
        else:
            hpose = self.simulator.get_entity(self.head).pose
        return hpose

    def pos_validityv2(self,mvect,header,pos=None):
        mpose = Vector6DStable(mvect.pos.x,mvect.pos.y,mvect.pos.z)
        frame_id = header.frame_id
        if frame_id[0]=='/':
            frame_id = frame_id[1:]
        if pos ==None:
            bool_,head_pose = self.tf_bridge.get_pose_from_tf("head_mount_kinect2_rgb_link" ,
                                          frame_id,header.stamp)
        else:
            head_pose=pos
        mpose.from_transform(np.dot(head_pose.transform(),mpose.transform()))
        #mpose is now in the head frame
        if mpose.pos.x==0:
            return False,None
        xy_angle = np.degrees(np.arctan(mpose.pos.y/mpose.pos.x))
        xz_angle = np.degrees(np.arctan(mpose.pos.z/mpose.pos.x))

        return (abs(xy_angle)<self.filtering_y_axis and
               abs(xz_angle)<self.filtering_z_axis),

    def update_heatmap(self,nodes,agent_id,time,view_pose=None):
        if not agent_id in self.heatmap:
            self.heatmap[agent_id]=Heatmap()
        to_send=[]
        for n in nodes:
            h=rospy.Header()
            h.frame_id='map'
            if self.pos_validityv2(n.pose,h,view_pose):
                to_send.append(n)
        self.heatmap[agent_id].heat(to_send,time)




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
            # print time-self.time_view
            self.time_view=time
            if self.agent_type== AgentType.ROBOT:
                for obj_id in self.mocap_obj.keys():
                    if not obj_id in self.publish_dic:
                        self.publish_dic[obj_id]=WorldPublisher(str(obj_id)+"_tracks", self.global_frame_id)
                    if not obj_id in self.marker_pub:
                        self.marker_pub[obj_id]=MarkerPublisher("ar_perception_markers"+str(obj_id))

                for obj_id in self.agent_monitor_map.keys():
                    view_pose=self.mocap_obj[obj_id].pose + Vector6DStable(0.15,0,0,0,np.pi/2)

                    img, _, _, nodes = self.simulator.get_camera_view(view_pose, self.camera,occlusion_threshold=0.001 )
                    # if obj_id == "Helmet_2":
                    # for i in nodes:
                    #     print i.id
                    self.update_heatmap(nodes,obj_id,time,view_pose)

                    if obj_id in self.publisher_map:
                        self.publisher_map[obj_id].publish(img,[],rospy.Time.now())
                    else:
                        self.publisher_map[obj_id] = ViewPublisher(obj_id+"_view")
                    node_to_keep=[self.mocap_obj[obj_id]]
                    if obj_id + "_body" in self.mocap_body:
                        node_to_keep.append(self.mocap_body[obj_id+"_body"])
                    for node in nodes:
                        if node.label !="appartment" and node.label!= "robot" and not "elmet" in node.label:
                            #we remove the background
                            node.last_update = self.mocap_obj[obj_id].last_update
                            node_to_keep.append(node)
                        if node.label== "robot":
                            # if node.agent==True:
                            #     node.last_update = self.mocap_obj[obj_id].last_update
                            #     node_to_keep.append(node)
                            # else:
                            #     node_ = SceneNode(pose=self.get_head_pose(),agent = True,label="robot")
                            #     node_.last_update = self.mocap_obj[obj_id].last_update
                            #     node_to_keep.append(node_)
                            scene_node = self.internal_simulator.get_entity(self.internal_simulator.my_id)
                            scene_node.agent=True
                            scene_node.id="robot"
                            scene_node.last_update = self.mocap_obj[obj_id].last_update
                            node_to_keep.append(scene_node)
                    header.stamp=self.mocap_obj[obj_id].last_update
                    # self.publish_dic[obj_id].publish(nodes,[],header)
                    self.agent_monitor_map[obj_id].monitor(node_to_keep,Vector6DStable(),header)
                    self.marker_pub[obj_id].publish(node_to_keep,header)


            # self.internal_simulator.step_simulation()
        # # print self.internal_simulator.entity_id_map
        # self.frame_count %= self.n_frame

        #     # self.time=time
        #     # self.head.pose.rot.x+=0.03
        #     # print self.head.pose.rot.x
            # self.head.pose.rot.y+=0.06
            # print self.head.pose.rot.y
            # print self.head.pose.pos
            # print "   "

    def create_internal_sim(self):
        simulation_config_filename = rospy.get_param("~simulation_config_filename", "")
        cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")
        static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")
        robot_urdf_file_path = rospy.get_param("~robot_urdf_file_path", "")
        internal_simulator = InternalSimulator(False,
                                                    simulation_config_filename,
                                                    cad_models_additional_search_path,
                                                    static_entities_config_filename,
                                                    robot_urdf_file_path,
                                                    self.global_frame_id,
                                                    self.base_frame_id,
                                                    load_robot = True,
                                                    update_robot_at_each_step = False)
        return internal_simulator

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
                if not "_body" in object.id and not "pick" in object.id:
                    self.agent_map[object.id]=object
                    if not object.id in self.agent_monitor_map:
                        self.onto.feeder.addConcept(object.id)
                        self.onto.feeder.addInheritage(object.id,"human")
                        self.agent_monitor_map[object.id]=GraphicMonitor(agent=None,agent_type =AgentType.HUMAN,
                         handL = None,handR=None, head = object.id, internal_simulator=self.create_internal_sim(),
                           position_tolerance=0.04,name= object.id)
                    self.mocap_obj[object.id]=object
                    self.simulator.reset_entity_pose(object.id, object.pose)
                if "_body" in object.id:
                    self.mocap_body[object.id]=object
                if "pick" in object.id:
                    self.grasp_map[object.id]=object

                # self.human_pose=object.pose
                # self.simulator.change_joint(object.id,0,2)
                # print p.getVisualShapeData(self.simulator.entity_id_map[object.id])
                # p.resetJointState(self.simulator.entity_id_map[object.id],0,0.1)


    #main element :

    def monitor(self, object_tracks, pose, header):
        """ Monitor the physical consistency of the objects and compute fact

        """
        #place all object of the tracks in the simulator
        # print "=================="
        # print self.relations
        # print self.relations
        # print "=============="
        time = header.stamp
        self.cleanup_relations(time)
        check_missing_object = False
        node_seen = []
        # print time.to_sec()
        # print self.time_monitor
        self.time_max=+time.to_sec()-self.time_monitor
        self.number_iteration+=1.0
        # if self.agent_type== AgentType.ROBOT:
        #     print self.name + " : " + str(self.time_max/self.number_iteration)

        if self.agent_type== AgentType.ROBOT and abs(time.to_sec()-self.time_monitor) >1./(self.n_frame_monitor):
            hpose=self.get_head_pose(time)
            # print hpose
            image,_,_,nodes =  self.simulator.get_camera_view(hpose, self.camera,occlusion_threshold=0.001)
            self.update_heatmap(nodes,self.name,time.to_sec(),hpose)
            for k in object_tracks:
                self.heatmap[self.name].color_node(k)
            self.marker_pub[self.name].publish(object_tracks,header)
            self._publisher.publish(image,[],time)
            # print "ooooooo"
            # for i in nodes:
            #     print i.id
            # print "pppppppp"
            self.time_monitor=time.to_sec()
            check_missing_object = True
            for node in nodes:
                node_seen.append(node.id)



        if pose != None:
            for object in object_tracks:
                if object.is_located() and object.has_shape() and object.label!="robot":
                    # if check_missing_object == True:
                    #     if object.id in node_seen:
                    #         if (time -object.last_update).to_sec()>DELTA_TIME:
                    #             self.missing_object[object.id]=object.last_update

                            # if object should be seen but are not, they are missing

                    object.pose.from_transform(np.dot(pose.transform(),object.pose.transform()))
                    # if object.id in self.missing_object:
                    #     if (object.last_update == self.missing_object[object.id]):
                    #         object.pose.pos.z =object.pose.pos.z - 100
                    #     else:
                    #         del( self.missing_object[object.id])
                    if not self.simulator.is_entity_loaded(object.id):
                        self.simulator.load_node(object)
                    base_link_sim_id = self.simulator.entity_id_map[object.id]
                    self.simulator.reset_entity_pose(object.id, object.pose)
                if object.label=="robot":
                    self.simulator.load_robot=True
                if object.agent:
                    self.agent_map[object.id]=object
            # self.marker_publisher.publish(object_tracks,header)

        # self.list.append(time.to_sec()-self.time_monitor)
        # print time.to_sec()-self.time_monitor
        # print self.list
        # print (sum(self.list)/(1.0*len(self.list)))
        #publish the head view




        #compute the facts
        # if rospy.Time().now().to_sec()<1607675257.84:
        self.compute_allocentric_relations(object_tracks, time)
        if self.agent_type== AgentType.ROBOT:
            for obj_id in self.mocap_obj.keys():
                if not obj_id in self.movement_pub:
                    self.movement_pub[obj_id]=rospy.Publisher('movement_pub_'+str(obj_id), Int8, queue_size=10)
                mv=self.movement_validity(obj_id,self.mocap_obj[obj_id].pose,header)
                result=Int8()
                result.data=int(mv)
                self.movement_pub[obj_id].publish(result)


            mv=self.movement_validity(self.name,self.get_head_pose(time),header)
            result=Int8()
            result.data=int(mv)
            self.movement_pub[self.name].publish(result)
        # print self.relations_index
        # self.compute_egocentric_relations(object_tracks+self.agent_map.values(), time)

        # self.pick(object_tracks,time,node_seen)
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

        # if self.name!= "robot":
        #     print "oooooooooooo"
        #     for i in object_tracks:
        #         print i.id
        #     print "ppppppppppppp"

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
    def movement_validity(self,name,head_pose,header):

        if not name in self.last_head_pose:
            self.last_head_pose[name] = head_pose
            self.last_time_head_pose[name] = header.stamp
        vel_movement = 0
        ang_movement = 0
        delta= header.stamp-self.last_time_head_pose[name]
        #If we are on a different time frame : (the head might have moved)
        if header.stamp != self.last_time_head_pose[name]:
            vel_movement = np.linalg.norm(
                                    head_pose.pos.to_array() -
                                    self.last_head_pose[name].pos.to_array() )/delta.to_sec()
            ang_movement = np.linalg.norm(
                                    head_pose.rot.to_array() -
                                    self.last_head_pose[name].rot.to_array() )/delta.to_sec()



            self.last_time_head_pose[name] = header.stamp
            self.last_head_pose[name] = head_pose
        return ((vel_movement> self.minimum_velocity) or
          ang_movement> self.minimum_angular_velocity)


    def canSee(self,start_pose_vector6,obj):
        """ compute if obj can be seen from start pose"""
        end_id=self.simulator.entity_id_map[obj.id]
        start_pose = start_pose_vector6.pos.to_array()[:3]
        start_pose=[start_pose[0][0],start_pose[1][0],start_pose[2][0]]
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
        # print r
        # print r[0]
        # print end_id
        if r[0][0] == end_id:
            return True
        if r[0][0]==-1:
            return False
        if not (r[0][0],r[0][1]) in self.alpha_dic:
            data = p.getVisualShapeData(r[0][0])

            if r[0][1] +1 >0 and r[0][1] +1 <len(data) and data[r[0][1] +1][1] == r[0][1]:
                self.alpha_dic[(r[0][0],r[0][1])] =data[ r[0][1] +1][7]
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
#             if obj1.is_located() and obj1.has_shape() and obj1.label!="human":
#                 for obj2 in objects:
#                     if obj1.id != obj2.id:
#                         # evaluate allocentric relation
#                         if obj2.is_located() and obj2.has_shape() and obj2.label!="human":
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
    #             if obj1.is_located() and obj1.has_shape() and obj1.label!="human":
    #                 for obj2 in objects:
    #                     if obj1.id != obj2.id:
    #                         # evaluate allocentric relation
    #                         if obj2.is_located() and obj2.has_shape() and obj2.label!="human":
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


    def compute_egocentric_relations(self,objects,time):
        """ compute the egocentric relations (can see can reach)"""
        self.myself.pose = self.get_head_pose(time)
        self.agent_map[self.name]=self.myself
        for agent in self.agent_map.values():
            for obj1 in objects:
                if obj1.is_located() and obj1.has_shape() and obj1.label!="robot" and obj1.id != agent.id:
                    if agent.id == "Helmet_2" and "able" in obj1.id:
                        print self.canSee(agent.pose,obj1)
                    if self.canSee(agent.pose,obj1):
                        self.start_fact(agent, "canSee",object=obj1, time=time)
                    else:
                        self.end_fact(agent, "canSee",object=obj1, time=time)

                    if self.can_reach(agent.pose,obj1):
                        self.start_fact(agent, "CanReach",object=obj1, time=time)
                    else:
                        self.end_fact(agent, "CanReach",object=obj1, time=time)


    def pick(self,obj_list,time,node_seen):
        for obj in obj_list:
            if "Pickable" in self.onto.individuals.getUp(obj.id):
                for hand in self.grasp_map.values():
                    has_pick(hand,obj,time,node_seen)

    def has_pick(self,hand,obj,time,node_seen):
        hand_pose = hand.pose.pos.to_array()[:3]
        hand_aabb=[
        [hand_pose[0]-2,hand_pose[1]-2,hand_pose[2]-2],
        [hand_pose[0]+2,hand_pose[1]+2,hand_pose[2]+2]]
        success, obj_aabb = self.simulator.get_aabb(obj)
        if not success:
            return
        if obj.id in self.picked_map:
            return
        # if obj.id in self.picked_map:
        #     time =  self.picked_map[obj.id]
        #     # if obj.last_update

        if obj.id in self.may_be_picked_map:
            old_time,old_pose = self.may_be_picked_map[obj.id]
            if obj.id in self.node_seen:
                del self.may_be_picked_map[obj.id]
                if obj.last_update ==old_time:
                    self.picked_map[obj.id]=old_time,old_pose
                else:
                    vect = obj.old_pose.pos - obj.pose.pos
                    dist = np.linalg(vect.to_array()[:3])
                    if dist> MOVED_DIST:
                        self.picked_map[obj.id]=old_time,old_pose
                        # AND DROP!!!
            return

        if obj.id in self.hand_close_map:
            computation_old_time,obj_old_time,old_pose = self.hand_close_map[obj.id]
            if is_included(hand_aabb,obj_aabb):
                del self.hand_close_map[obj.id]
            else:
                if time-computation_old_time >DELTA_TIME:
                    self.may_be_picked_map[obj.id]=(obj_old_time,old_pose)
                    del self.hand_close_map[obj.id]
            return

        if distance(obj_aabb,hand_aabb)<PICK_DIST:
            self.may_be_picked_map[obj.id]=(time,obj_old_time,old_pose)


                        # obj_pose,time =  self.picked_map[obj.id]





    def compute_allocentric_relations(self, objects, time):
        included_map={}
        #included_map[a] = [b,c,d] <=> a is in b in c and in d
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape() and obj1.label!="human" :
                included_map[obj1.id]=[]
                for obj2 in objects:
                    if obj1.id != obj2.id:
                        # evaluate allocentric relation
                        if obj2.is_located() and obj2.has_shape() and obj2.label!="human":
                            # get 3d aabb
                            success1, aabb1 = self.simulator.get_aabb(obj1)
                            success2, aabb2 = self.simulator.get_aabb(obj2)
                            if success1  and success2 :
                                if (not (obj1.id in self.pick_map)) and (not (obj2.id in self.pick_map)):

                                    if obj1.id+"in"+obj2.id in self.relations_index:
                                        hyst=0
                                    else:
                                        # print self.name +": "+ obj1.id+"in"+obj2.id
                                        hyst=0
                                    # if obj1.id=="cube_BGTG" and obj2.id=="box_C4":
                                    #     print self.name
                                    #     print is_included(aabb1,aabb2,0)
                                    #     print " "
                                    #     print "==========================================================="
                                    is_hm=False
                                    if self.name in self.heatmap:
                                        hm=self.heatmap[self.name].heatm
                                        if obj1.id in hm and obj2.id in hm:
                                            if hm[obj2.id]!=0:
                                                k=hm[obj1.id]/hm[obj2.id]
                                                if k>0.9 and k<1.1:
                                                    is_hm=True
                                    if is_hm and is_included(aabb1, aabb2,hyst):
                                        self.start_fact(obj1, "in", object=obj2, time=time)
                                        included_map[obj1.id].append(obj2.id)

                                    else:
                                        self.end_fact(obj1, "in", object=obj2, time=time)
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape() and obj1.label!="human":
                for obj2 in objects:
                    if obj1.id != obj2.id:
                        # evaluate allocentric relation
                        if obj2.is_located() and obj2.has_shape() and obj2.label!="human":
                            # get 3d aabb
                            # print included_map

                            success1, aabb1 = self.simulator.get_aabb(obj1)
                            success2, aabb2 = self.simulator.get_aabb(obj2)
                            if success1  and success2 :
                                is_hm=False
                                if self.name in self.heatmap:
                                    hm=self.heatmap[self.name].heatm
                                    if obj1.id in hm and obj2.id in hm:
                                        if hm[obj2.id]!=0:
                                            k=hm[obj1.id]/hm[obj2.id]
                                            if k>0.9 and k<1.1:
                                                is_hm=True
                                if is_hm and included_map[obj1.id]==included_map[obj2.id] and (not (obj1.id in self.pick_map)) and (not (obj2.id in self.pick_map)):
                                    if obj1.id+"on"+obj2.id in self.relations_index:
                                        hyst=0
                                    else:
                                        hyst=0

                                    if  is_on_top(aabb1, aabb2,hyst) :
                                        self.start_fact(obj1, "on", object=obj2, time=time)
                                    else:
                                        self.end_fact(obj1, "on", object=obj2, time=time)
