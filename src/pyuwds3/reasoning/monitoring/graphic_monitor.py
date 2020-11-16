import numpy as np
import rospy
import cv2
from ..assignment.linear_assignment import LinearAssignment
from .monitor import Monitor
from ...utils.bbox_metrics import overlap
from ...utils.allocentric_spatial_relations import is_on_top, is_in, is_close
from ...utils.egocentric_spatial_relations import is_right_of, is_left_of
from ...types.camera import Camera
from scipy.spatial.distance import euclidean
from pyuwds3.types.vector.vector6d_stable import Vector6DStable
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.shape.mesh import Mesh
from .physics_monitor import ActionStates
from pyuwds3.utils.view_publisher import ViewPublisher
import math
import tf
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import tf2_ros
import tf2_geometry_msgs
from ...types.vector.vector6d import Vector6D
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
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
     hand1 = None,hand2=None, head = "head_mount_kinect2_rgb_optical_frame", internal_simulator=None,
      beliefs_base=None, position_tolerance=0.04,name="robot"):
        """ Tabletop monitor constructor
        """
        super(GraphicMonitor, self).__init__(internal_simulator=internal_simulator, beliefs_base=beliefs_base)
        self.internal_simulator = internal_simulator
        # self.onto = OntologyManipulator("")
        self.previous_object_states = {}
        self.camera = Camera()
        self.previous_object_tracks_map = {}
        self.position_tolerance = position_tolerance
        self.content_map = {}
        self.centroid_assignement = LinearAssignment(centroid_cost, max_distance=None)
        self.alpha_dic = {}
        self.agent_type = agent_type
        self.agent = agent
        self.hand1 = hand1
        self.hand2 = hand2
        self.head  = head
        self.n_frame=4
        self.frame_count = 0
        self._publisher = ViewPublisher(name+"_view")
        self.ar_tags_sub = rospy.Subscriber("/tf", rospy.AnyMsg, self.publish_view)


        # node = SceneNode(pose=Vector6DStable(-1,-1,1))
        # self.cad_models_search_path = rospy.get_param("~cad_models_search_path", "")
        # mesh_path = self.cad_models_search_path + "/obj/dt_cube.obj"
        # shape = Mesh(mesh_path,
        #              x=-1, y=-1, z=1,
        #              rx=0, ry=0, rz=0)
        # shape.color[0] = 1
        # shape.color[1] = 1
        # shape.color[2] = 1
        # shape.color[3] = 1
        # node.shapes.append(shape)
        # self.internal_simulator.load_node(node)
        self.time=rospy.Time().now().to_nsec()

    def get_head_pose(self,time):
        s,hpose=self.simulator.tf_bridge.get_pose_from_tf(self.simulator.global_frame_id,
                                                        self.head,time)
        return hpose

    def publish_view(self,tfm):
        time = rospy.Time.now().to_nsec()
        # if time-self.time > 7166666:
        # #     # print "test"
        # #     self.time=time
        # #     self.internal_simulator.step_simulation()
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


        self.frame_count+=1


    def monitor(self, object_tracks, pose, time):
        """ Monitor the physical consistency of the objects and detect human tabletop actions
        """
        self.cleanup_relations()
        next_object_states = {}
        object_tracks_map = {}
        if pose != None:
            for object in object_tracks:
                object.pose.from_transform(np.dot(pose.transform(),object.pose.transform()))

                if object.is_located() and object.has_shape():
                    if not self.simulator.is_entity_loaded(object.id):
                        self.simulator.load_node(object)
                    self.simulator.reset_entity_pose(object.id, object.pose)

        if time.to_nsec()-self.time > 7166666:
            hpose=self.get_head_pose(time)
            print hpose
            image,_,_,_ =  self.simulator.get_camera_view(hpose, self.camera)
            self._publisher.publish(image,[],time)
            self.time=time.to_nsec()
        # for object in object_tracks:
        #     if object.is_located() and object.has_shape():
        #         if object.is_confirmed():
        #             simulated_object = self.simulator.get_entity(object.id)
        #
        #             object_tracks_map[object.id] = simulated_object
        #             # compute scene node input
        #             simulated_position = simulated_object.pose.position()


        # self.compute_allocentric_relations(object_tracks, time)


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

    def can_reach(self,start_pose,end_id,working_area,xy_n=10,z_n=5):
        x_init = working_area[0][0]
        y_init = working_area[0][1]
        z_init = working_area[0][2]
        x_step = (working_area[1][0] - x_init)/(xy_n*1.0)
        y_step = (working_area[1][1] - y_init)/(xy_n*1.0)
        z_step = (working_area[1][2] - z_init)/(z_n*1.0)
        for x in range(xy_n):
            for y in range(xy_n):
                for z in range(z_n):
                    if can_reach_rot(self,
                    [x_init + x*x_step,
                    y_init + y*y_step,
                    z_init + z*z_step],end_id):
                        return True

        return False
    def can_reach_rot(self,start_pose,end_id):
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

    def canSee(self,start_pose,end_id):
        [xmin,ymin,zmin],[xmax,ymax,zmax] = p.getAABB(end_id)
        xlength = xmax - xmin
        ylength = ymax - ymin
        zlength = zmax - zmin
        pose_list = []
        for i in range(N_RAY):
            end_pose = [xmin +i*xlength/(N_RAY-1),
            ymin +i*xlength/(N_RAY-1),
            zmin +i*xlength/(N_RAY-1)]
            if self.canSeeRec(start_pose,end_pose,end_id,0):
                return True
        return False

    def canSeeRec(self,start_pose,end_pose,end_id,hitnumber):
        print("here")
        r=p.rayTestBatch([start_pose],[end_pose],reportHitNumber = hitnumber)
        print (r)
        if r[0][0] == end_id:
            return True
        if r[0][0]==-1:
            return False
        if not (r[0][0],r[0][1]) in self.alpha_dic:
            data = p.getVisualShapeData(r[0][0],r[0][1])
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


    def compute_allocentric_relations(self, objects, time):


        print self.relations_index
        for obj1 in objects:
            if obj1.is_located() and obj1.has_shape():
                success1, aabb1 = self.simulator.get_aabb(obj1)
                if success1:
                    is_on = True
                    is_in = True
                    on = self.onto.indivividuals.getOn(obj1.id,"onTopOf")
                    in_ = self.onto.indivividuals.getOn(obj1.id,"isInContainer")

                    if len(on)>0:
                        if on[0].is_located() and on[0].has_shape():
                            success2, aabb2 = self.simulator.get_aabb(on[0])
                            if success2:
                                is_on = is_on_top(aabb1, aabb2)
                                # if is_on == False:
                                    #DELETE LINK IN ONTO
                    if len(in_)>0:
                        if in_[0].is_located() and in_[0].has_shape():
                            success2, aabb2 = self.simulator.get_aabb(in_[0])
                            if success2:
                                is_in = is_in(aabb1, aabb2)
                                # if is_on == False:
                                    #DELETE LINK IN ONTO
                if not (is_on and is_in):
                    for obj2 in objects:
                        if obj2 != obj1:
                            success2, aabb2 = self.simulator.get_aabb(obj2)
                            if success2:
                                if not is_on:
                                    is_on = is_on_top(aabb1, aabb2)
                                    if is_on:
                                        self.onto.feeder.insert(obj1.id,"onTopOf",obj2.id)
                                if not is_in:
                                    is_in = is_in(aabb1, aabb2)
                                    if is_in:
                                        self.onto.feeder.insert(obj1.id,"isInContainer",obj2.id)
                for obj2 in objects:
                    if obj1.id != obj2.id:
                        # evaluate allocentric relation
                        if obj2.is_located() and obj2.has_shape():
                            # get 3d aabb
                            success1, aabb1 = self.simulator.get_aabb(obj1)
                            success2, aabb2 = self.simulator.get_aabb(obj2)
