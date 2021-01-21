#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_msgs.msg import WorldStamped
from uwds3_msgs.srv import Query
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.situations import Fact
from ontologenius import OntologyManipulator, OntologiesManipulator


class OntologeniusReaderNode(object):
    """ The ontologenius reader which allow to query uwds3 entities with SPARQL-like queries """
    def __init__(self,onto_name="robot"):
        """ Default constructor """
        self.onto_name = onto_name
        self.ontologies_manip = OntologiesManipulator()
        self.ontologies_manip.add(self.onto_name)
        self.ontologenius_client =self.ontologies_manip.get(self.onto_name)
        self.ontologenius_client.close()
        rospy.loginfo("[ontologenius_reader] Connected to Ontologenius !")

        self.created_nodes = {}
        self.scene_nodes = {}
        self.created_situations = {}

        self.uwds3_oro_id_map = {}
        self.oro_uwds3_id_map = {}

        self.support = {}
        self.graspable = {}
        self.container = {}
        self.movable = {}
        self.held_by = {}
        self.relations = {}

        # self.query_service = rospy.Service("uwds3/query_knowledge_base", Query, self.handle_query)

        input_world_topic = rospy.get_param("~input_world_topic", "corrected_tracks")
        rospy.loginfo("[ontologenius_reader] Connecting to '" + input_world_topic +  "'...")
        self.world_subscriber = rospy.Subscriber(input_world_topic, WorldStamped, self.callback, queue_size=1)
        rospy.loginfo("[ontologenius_reader] Connected to Underworlds !")

    def handle_query(self, req, agent="myself"):
        """ Handle the query """
        try:
            query_tokens = req.query.split(",")
            #first_variable = query_tokens[0].split(" ")[0]
            query = req.query
            result_nodes = []
            if len(query_tokens) > 1:
                results = self.ontologenius_client.sparql.call(query)
            else:
                results = self.ontologenius_client.sparql.call(query)
            rospy.logwarn(results)
            if len(results) > 0:
                for node_id in results[0]:
                    if node_id in self.scene_nodes:
                        result_nodes.append(self.scene_nodes[node_id])
                return result_nodes, True, ""
            else:
                return [], True, ""
        except Exception as e:
            rospy.logwarn("[ontologenius_reader] Exception occurred : "+str(e))
            return [], False, str(e)

    def callback(self, world_msg):
        """ World callback """
        scene_nodes = {}
        for node_msg in world_msg.world.scene:
            node = SceneNode().from_msg(node_msg)
            scene_nodes[node.id] = node
            if node.id not in self.created_nodes:
                self.add_scene_node(node)
                self.created_nodes[node.id] = True

        for situation_msg in world_msg.world.timeline:
            situation = Fact().from_msg(situation_msg)
            # self.learn_affordance(situation)
            self.update_situation(situation)

        self.scene_nodes = scene_nodes

    def add_scene_node(self, scene_node, agent="myself"):
        """ Add the given scene node """
        types = []

        if scene_node.label == "myself":
            types.append("Robot")
        elif scene_node.label == "person":
            types.append("Human")
        else:
            if scene_node.has_shape():
                types.append("SolidObject")
            elif scene_node.is_located():
                types.append("SpatialThingLocated")
            else:
                types.append("PartiallyTangibleThing")

        for type in types:
            rospy.loginfo("add: "+scene_node.id+" isA "+type)
            self.ontologenius_client.feeder.addObjectProperty(scene_node.id, "isA", type)
        rospy.loginfo("add: "+scene_node.id+" hasName "+scene_node.description)
        self.ontologenius_client.feeder.addObjectProperty(scene_node.id, "hasName", scene_node.description)

    def update_situation(self, situation, agent="myself"):
        """ Updates the given situations """
        if situation.predicate == "pick":
            if situation.object not in self.held_by:
                self.ontologenius_client.feeder.addObjectProperty(situation.object, "isHeldBy", situation.subject)
                self.held_by[situation.object] = situation.subject
        elif situation.predicate == "place":
            if situation.object in self.held_by:
                self.ontologenius_client.feeder.removeObjectProperty(situation.object, "isHeldBy", situation.subject)
                del self.held_by[situation.object]
        elif situation.predicate == "release":
            if situation.object in self.held_by:
                self.ontologenius_client.feeder.removeObjectProperty(situation.object, "isHeldBy", situation.subject)
                del self.held_by[situation.object]

        if situation.predicate == "in":

            if not situation.is_finished():
                if situation.subject+"isInside"+situation.object not in self.relations:
                    rospy.loginfo("add: "+situation.subject+" isInside "+situation.object)
                    self.ontologenius_client.feeder.addObjectProperty(situation.subject, "isInContainer", situation.object,situation.start_time)
                    self.relations[situation.subject+"isInside"+situation.object] = True
            else:
                if situation.subject+"isInside"+situation.object in self.relations:
                    rospy.loginfo("remove: "+situation.subject+" isInside "+situation.object)
                    self.ontologenius_client.feeder.removeObjectProperty(situation.subject, "isInContainer", situation.object,situation.end_time)
                    del self.relations[situation.subject+"isInside"+situation.object]
        elif situation.predicate == "on":
            if not situation.is_finished():
                if situation.subject+"isOnTop"+situation.object not in self.relations:
                    rospy.loginfo("add: "+situation.subject+" isOnTop "+situation.object)
                    self.ontologenius_client.feeder.addObjectProperty(situation.subject, "isOnTopOf", situation.object,situation.start_time)
                    self.relations[situation.subject+"isOnTop"+situation.object] = True
            else:
                if situation.subject+"isOnTop"+situation.object in self.relations:
                    rospy.loginfo("remove: "+situation.subject+" isOnTop "+situation.object)
                    self.ontologenius_client.feeder.removeObjectProperty(situation.subject, "isOnTopOf", situation.object,situation.end_time)
                    del self.relations[situation.subject+"isOnTop"+situation.object]
        elif situation.predicate == "canSee":
            if not situation.is_finished():
                if situation.subject+"canSee"+situation.object not in self.relations:
                    rospy.loginfo("add: "+situation.subject+" canSee "+situation.object)
                    self.ontologenius_client.feeder.addObjectProperty(situation.subject, "canSee", situation.object,situation.start_time)
                    self.relations[situation.subject+"canSee"+situation.object] = True
            else:
                if situation.subject+"canSee"+situation.object in self.relations:
                    rospy.loginfo("remove: "+situation.subject+" canSee "+situation.object)
                    self.ontologenius_client.feeder.removeObjectProperty(situation.subject, "canSee",situation.object, situation.end_time)
                    del self.relations[situation.subject+"canSee"+situation.object]
        elif situation.predicate == "CanReach":
            if not situation.is_finished():
                if situation.subject+"CanReach"+situation.object not in self.relations:
                    rospy.loginfo("add: "+situation.subject+" CanReach "+situation.object)
                    self.ontologenius_client.feeder.addObjectProperty(situation.subject, "CanReach",situation.object, situation.start_time)
                    self.relations[situation.subject+"CanReach"+situation.object] = True
            else:
                if situation.subject+"CanReach"+situation.object in self.relations:
                    rospy.loginfo("remove: "+situation.subject+" CanReach "+situation.object)
                    self.ontologenius_client.feeder.removeObjectProperty(situation.subject, "CanReach",situation.object, situation.end_time)
                    del self.relations[situation.subject+"CanReach"+situation.object]
    # elif situation.predicate == "close":
    #     if not situation.is_finished():
    #         if situation.subject+"isCloseTo"+situation.object not in self.relations:
    #             rospy.loginfo("add: "+situation.subject+" isCloseTo "+situation.object)
    #             self.ontologenius_client.feeder.addObjectProperty(situation.subject, "isCloseTo", situation.object_time)
    #             self.relations[situation.subject+"isCloseTo"+situation.object] = True
    #     else:
    #         if situation.subject+"isCloseTo"+situation.object in self.relations:
    #             rospy.loginfo("remove: "+situation.subject+" isCloseTo "+situation.object)
    #             self.ontologenius_client.feeder.removeObjectProperty(situation.subject, "isCloseTo", situation.object_time)
    #             del self.relations[situation.subject+"isCloseTo"+situation.object]
        else:
            pass

    # def learn_affordance(self, situation, agent="myself"):
    #     """ Learn the affordances from the physical reasoner """
    #     if situation.predicate == "pick":
    #         if situation.object not in self.graspable:
    #             rospy.loginfo("add: "+situation.object+" isA "+"GraspableObject")
    #             self.ontologenius_client.feeder.addObjectProperty(situation.subject, "isA", "GraspableObject")
    #             self.graspable[situation.object] = True
    #     elif situation.predicate == "on":
    #         if situation.object not in self.support:
    #             rospy.loginfo("add: "+situation.object+" isA "+"SupportObject")
    #             self.ontologenius_client.feeder.addObjectProperty(situation.object, "isA", "SupportObject")
    #             self.support[situation.object] = True
    #     elif situation.predicate == "in":
    #         if situation.object not in self.container:
    #             rospy.loginfo("add: "+situation.object+" isA "+"ContainerObject")
    #             self.ontologenius_client.feeder.addObjectProperty(situation.object, "isA", "ContainerObject")
    #             self.container[situation.object] = True

#     def run(self):
#         """ Run the component """
#         while not rospy.is_shutdown():
#             rospy.spin()
#
#
# if __name__ == "__main__":
#     rospy.init_node("ontologenius_reader", anonymous=True)
#     recorder = OntologeniusReaderNode().run()
