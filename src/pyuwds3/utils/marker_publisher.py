import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3


class MarkerPublisher(object):
    def __init__(self, topic_name, queue_size=1, alpha=1.0):
        self.publisher = rospy.Publisher(topic_name, MarkerArray, queue_size=1)
        self.alpha = alpha
        self.marker_id_map = {}
        self.last_marker_id = 0

    def publish(self, tracks, header):
        markers_msg = MarkerArray()
        for track in tracks:
            if track.is_confirmed():
                if track.has_shape() and track.is_located():
                    for shape_idx, shape in enumerate(track.shapes):
                        marker = Marker()
                        marker_name = track.id + ":" + str(shape_idx)
                        if marker_name not in self.marker_id_map:
                            self.marker_id_map[marker_name] = self.last_marker_id
                            self.last_marker_id += 1
                        marker_id = self.marker_id_map[marker_name]
                        marker.id = marker_id
                        marker.ns = track.id
                        marker.action = Marker.MODIFY
                        marker.header = header

                        node_pose = track.pose
                        shape_pose = shape.pose
                        final_pose = node_pose + shape_pose
                        position = final_pose.position()
                        orientation = final_pose.quaternion()

                        marker.pose.position.x = position.x
                        marker.pose.position.y = position.y
                        marker.pose.position.z = position.z

                        marker.pose.orientation.x = orientation[0]
                        marker.pose.orientation.y = orientation[1]
                        marker.pose.orientation.z = orientation[2]
                        marker.pose.orientation.w = orientation[3]

                        marker.color.r = shape.color[0]
                        marker.color.g = shape.color[1]
                        marker.color.b = shape.color[2]

                        if track.label in ["person", "face"]:
                            marker.color.a = 0.2
                        else:
                            marker.color.a = shape.color[3]

                        if track.is_static() is True:
                            marker.color.a = 1.0

                        if shape.is_cylinder():
                            marker.type = Marker.CYLINDER
                            marker.scale = Vector3(x=shape.width(),
                                                   y=shape.width(),
                                                   z=shape.height())
                        elif shape.is_sphere():
                            marker.type = Marker.SPHERE
                            marker.scale = Vector3(x=shape.width(),
                                                   y=shape.width(),
                                                   z=shape.width())
                        elif shape.is_box():
                            marker.type = Marker.CUBE
                            marker.scale = Vector3(x=shape.x,
                                                   y=shape.y,
                                                   z=shape.z)
                        elif shape.is_mesh():
                            if shape.mesh_resource != "":
                                marker.type = Marker.MESH_RESOURCE
                                marker.mesh_resource = shape.mesh_resource
                                marker.mesh_use_embedded_materials = True
                            else:
                                marker.type = Marker.TRIANGLE_LIST
                                marker.points = shape.vertices
                            marker.scale = Vector3(x=shape.scale.x, y=shape.scale.y, z=shape.scale.z)
                        else:
                            raise NotImplementedError("Shape not implemented")

                        marker.lifetime = rospy.Duration(20.0)
                        markers_msg.markers.append(marker)
        self.publisher.publish(markers_msg)
