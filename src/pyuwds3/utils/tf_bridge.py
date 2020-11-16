import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from pyuwds3.types.vector.vector6d import Vector6D


class TfBridge(object):
    """ Utility class to interface with tf2 """
    def __init__(self, prefix=""):
        """ """
        self.tf_buffer = Buffer()
        self.prefix = prefix
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

    def publish_pose_to_tf(self, pose, source_frame, target_frame, header=None):
        """ Publish the given pose to /tf2 """
        msg = pose.to_msg()
        transform = TransformStamped()
        transform.child_frame_id = target_frame
        transform.header.frame_id = source_frame
        if header is not None:
            transform.header.stamp = header.stamp
        else:
            transform.header.stamp = rospy.Time().now()
        transform.transform.translation = msg.position
        transform.transform.rotation = msg.orientation
        self.tf_broadcaster.sendTransform(transform)

    def get_pose_from_tf(self, source_frame, target_frame, time=None):
        """ Get the pose from /tf2 """
        try:
            if time is not None:
                # self.tf_listener.wait_for_transform(source_frame, target_frame, time, rospy.Duration(4.0))
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, time)
            else:
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w
            pose = Vector6D(x=x, y=y, z=z).from_quaternion(rx, ry, rz, rw)
            return True, pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[tf_bridge] Exception occureddd: {}".format(e))
            # return False, None
            pose = Vector6D(x=0, y=0, z=0).from_quaternion(0, 0, 0, 0)
            return False,pose
    def publish_tf_frames(self, tracks, events, header):
        for track in tracks:
            if track.is_located() is True and track.is_confirmed() is True:
                self.publish_pose_to_tf(track.pose, header.frame_id, self.prefix + track.id, header=header)
        for event in events:
            if event.is_located() is True:
                frame = event.subject+event.description+event.object
                pose = Vector6D(x=event.point.x, y=event.point.y, z=event.point.z)
                self.publish_pose_to_tf(pose, header.frame_id, self.prefix + frame, header=header)
