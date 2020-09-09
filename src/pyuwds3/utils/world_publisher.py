import rospy
from uwds3_msgs.msg import WorldStamped


class WorldPublisher(object):
    def __init__(self, world_name, queue_size=1, global_frame_id="odom"):
        self.global_frame_id = global_frame_id
        self.publisher = rospy.Publisher(world_name, WorldStamped, queue_size)

    def publish(self, tracks, events, header):
        """ """
        world_msg = WorldStamped()
        world_msg.header = header
        world_msg.header.frame_id = self.global_frame_id
        for track in tracks:
            if track.is_confirmed():
                world_msg.world.scene.append(track.to_msg(header))
        for event in events:
            world_msg.world.timeline.append(event.to_msg(header))
        self.publisher.publish(world_msg)
