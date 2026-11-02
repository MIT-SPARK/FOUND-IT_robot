#!/usr/bin/env python3
"""Publish a recorded session as a frozen wall-clock scene -- no bag playback.

`ros2 bag play --loop` was the obvious way to do this and it does not work.
Looping rewinds the clock every time the bag restarts: every tf2 buffer in the
system dumps itself with TF_OLD_DATA, MoveIt's CurrentStateMonitor resets, and
any plan in flight is lost. Nothing in a recording is actually dynamic -- the
object does not move and the arm cannot, since offline nothing subscribes to
control/joint_states -- so playback buys nothing at all. This reads the bag once
and publishes an unchanging snapshot on the wall clock: /tf_static latched as a
single accumulated message, the object cloud restamped and republished slowly,
and the first recorded joint state republished at a steady rate. No /clock and
no use_sim_time anywhere.

    ros2 run nero_bringup session_publisher.py --ros-args \
      -p bag_path:=$HOME/nero_recordings/apple
"""

import os
import re

import rclpy
import rosbag2_py
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, PointCloud2
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster


class SessionPublisher(Node):
    """Republishes one recorded session as a scene that never changes."""

    def __init__(self):
        super().__init__('session_publisher')

        self.declare_parameter('bag_path', '')
        self.declare_parameter('namespace', 'nero_right')
        self.declare_parameter('pcd_topic', 'object_pcd')
        self.declare_parameter('tf_static_topic', '/tf_static')
        self.declare_parameter('exclude_child_frames', r'^no\d+/')
        self.declare_parameter('joint_state_rate_hz', 50.0)
        self.declare_parameter('cloud_rate_hz', 1.0)

        bag_path = os.path.expanduser(
            os.path.expandvars(self.get_parameter('bag_path').value)
        )
        if not bag_path:
            self.get_logger().fatal('bag_path parameter is required')
            raise SystemExit(1)

        # The whole contract of this node is "everything is stamped now()". Under
        # sim time now() sits at zero until a /clock nobody publishes ticks, so
        # every timer below would simply never fire -- a silent hang rather than
        # an error, which is worth one explicit check.
        if self.get_parameter('use_sim_time').value:
            self.get_logger().fatal(
                'use_sim_time is true, but this node restamps to the wall clock '
                'and nothing publishes /clock offline -- every timer would stall'
            )
            raise SystemExit(1)

        namespace = str(self.get_parameter('namespace').value).strip('/')
        pcd_topic = str(self.get_parameter('pcd_topic').value).lstrip('/')
        tf_static_topic = str(self.get_parameter('tf_static_topic').value)
        joint_state_topic = f'/{namespace}/puppet/joint_states'

        pattern_text = str(self.get_parameter('exclude_child_frames').value)
        try:
            self.exclude = re.compile(pattern_text) if pattern_text else None
        except re.error as exc:
            self.get_logger().fatal(f"exclude_child_frames is not a regex: {exc}")
            raise SystemExit(1)

        joint_state_rate = float(self.get_parameter('joint_state_rate_hz').value)
        cloud_rate = float(self.get_parameter('cloud_rate_hz').value)
        if joint_state_rate <= 0.0 or cloud_rate <= 0.0:
            self.get_logger().fatal('publish rates must be positive')
            raise SystemExit(1)

        statics, joint_state, cloud = self._read_session(
            bag_path, tf_static_topic, joint_state_topic, f'/{pcd_topic}'
        )

        # A standstill snapshot: a session recorded mid-motion would otherwise
        # keep telling MoveIt the arm is moving at a pose it will never leave.
        joint_state.velocity = [0.0] * len(joint_state.velocity)
        self.joint_state = joint_state
        self.cloud = cloud

        # One sendTransform, not one per recorded message. Humble's Python
        # StaticTransformBroadcaster does not accumulate -- it publishes exactly
        # what it is handed, on a depth-1 transient-local publisher -- so
        # replaying the bag's messages one by one would leave a late-joining
        # subscriber holding only whichever happened to be last. Merging them
        # here is what lets a restarted node still see the whole tree.
        self.static_broadcaster = StaticTransformBroadcaster(self)
        stamp = self.get_clock().now().to_msg()
        for transform in statics.values():
            transform.header.stamp = stamp
        self.static_broadcaster.sendTransform(list(statics.values()))

        # Transient local to match grasp_selection's subscription, which is
        # transient local depth 1 and would not match a volatile publisher at
        # all. Republished slowly as well as latched because view_grasp2.rviz's
        # "Grasp Object (found-it)" display subscribes Volatile, and a volatile
        # subscriber gets nothing from the latch -- only whatever is published
        # after it comes up.
        self.cloud_pub = self.create_publisher(
            PointCloud2,
            f'/{pcd_topic}',
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        # Deliberately not transient local: a latched stale joint state is
        # exactly what CurrentStateMonitor's freshness check exists to reject.
        # This matches the profile agx_arm_ctrl offers on the real arm.
        self.joint_state_pub = self.create_publisher(JointState, joint_state_topic, 10)

        self.create_timer(1.0 / joint_state_rate, self._publish_joint_state)
        self.create_timer(1.0 / cloud_rate, self._publish_cloud)

        self._log_snapshot(
            bag_path, statics, joint_state, cloud, joint_state_topic,
            pcd_topic, joint_state_rate,
        )

    # -- inputs --------------------------------------------------------------

    def _read_session(self, bag_path, tf_static_topic, joint_state_topic, pcd_topic):
        """Read the bag once, returning the accumulated statics and first of each."""
        try:
            metadata = rosbag2_py.Info().read_metadata(bag_path, '')
        except Exception as exc:
            self.get_logger().fatal(f'cannot read a bag at {bag_path}: {exc}')
            raise SystemExit(1)

        available = [t.topic_metadata.name for t in metadata.topics_with_message_count]
        wanted = [tf_static_topic, joint_state_topic, pcd_topic]
        missing = [name for name in wanted if name not in available]
        if missing:
            self.get_logger().fatal(
                f"bag has no {', '.join(repr(m) for m in missing)} -- check the "
                f"namespace and pcd_topic parameters. this bag holds: "
                f"{', '.join(available)}"
            )
            raise SystemExit(1)

        reader = rosbag2_py.SequentialReader()
        # storage_id left empty: Humble reads it out of the bag's metadata.yaml.
        reader.open(
            rosbag2_py.StorageOptions(uri=bag_path),
            rosbag2_py.ConverterOptions('', ''),
        )
        # A storage-level filter rather than an `if` per message, so that once
        # camera topics get added to a recording their rows are never fetched.
        reader.set_filter(rosbag2_py.StorageFilter(topics=wanted))

        statics, dropped = {}, set()
        joint_state, cloud = None, None

        # Read all the way to EOF rather than stopping at the first of each.
        # calibration_publisher only starts emitting the camera extrinsics about
        # a second in (message 10 of 33 in the apple bag), so the static tree is
        # not complete until the end -- and that late chain is precisely what the
        # object cloud's frame hangs off. The storage filter keeps this cheap.
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic == tf_static_topic:
                for transform in deserialize_message(data, TFMessage).transforms:
                    child = transform.child_frame_id
                    if self.exclude is not None and self.exclude.search(child):
                        dropped.add(child)
                        continue
                    # Keyed by child, last write wins -- the same semantics tf2's
                    # own static buffer applies, so what we end up publishing is
                    # what a live subscriber would have converged to.
                    statics[child] = transform
            elif topic == joint_state_topic and joint_state is None:
                joint_state = deserialize_message(data, JointState)
            elif topic == pcd_topic and cloud is None:
                cloud = deserialize_message(data, PointCloud2)

        if not statics:
            self.get_logger().fatal(f'no usable transforms on {tf_static_topic}')
            raise SystemExit(1)
        if joint_state is None:
            self.get_logger().fatal(f'no joint state recorded on {joint_state_topic}')
            raise SystemExit(1)
        if cloud is None:
            self.get_logger().fatal(f'no point cloud recorded on {pcd_topic}')
            raise SystemExit(1)

        self._dropped = sorted(dropped)
        return statics, joint_state, cloud

    # -- outputs -------------------------------------------------------------

    def _publish_joint_state(self):
        # Restamped every tick rather than latched once: MoveIt's
        # CurrentStateMonitor discards any state stamped more than a second from
        # now(), which is the mismatch that made replaying bag-stamped states on
        # the wall clock abort every trajectory before it could execute.
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state_pub.publish(self.joint_state)

    def _publish_cloud(self):
        self.cloud.header.stamp = self.get_clock().now().to_msg()
        self.cloud_pub.publish(self.cloud)

    def _log_snapshot(self, bag_path, statics, joint_state, cloud,
                      joint_state_topic, pcd_topic, joint_state_rate):
        """One diagnosable block describing what was frozen."""
        self.get_logger().info(f'froze the scene in {bag_path}')
        self.get_logger().info(f'{len(statics)} static transforms latched on /tf_static')
        if self._dropped:
            self.get_logger().info(
                f"dropped {len(self._dropped)} frames matching "
                f"'{self.exclude.pattern}': {', '.join(self._dropped)}"
            )
        for child, transform in sorted(statics.items()):
            self.get_logger().debug(f'  {transform.header.frame_id} -> {child}')
        self.get_logger().info(
            f'{len(joint_state.name)} joints on {joint_state_topic} '
            f'at {joint_state_rate:g} hz'
        )
        self.get_logger().info(
            f'{cloud.width * cloud.height} points on /{pcd_topic} '
            f"in frame '{cloud.header.frame_id}'"
        )
        # Cheap check for an over-greedy exclude pattern, which would otherwise
        # show up much later as grasp_selection quietly never proposing anything.
        if cloud.header.frame_id.lstrip('/') not in statics:
            self.get_logger().warning(
                f"cloud frame '{cloud.header.frame_id}' is not a child of any "
                'published static transform -- grasp_selection will never resolve '
                'it into the planning frame'
            )


def main():
    rclpy.init()
    node = SessionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy installs its own signal handler, so on Ctrl-C or SIGTERM the
        # context is often already down by the time we get here and shutting it
        # down again raises -- which in a tmux pane means a traceback every time
        # you restart the node.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
