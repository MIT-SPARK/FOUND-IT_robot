#!/usr/bin/env python3
"""Watch for a grasp or a place target, plan the sequence, preview it, run it.

Everything about *what* the arm should do lives in nero_motion_planning's
TaskPathGenerator; everything about *how* it gets there lives in a MotionBackend.
This node is only the ROS glue between them: TF in, gripper and preview out, and
the execute_grasp gate in the middle.

A pick target is a ranked grasp frame published on TF by analytic_grasp_ros. A
place target is either a bare point (place_target.topic, orientation synthesized
here as straight down) or a fully specified pose on `place_target`.

    ros2 launch nero_bringup pick_place.launch.py
"""

import math
from typing import List, Optional, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_path
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped
from moveit_msgs.msg import DisplayTrajectory, RobotState, RobotTrajectory
from pymoveit2 import MoveIt2
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, UInt32
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)

from nero_motion_planning.backend import MoveItBackend
from nero_motion_planning.geometry import (
    flip_180_about_approach_axis,
    pose_from_transform,
    poses_close,
    position_xyz,
    quat_xyzw,
    top_down_pose,
    transform_point,
)
from nero_motion_planning.path_generator import TaskPathGenerator
from nero_motion_planning.segment import Grip


class PoseCommander(Node):
    def __init__(self):
        super().__init__(
            'pose_commander',
            namespace='nero_right',
            cli_args=['--ros-args', '-r', 'joint_states:=puppet/joint_states'],
        )
        self.declare_parameter('config_file', '')
        self.config = self._load_config()

        self.generator = TaskPathGenerator(self.config)
        self.joint_names = list(self.config['joints'])
        self.frames = self.config['frames']
        self.gripper_cfg = self.config['gripper']
        self.execution_cfg = self.config['execution']
        self.place_cfg = self.config['place_target']
        self.place_yaws = [math.radians(d) for d in self.place_cfg['yaws_deg']]
        candidates = self.config['candidates']
        self.candidate_frames = [
            candidates['frame_template'].format(i=i) for i in range(candidates['count'])
        ]
        self.retry_180_flip = candidates['retry_180_flip']
        self.position_tolerance_m = candidates['position_tolerance_m']
        self.orientation_tolerance_rad = candidates['orientation_tolerance_rad']
        self.poll_rate_hz = candidates['poll_rate_hz']

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.display_pub = self.create_publisher(DisplayTrajectory, 'display_planned_path', 1)
        self.waypoint_pub = self.create_publisher(PoseArray, 'manipulation_waypoints', 1)
        self.gripper_pub = self.create_publisher(JointState, 'control/joint_states', 1)
        self.picked_pub = self.create_publisher(
            Bool, '/object_picked_up',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )
        #self.placed_pub = self.create_publisher(
        #    Bool, '/object_dropped_off',
        #    QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
        #               history=HistoryPolicy.KEEP_LAST),
        #)


        self.create_service(Trigger, 'execute_grasp', self._on_execute_requested)
        self.create_subscription(PoseStamped, 'place_target', self._on_place_target, 1)
        # TRANSIENT_LOCAL to match the publisher (vggt_slam_place_obj latches its
        # point). With the default volatile QoS the two are still compatible, but
        # a point published before this node came up would never be delivered --
        # the same reason grasp_selection subscribes to object_pcd this way.
        self.create_subscription(
            PointStamped, self.place_cfg['topic'], self._on_place_point,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )
        self.clear_grasps_client = self.create_client(Trigger, '/clear_grasps')
        # Which sampling run the candidate frames currently belong to. TF carries
        # no batch identity -- grasp_selection rebroadcasts the same frames at
        # 4 Hz and lookup_target reads them at time-zero -- so this is the only
        # way to tell a fresh candidate set from one already picked. Latched, to
        # match the publisher and so a batch predating this node still arrives.
        self.create_subscription(
            UInt32, '/grasp_batch', self._on_grasp_batch,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST),
        )

        self.execute_requested = False
        self.pending_place: Optional[PoseStamped] = None
        self.pending_place_point: Optional[PointStamped] = None
        self.candidate_seq: Optional[int] = None

    def _load_config(self):
        path = self.get_parameter('config_file').value
        if not path:
            self.get_logger().fatal('config_file parameter is required')
            raise SystemExit(1)
        with open(path) as f:
            config = yaml.safe_load(f)
        # Default to the same joint_limits.yaml move_group and trajectory_bridge
        # use, rather than baking a machine-specific install path into config.
        if not config['moveit'].get('joint_limits_yaml'):
            config['moveit']['joint_limits_yaml'] = str(
                get_package_share_path('nero_moveit_config') / 'config' / 'joint_limits.yaml'
            )
        return config

    # -- inputs ------------------------------------------------------------

    def _on_execute_requested(self, request, response):
        # Only flips a flag: execute() and wait_until_executed() spin this node
        # internally, so they cannot run inside a callback that spin_once() is
        # already processing.
        self.execute_requested = True
        response.success = True
        response.message = 'execute requested'
        return response

    def _on_place_target(self, msg: PoseStamped):
        self.pending_place = msg

    def _on_place_point(self, msg: PointStamped):
        self.pending_place_point = msg

    def _on_grasp_batch(self, msg: UInt32):
        self.candidate_seq = msg.data

    def place_attempts(self) -> List[Tuple[str, PoseStamped]]:
        """Release poses for the pending place point, best yaw first; [] if not resolvable yet."""
        msg = self.pending_place_point
        if not msg.header.frame_id:
            self.get_logger().error(
                f"place point on {self.place_cfg['topic']} has an empty frame_id; dropping it"
            )
            self.pending_place_point = None
            return []
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frames['planning'], msg.header.frame_id, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException) as exc:
            # Kept pending rather than dropped: a point can easily arrive before
            # the transform that carries its frame does, and the caller retries.
            self.get_logger().warning(
                f"waiting for {self.frames['planning']} <- {msg.header.frame_id} "
                f'to place at that point: {exc}',
                throttle_duration_sec=5.0,
            )
            return []

        x, y, z = transform_point(transform, (msg.point.x, msg.point.y, msg.point.z))
        # The flange stands off above the drop point; the object hangs below it.
        position = (x, y, z + self.place_cfg['release_standoff_m'])
        # A point says where the object goes but nothing about the hand, so the
        # orientation is synthesized: straight down, once per configured yaw. See
        # geometry.top_down_pose for why yaw 0 leads and why this is a fan.
        attempts = []
        for yaw in self.place_yaws:
            stamped = PoseStamped()
            stamped.header.frame_id = self.frames['planning']
            stamped.pose = top_down_pose(position, yaw)
            attempts.append((f'place yaw {math.degrees(yaw):+.0f}', stamped))
        return attempts

    def lookup_target(self, tf_frame: str) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.frames['planning'], tf_frame, rclpy.time.Time()
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        stamped = PoseStamped()
        stamped.header.frame_id = self.frames['planning']
        stamped.pose = pose_from_transform(transform)
        return stamped

    # -- outputs -----------------------------------------------------------

    def send_gripper_command(self, grip: Grip):
        cfg = self.gripper_cfg
        width, force = (
            (cfg['open_width_m'], cfg['open_force_n']) if grip is Grip.OPEN
            else (cfg['close_width_m'], cfg['close_force_n'])
        )
        msg = JointState()
        msg.name = ['gripper']
        msg.position = [width]
        msg.effort = [force]
        self.gripper_pub.publish(msg)

    def publish_waypoints(self, segments):
        """The generator's own output, independent of any planner.

        Published for every *attempted* sequence, not just solved ones: when IK
        fails there is no trajectory to preview, and seeing where the waypoints
        actually landed is the whole diagnosis. Note that RViz shares
        display_planned_path with move_group, whose own DisplayTrajectory always
        holds exactly one trajectory -- so on a failed plan the animation you see
        there is move_group's last solved segment, not this sequence.
        """
        poses = PoseArray()
        poses.header.frame_id = self.frames['planning']
        poses.header.stamp = self.get_clock().now().to_msg()
        poses.poses = [s.command.pose_goal for s in segments if s.command]
        self.waypoint_pub.publish(poses)

    def publish_preview(self, planned, start_joint_state):
        # DisplayTrajectory.trajectory is an array, so each segment goes in as
        # its own entry and RViz animates them back to back -- no concatenation.
        display = DisplayTrajectory()
        display.trajectory_start = RobotState(joint_state=start_joint_state)
        for trajectory in planned.trajectories or []:
            display.trajectory.append(RobotTrajectory(joint_trajectory=trajectory))
        self.display_pub.publish(display)
        self.publish_waypoints(planned.segments)

    def request_clear_grasps(self):
        """Stop grasp_selection republishing, so its TF frames age out."""
        if not self.clear_grasps_client.service_is_ready():
            self.get_logger().warning('/clear_grasps unavailable -- grasp TFs will keep republishing')
            return
        self.clear_grasps_client.call_async(Trigger.Request())


def spin_for(node, duration_sec, period_sec=0.1):
    """Wait while keeping callbacks alive, unlike a blocking time.sleep()."""
    for _ in range(max(1, int(duration_sec / period_sec))):
        rclpy.spin_once(node, timeout_sec=period_sec)


def current_joints(moveit2, joint_names) -> List[float]:
    state = moveit2.joint_state
    return [state.position[state.name.index(j)] for j in joint_names]


def resolve_named_configurations(node, moveit2):
    """Give every named configuration both goal forms, and cross-check the ones
    that came with a pose.

    Where config records a pose, FK of its joints must agree -- that catches a
    pose captured for the wrong link (link7 rather than gripper_flange, say,
    which differ by a fixed 0.0397 m and 120 degrees). Where config records none,
    FK supplies it, so a backend with no joint-space interface can still reach
    that waypoint without anyone hand-capturing it.
    """
    # Only a backend with no joint-space interface actually uses these poses;
    # MoveItBackend plans named waypoints straight from their joints. So a
    # mismatch is fatal only when something will act on it.
    pose_is_used = node.config['backend'] != 'moveit'
    for name, entry in node.config['named_configurations'].items():
        expected = node.generator.named_pose(name)
        result = moveit2.compute_fk(joint_state=list(entry['joints']))
        if result is None:
            node.get_logger().warning(f"could not FK named configuration '{name}'")
            continue
        if expected is None:
            node.generator.set_named_pose(name, result.pose)
            node.get_logger().info(f"named configuration '{name}': pose derived by FK")
            continue
        if poses_close(result.pose, expected, 0.01, 0.05):
            node.get_logger().info(f"named configuration '{name}' verified against FK")
            continue

        position = result.pose.position
        rpy = Rot.from_quat(list(quat_xyzw(result.pose))).as_euler('xyz')
        offset = math.dist(position_xyz(result.pose), position_xyz(expected))
        message = (
            f"named configuration '{name}': recorded pose is {offset:.4f} m from FK of "
            f"its joints, so it was captured for a different link than "
            f"{node.frames['end_effector']}. Correct values:\n"
            f'    pose:\n'
            f'      position: [{position.x:.6f}, {position.y:.6f}, {position.z:.6f}]\n'
            f'      rpy: [{rpy[0]!r}, {rpy[1]!r}, {rpy[2]!r}]'
        )
        if pose_is_used:
            node.get_logger().fatal(message)
            raise SystemExit(1)
        node.get_logger().error(
            message + f"\n  Continuing: backend '{node.config['backend']}' plans this "
            'waypoint from its joints and never reads the pose.'
        )


def home_arm(node, backend):
    """Move to home before doing anything else, through the configured backend.

    Fatal on failure: an arm that could not reach a known state should not go on
    to grasp. Goes through backend.plan()/execute_segment() like any other
    sequence, so whichever controller is running is the one that homes.
    """
    node.get_logger().info('homing...')
    planned = backend.plan(
        node.generator.move_to_named('home'),
        current_joints(node.moveit2, node.joint_names),
    )
    if planned is None:
        node.get_logger().fatal('could not plan a path home; refusing to start')
        raise SystemExit(1)
    if not execute_task(node, backend, planned):
        node.get_logger().fatal('home move failed to execute; refusing to start')
        raise SystemExit(1)
    node.get_logger().info('home reached')


def plan_task(node, backend, attempts: List[Tuple[str, Optional[PoseStamped]]], task: str):
    """Plan `task`, walking `attempts` best-first, and return the first that solves."""
    # Neither task is special-cased here because both arrive ranked:
    # grasp_selection orders its candidates no0, no1, ..., and a place point
    # expands to one release pose per configured yaw.
    start = current_joints(node.moveit2, node.joint_names)
    # Every attempt below shares this `start` and the same named waypoints, so the
    # backend may reuse a leg it already solved -- but only within this round, and
    # only against the live pose read on the line above.
    backend.clear_plan_cache()

    for index, (label, candidate) in enumerate(attempts):
        if candidate is None:
            continue
        segments = node.generator.generate(candidate, task)
        # Before planning, so the waypoints are visible even for the attempt
        # that fails -- the last attempt's waypoints are what stay on screen.
        node.publish_waypoints(segments)
        planned = backend.plan(segments, start)
        if planned is None and node.retry_180_flip:
            flipped = PoseStamped(header=candidate.header,
                                  pose=flip_180_about_approach_axis(candidate.pose))
            segments = node.generator.generate(flipped, task)
            node.publish_waypoints(segments)
            planned = backend.plan(segments, start)
        if planned is not None:
            if index:
                node.get_logger().info(f'planned using fallback candidate {label}')
            return planned
    return None


def execute_task(node, backend, planned):
    for index, segment in enumerate(planned.segments):
        node.get_logger().info(f"executing '{segment.name}' ({segment.motion.value})")
        if not backend.execute_segment(planned, index):
            node.get_logger().error(f"stopped: segment '{segment.name}' failed")
            return False
        if segment.grip is not None:
            node.get_logger().info(f'{segment.grip.value} gripper')
            node.send_gripper_command(segment.grip)
            spin_for(node, node.gripper_cfg['settle_sec'])
    return True


def main():
    rclpy.init()
    node = PoseCommander()

    moveit2 = MoveIt2(
        node=node,
        joint_names=node.joint_names,
        base_link_name=node.frames['planning'],
        end_effector_name=node.frames['end_effector'],
        group_name='arm',
    )
    # Explicit rather than relying on move_group's default_planning_pipeline, so
    # this node's behaviour cannot silently change if that default is edited.
    moveit2.pipeline_id = node.config['moveit']['free_pipeline']
    node.moveit2 = moveit2

    backend = MoveItBackend(node, moveit2, node.config, node.joint_names)

    # Order matters: a real start state is needed before anything can be planned
    # from it, and home's pose is derived by the resolve step below.
    while moveit2.joint_state is None:
        rclpy.spin_once(node, timeout_sec=1.0)
    resolve_named_configurations(node, moveit2)
    home_arm(node, backend)

    require_confirmation = node.execution_cfg['require_confirmation']
    min_delta = node.execution_cfg['auto_execute_min_target_delta_m']
    node.get_logger().info(
        'watching for grasp candidates; '
        + ('press Enter in execute_on_enter.py to run a plan'
           if require_confirmation else 'executing automatically once planned')
    )

    last_target: Optional[PoseStamped] = None
    planned, planned_task = None, None
    planned_seq: Optional[int] = None
    # The target of the armed plan, and the last target executed per task. Only
    # the auto-execute gate below uses these; last_target gates planning instead,
    # and the two thresholds are deliberately different. Keyed by task because a
    # place target is not a near-miss of the pick before it -- comparing the two
    # would gate a legitimate place that happened to land near the grasp.
    planned_target: Optional[PoseStamped] = None
    last_executed: dict = {}

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=1.0 / node.poll_rate_hz)

        # A place takes priority over a pick, and while one is pending no pick is
        # considered at all: the arm is holding something, so a place whose frame
        # TF cannot resolve yet has to wait rather than turn into a grasp. Both
        # place inputs are consumed as they are taken, so a stale target cannot
        # re-fire; a pick, by contrast, is re-read from TF every poll and needs
        # the de-dup below.
        attempts, task = [], 'place'
        if node.pending_place is not None:
            attempts = [('place_target', node.pending_place)]
            node.pending_place = None
        elif node.pending_place_point is not None:
            # [] while TF cannot reach the point's frame; retried next poll.
            attempts = node.place_attempts()
            if attempts:
                node.pending_place_point = None
        else:
            task = 'pick'
            attempts = [(frame, node.lookup_target(frame)) for frame in node.candidate_frames]
            best = attempts[0][1]
            # Two guards, and both are load-bearing.
            #
            # The batch sequence retires a candidate set once a plan has been made
            # from it. TF cannot express that: grasp_selection rebroadcasts the
            # same frames at 4 Hz and lookup_target reads them at time-zero, so a
            # grasp stays resolvable long after it has been picked -- which is
            # what used to re-fire a pick 2 ms after execution succeeded.
            #
            # The pose check covers the two cases a sequence cannot. A new batch
            # whose transforms have not landed yet still reads the *previous*
            # batch's frames, so wait for the next poll rather than plan the old
            # scene under the new number. And an unchanged cloud reprocessed under
            # seed 0 is a new batch of bit-identical grasps, which offline
            # pcd_publisher produces every second; replanning those every poll
            # would just retry the same unreachable pose ~5x/sec.
            if (best is None
                    or node.candidate_seq is None
                    or node.candidate_seq == planned_seq
                    or (last_target is not None and poses_close(
                        best.pose, last_target.pose,
                        node.position_tolerance_m, node.orientation_tolerance_rad,
                    ))):
                attempts = []

        if attempts:
            if planned is not None:
                node.get_logger().warning(
                    'replacing a previewed plan that was never executed'
                )
            node.get_logger().info(f'new {task} target -- planning...')
            planned, planned_task = plan_task(node, backend, attempts, task), task
            planned_target = attempts[0][1]
            if task == 'pick':
                last_target, planned_seq = attempts[0][1], node.candidate_seq
            if planned is None:
                node.get_logger().warning(f'planning failed for all {task} candidates')
            else:
                node.publish_preview(planned, moveit2.joint_state)
                node.get_logger().info(f'plan ready: {len(planned.segments)} segments')
                if not require_confirmation:
                    # With no human in the loop this is the last gate: refuse to
                    # move for a target that has barely shifted since the one just
                    # run, so a jittering estimate cannot walk the arm through the
                    # same grasp over and over. Deliberately coarser than the
                    # candidates tolerance that gates *planning* (0.002 m) --
                    # planning again is free, moving is not.
                    previous = last_executed.get(task)
                    moved_m = (
                        math.inf if previous is None
                        else math.dist(position_xyz(planned_target.pose),
                                       position_xyz(previous.pose))
                    )
                    if moved_m < min_delta:
                        node.get_logger().warning(
                            f'not auto-executing: target moved {moved_m:.4f} m '
                            f'since the last execution, under the '
                            f'{min_delta} m auto_execute_min_target_delta_m gate'
                        )
                    else:
                        node.execute_requested = True

        if node.execute_requested:
            node.execute_requested = False
            if planned is None:
                node.get_logger().warning('execute requested but no plan is available')
            elif execute_task(node, backend, planned):
                # /object_picked_up tracks whether the arm is holding something,
                # so a place reports the release. Only a pick consumed the grasp
                # candidates, so only a pick retires them.
                if planned_task == 'pick':
                    node.request_clear_grasps()
                node.picked_pub.publish(Bool(data=planned_task == 'pick'))
                #node.placed_pub.publish(Bool(data=planned_task == 'place'))
                # last_target and planned_seq deliberately survive: together they
                # are what retires the candidate set just executed. Clearing them
                # re-arms the same grasp on the next poll, 0.2 s later.
                last_executed[planned_task] = planned_target
                planned, planned_task = None, None
            else:
                # The arm stopped somewhere between segments, so the plan's start
                # state no longer describes it. Leaving it armed would dispatch it
                # verbatim on the next Enter; trajectory_bridge would reject it on
                # start_state_tolerance, but only after it had been sent.
                node.get_logger().warning(
                    f'{planned_task} execution failed -- dropping the plan. '
                    + ('Another pick arms on the next candidate batch.'
                       if planned_task == 'pick'
                       else 'Republish a place target to retry.')
                )
                planned, planned_task = None, None

    rclpy.shutdown()


if __name__ == '__main__':
    main()
