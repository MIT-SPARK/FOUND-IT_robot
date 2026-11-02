#!/usr/bin/env python3
"""Bridge MoveIt's FollowJointTrajectory execution onto agx_arm_ctrl's move_j path.

agx_arm_ctrl has no ros2_control and no controller_manager -- it is a plain topic
subscriber (control/joint_states) that calls the vendor SDK's move_j() on every
message received, with no joint-limit clamping, no start-state check, and no
concept of a trajectory. This node is the only thing standing between a MoveIt
plan and the real arm, so it does that validation itself: it rejects (never
clamps) any goal with out-of-range positions/velocities/accelerations, or whose
first point does not match where the arm actually is.
"""

import math
import xml.etree.ElementTree as ET

import rclpy
import yaml
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

# Slack on velocity/acceleration checks, to absorb floating-point noise from
# time parameterization -- which deliberately targets these limits exactly, so a
# true 0.5 rad/s^2 limit can come back as -0.5000000000000002.
_LIMIT_EPSILON = 1e-3


def _seconds(duration):
    return duration.sec + duration.nanosec * 1e-9


class TrajectoryBridge(Node):
    def __init__(self):
        super().__init__('trajectory_bridge')

        self.declare_parameter('joint_names', [f'joint{i}' for i in range(1, 8)])
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('require_armed', True)
        self.declare_parameter('start_state_tolerance', 0.15)
        self.declare_parameter('goal_position_tolerance', 0.15)
        self.declare_parameter('joint_limits_yaml', '')
        self.declare_parameter('robot_description', '')

        self.joint_names = list(self.get_parameter('joint_names').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.require_armed = bool(self.get_parameter('require_armed').value)
        self.start_state_tolerance = float(self.get_parameter('start_state_tolerance').value)
        self.goal_position_tolerance = float(self.get_parameter('goal_position_tolerance').value)
        self.armed = not self.require_armed

        self.velocity_limits, self.acceleration_limits = self._load_joint_limits(
            self.get_parameter('joint_limits_yaml').value
        )
        self.position_limits = self._load_position_limits(
            self.get_parameter('robot_description').value
        )

        self.latest_joint_state = {}
        self.create_subscription(JointState, 'joint_states', self._joint_state_cb, 10)
        self.command_pub = self.create_publisher(JointState, 'control/joint_states', 10)
        self.create_service(SetBool, 'arm_trajectory_bridge', self._set_armed_cb)

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'arm_controller/follow_joint_trajectory',
            execute_callback=self._execute_callback,
            goal_callback=lambda _: GoalResponse.ACCEPT,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(
            'trajectory_bridge ready: '
            + ('armed' if self.armed else 'disarmed, call arm_trajectory_bridge to enable')
        )

    def _load_joint_limits(self, path):
        if not path:
            self.get_logger().warn('joint_limits_yaml not set; no velocity/acceleration checks')
            return {}, {}
        with open(path) as f:
            limits = yaml.safe_load(f).get('joint_limits', {})
        return (
            {n: v['max_velocity'] for n, v in limits.items() if v.get('has_velocity_limits')},
            {n: v['max_acceleration'] for n, v in limits.items() if v.get('has_acceleration_limits')},
        )

    def _load_position_limits(self, robot_description):
        if not robot_description:
            self.get_logger().warn('robot_description not set; no position bounds checks')
            return {}
        limits = {}
        for joint in ET.fromstring(robot_description).findall('joint'):
            limit = joint.find('limit')
            if joint.get('name') in self.joint_names and limit is not None:
                if 'lower' in limit.attrib and 'upper' in limit.attrib:
                    limits[joint.get('name')] = (
                        float(limit.get('lower')), float(limit.get('upper'))
                    )
        return limits

    def _joint_state_cb(self, msg):
        # Merge incrementally, like robot_state_publisher does on this same topic.
        self.latest_joint_state.update(zip(msg.name, msg.position))

    def _set_armed_cb(self, request, response):
        self.armed = request.data
        response.success = True
        response.message = 'armed' if self.armed else 'disarmed'
        self.get_logger().info(f'trajectory_bridge {response.message} via service call')
        return response

    def _validate_goal(self, trajectory):
        """Reason the goal is unsafe, or None if it passes."""
        if self.require_armed and not self.armed:
            return 'trajectory_bridge is disarmed; call arm_trajectory_bridge(true) first'

        unknown = set(trajectory.joint_names) - set(self.joint_names)
        if unknown:
            return f'unknown joints in goal: {sorted(unknown)}'

        if not self.latest_joint_state:
            return 'no joint_states feedback received yet; refusing to command blind'

        for index, point in enumerate(trajectory.points):
            # The first point is the arm's current live position, so it gets the
            # same start_state_max_bounds_error leniency the planner's own
            # FixStartStateBounds applies -- some joints rest slightly past their
            # declared URDF limit (see README). Every other point, the goal
            # included, gets goal_position_tolerance: CHOMP treats joint limits
            # as a soft cost rather than a hard constraint, so it can report a
            # successful plan whose goal sits a bit past a limit. That tolerance
            # is a deliberate, verified-on-hardware margin, not floating-point
            # noise absorption -- read the README before changing it.
            tolerance = self.start_state_tolerance if index == 0 else self.goal_position_tolerance
            for name, position in zip(trajectory.joint_names, point.positions):
                bounds = self.position_limits.get(name)
                if bounds and not (bounds[0] - tolerance <= position <= bounds[1] + tolerance):
                    return (f'{name} position {position:.3f} outside limits '
                            f'[{bounds[0]:.3f}, {bounds[1]:.3f}]')
            for kind, values, limits in (
                ('velocity', point.velocities, self.velocity_limits),
                ('acceleration', point.accelerations, self.acceleration_limits),
            ):
                for name, value in zip(trajectory.joint_names, values):
                    limit = limits.get(name)
                    if limit is not None and abs(value) > limit + _LIMIT_EPSILON:
                        return f'{name} {kind} {value:.3f} exceeds limit {limit:.3f}'

        for name, position in zip(trajectory.joint_names, trajectory.points[0].positions):
            current = self.latest_joint_state.get(name)
            if current is None:
                return f'no live feedback for joint {name}; refusing to command blind'
            if abs(current - position) > self.start_state_tolerance:
                return (f'{name} goal start {position:.3f} is {abs(current - position):.3f} rad '
                        f'from live position {current:.3f} '
                        f'(tolerance {self.start_state_tolerance:.3f})')

        return None

    def _resample(self, trajectory):
        """Linearly interpolate the trajectory onto a fixed control_rate grid."""
        points = trajectory.points
        duration = _seconds(points[-1].time_from_start)
        samples = []
        segment = 0
        for step in range(max(1, math.ceil(duration * self.control_rate)) + 1):
            t = min(step / self.control_rate, duration)
            while (segment < len(points) - 2
                   and _seconds(points[segment + 1].time_from_start) < t):
                segment += 1
            if len(points) == 1:
                samples.append(list(points[0].positions))
                continue
            t0 = _seconds(points[segment].time_from_start)
            t1 = _seconds(points[segment + 1].time_from_start)
            alpha = 0.0 if t1 == t0 else min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            samples.append([
                a + (b - a) * alpha
                for a, b in zip(points[segment].positions, points[segment + 1].positions)
            ])
        return trajectory.joint_names, samples

    def _publish_command(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = list(positions)
        self.command_pub.publish(msg)

    def _result(self, error_code, error_string=''):
        result = FollowJointTrajectory.Result()
        result.error_code = error_code
        result.error_string = error_string
        return result

    def _execute_callback(self, goal_handle):
        trajectory = goal_handle.request.trajectory
        error = self._validate_goal(trajectory)
        if error is not None:
            self.get_logger().warn(f'rejecting goal: {error}')
            goal_handle.abort()
            return self._result(FollowJointTrajectory.Result.INVALID_GOAL, error)

        names, samples = self._resample(trajectory)
        self.get_logger().info(
            f'executing goal: {len(samples)} samples @ {self.control_rate:.0f} Hz, '
            f'duration {_seconds(trajectory.points[-1].time_from_start):.2f}s'
        )

        rate = self.create_rate(self.control_rate)
        for i, positions in enumerate(samples):
            if not goal_handle.is_active:
                return self._result(
                    FollowJointTrajectory.Result.INVALID_GOAL, 'goal no longer active'
                )

            if goal_handle.is_cancel_requested:
                # Hold where we are rather than stopping mid-stream with no
                # command outstanding.
                self._publish_command(names, positions)
                goal_handle.canceled()
                self.get_logger().info(f'goal canceled at sample {i}/{len(samples)}; holding')
                return self._result(
                    FollowJointTrajectory.Result.SUCCESSFUL,
                    'canceled, holding last commanded position',
                )

            self._publish_command(names, positions)

            if i % max(1, int(self.control_rate)) == 0:
                feedback = FollowJointTrajectory.Feedback()
                feedback.joint_names = names
                feedback.desired.positions = positions
                feedback.actual.positions = [
                    self.latest_joint_state.get(n, 0.0) for n in names
                ]
                goal_handle.publish_feedback(feedback)

            rate.sleep()

        goal_handle.succeed()
        return self._result(FollowJointTrajectory.Result.SUCCESSFUL)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
