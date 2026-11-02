"""Motion backends: a Segment sequence in, arm motion out.

The seam that lets the controller change without the task logic changing.
MoveItBackend is the only implementation today; an ImpedanceBackend publishing
PoseCommand to impedance_control is the intended second one.
"""

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot, Slerp
from trajectory_msgs.msg import JointTrajectory

from .geometry import make_pose, position_xyz, quat_xyzw
from .segment import Motion, Segment

# Fraction of joint_limits.yaml a plan is timed to when config sets no explicit
# scaling. The gap only has to swallow TOTG's resampling error (~0.4% of the
# limit, measured on joint4); 5% leaves room without visibly slowing the arm.
_DEFAULT_SCALING = 0.95

# Aim a hair inside the limit rather than exactly at it. Scaling to land on the
# limit leaves the result at the mercy of floating point, and the overshoot that
# started all this was 0.4% -- trajectory_bridge's own epsilon is 1e-3, so a
# 0.1% margin lands clear of it while costing no visible speed.
_LIMIT_MARGIN = 1.001

# Rung spacing for the IK ladder in _ik_ladder. Small enough that consecutive
# solutions stay in the same branch, coarse enough not to spend a service
# round-trip per centimetre: 0.05 m is ~8 rungs over the ~0.4 m staging ->
# pregrasp move, and 0.35 rad (~20 deg) is the binding one in practice, since
# the wrist runs out of room long before reach does.
_LADDER_STEP_M = 0.05
_LADDER_STEP_RAD = 0.1
# Ceiling on the IK calls one failed pregrasp can cost, whatever the distance.
_LADDER_MAX_STEPS = 15


@dataclass
class PlannedSequence:
    """A solved sequence, ready to execute one segment at a time."""

    segments: List[Segment]
    #: One per segment. None for a backend that has no joint-space plan to show
    #: (an impedance controller servos to setpoints), in which case there is no
    #: RViz preview either.
    trajectories: Optional[List[JointTrajectory]] = None
    #: Joint positions the arm ends at, for chaining onto a following sequence.
    final_joints: Optional[List[float]] = None


class MotionBackend(ABC):
    """Turns Segments into motion. Knows nothing about grippers or grasps."""

    @abstractmethod
    def plan(self, segments: List[Segment], start_joints: List[float]) -> Optional[PlannedSequence]:
        """Solve the whole sequence up front; None if any segment is unreachable.

        Planning everything before moving is what lets a bad grasp candidate be
        rejected while the arm is still stationary.
        """

    @abstractmethod
    def execute_segment(self, planned: PlannedSequence, index: int) -> bool:
        """Run one segment. The caller applies that segment's gripper action."""

    def clear_plan_cache(self) -> None:
        """Drop anything memoized across plan() calls. Call once per round.

        Concrete rather than abstract: a backend that memoizes nothing is a
        correct backend, and the caller should not have to know which is which.
        """


class MoveItBackend(MotionBackend):
    """Plans through move_group: CHOMP for FREE legs, Cartesian for LINEAR ones."""

    def __init__(self, node, moveit2, config: Dict, joint_names: List[str]):
        self._node = node
        self._moveit2 = moveit2
        self._joint_names = joint_names
        cfg = config['moveit']
        self._pipeline = cfg['free_pipeline']
        self._linear = cfg['linear']
        self._enforce_limits = cfg['enforce_joint_limits']
        self._velocity_limits, self._acceleration_limits = _load_joint_limits(
            cfg.get('joint_limits_yaml'), node
        )
        self._moveit2.cartesian_avoid_collisions = self._linear['avoid_collisions']
        # pymoveit2 leaves the request's scaling factors at 0.0, which MoveIt
        # reads as "full speed", so its Add Time Optimal Parameterization
        # adapter times every plan to ride joint_limits.yaml exactly. TOTG then
        # resamples onto its resample_dt grid, and the finite differences of
        # that resampling come back a hair past the limit it just parameterized
        # to -- joint4 acceleration 0.502 against a 0.500 limit on a plain home
        # move. trajectory_bridge rejects rather than clamps, by design, so that
        # hair is a dead home sequence. Asking for a few percent of headroom
        # fixes it at the source instead of widening the bridge's safety margin.
        self._moveit2.max_velocity = cfg.get('velocity_scaling', _DEFAULT_SCALING)
        self._moveit2.max_acceleration = cfg.get('acceleration_scaling', _DEFAULT_SCALING)
        # Extra IK seeds, tried in file order (staging, then home) after the
        # chained state. TRAC-IK searches outward from its seed, so a pose that
        # is unreachable from one arm configuration is often solvable from
        # another -- see _plan_free_pose.
        self._fallback_seeds = [
            (name, list(entry['joints']))
            for name, entry in config['named_configurations'].items()
        ]
        # Joint-space plans solved so far this round; see _plan_joint.
        self._joint_plan_cache: Dict[Tuple, JointTrajectory] = {}

    def clear_plan_cache(self):
        self._joint_plan_cache.clear()

    def plan(self, segments, start_joints):
        trajectories = []
        joints = list(start_joints)
        for segment in segments:
            trajectory = self._plan_segment(segment, joints)
            if trajectory is None or not trajectory.points:
                self._node.get_logger().warning(
                    f"planning failed at segment '{segment.name}' "
                    f'({len(trajectories)}/{len(segments)} solved)'
                )
                return None
            trajectories.append(trajectory)
            # Chain: the next segment starts where this one ended, not where the
            # arm is now. Without this every segment would be planned from the
            # live pose and the sequence would not compose.
            joints = list(trajectory.points[-1].positions)
        return PlannedSequence(segments=segments, trajectories=trajectories, final_joints=joints)

    def execute_segment(self, planned, index):
        segment = planned.segments[index]
        self._moveit2.execute(planned.trajectories[index])
        succeeded = self._moveit2.wait_until_executed()
        if not succeeded:
            self._node.get_logger().warning(
                f"segment '{segment.name}' failed: "
                f'{self._moveit2.get_last_execution_error_code()}'
            )
        return succeeded

    def _plan_segment(self, segment, start_joints):
        if segment.joints is not None:
            trajectory = self._plan_joint(segment.joints, start_joints)
        elif segment.motion is Motion.LINEAR:
            trajectory = self._plan_linear(segment, start_joints)
        else:
            trajectory = self._plan_free_pose(segment, start_joints)
        if trajectory is None or not self._enforce_limits:
            return trajectory
        # Every segment, not just the Cartesian ones. velocity_scaling only asks
        # MoveIt for headroom on the legs it times itself; it is not a guarantee,
        # and trajectory_bridge rejects the whole goal over a 0.4% overshoot. The
        # fitter is a no-op on a trajectory already inside the limits, so this
        # costs nothing when the scaling did its job.
        #
        # Safe against _plan_joint's cache despite editing points in place: on a
        # miss this fits the just-cached object, so the entry is stored fitted;
        # on a hit the deepcopy is already fitted and this is a no-op.
        return self._fit_to_joint_limits(trajectory)

    def _plan_joint(self, goal_joints, start_joints):
        """Solve a joint-to-joint leg, reusing an identical one from this round.

        pose_commander walks up to candidates.count grasps (twice over, with
        retry_180_flip) and every one of those sequences opens with the same
        'staging' segment: same start joints, since plan_task reads the live pose
        once per round, and same goal joints, since they come from
        named_configurations. That is one CHOMP problem re-solved dozens of times
        per perception event. The cache is cleared at the top of each round, so a
        plan is only ever reused against the state it was actually solved from.
        """
        key = (_quantize(start_joints), _quantize(goal_joints))
        cached = self._joint_plan_cache.get(key)
        if cached is not None:
            # Copied out: each attempt's PlannedSequence would otherwise alias one
            # trajectory, and _fit_to_joint_limits edits points in place.
            return deepcopy(cached)

        self._moveit2.pipeline_id = self._pipeline
        trajectory = self._moveit2.plan(
            joint_positions=list(goal_joints),
            joint_names=self._joint_names,
            start_joint_state=start_joints,
        )
        # A failure is not cached: CHOMP is seeded from a straight line and can
        # fail on a problem it would solve on a retry, and plan() treats an empty
        # trajectory as a failure too.
        if trajectory is not None and trajectory.points:
            self._joint_plan_cache[key] = trajectory
        return trajectory

    def _plan_free_pose(self, segment, start_joints):
        # CHOMP optimizes between two joint configurations and has no goal-region
        # IK sampling of its own, so the pose goal has to be solved to joints
        # first. Seeding from the chained state keeps consecutive solutions in
        # the same IK branch, which is what stops the elbow flipping mid-sequence
        # -- so it is always tried first, and the two fallbacks below only come
        # into play when it finds nothing at all.
        pose = segment.command.pose_goal
        goal = self._solve_ik(pose, start_joints, segment.name)

        # A one-shot solve from a distant seed is all-or-nothing, and this pose
        # is the one the arm most often cannot get from where it stands. Walking
        # there in small steps turns it into a series of solves that each start
        # a few centimetres from their answer, which is the regime TRAC-IK is
        # reliable in. Only the last rung's joints survive -- see _ik_ladder.
        if goal is None:
            goal = self._ik_ladder(segment.name, pose, start_joints)

        # Last resort: a fixed configuration elsewhere in the workspace. Unlike
        # the ladder this abandons branch continuity with the chained state, so
        # it comes after.
        if goal is None:
            for seed_name, seed in self._fallback_seeds:
                goal = self._solve_ik(pose, seed, segment.name)
                if goal is not None:
                    self._node.get_logger().info(
                        f"IK for '{segment.name}' needed the '{seed_name}' seed"
                    )
                    break

        if goal is None:
            self._node.get_logger().warning(
                f"no IK for '{segment.name}' from the chained state, a ladder, or "
                f"{[n for n, _ in self._fallback_seeds]}: {_describe(pose)}"
            )
            return None
        return self._plan_joint(goal, start_joints)

    def _solve_ik(self, pose, seed, name) -> Optional[List[float]]:
        """One compute_ik call, as joints in self._joint_names order; None if unsolved."""
        solution = self._moveit2.compute_ik(
            position=list(position_xyz(pose)),
            quat_xyzw=list(quat_xyzw(pose)),
            start_joint_state=list(seed),
        )
        if solution is None:
            return None
        try:
            # Rounded: TRAC-IK clamps a saturated joint against a bound that has
            # been through a 32-bit float, so pinning joint6 at its -0.73 stop
            # returns -0.7300000190734863 -- 1.9e-8 rad outside the limit CHOMP
            # then checks with satisfiesBounds(), whose tolerance is zero. That
            # killed every pick at 'pregrasp' with INVALID_ROBOT_STATE ("Goal
            # state violates joint limits"). 1e-6 rad is far below anything the
            # arm can resolve and lands such a joint back on its stop exactly.
            return [round(solution.position[solution.name.index(j)], 6) for j in self._joint_names]
        except ValueError:
            self._node.get_logger().warning(
                f"IK solution for '{name}' is missing one of {self._joint_names}"
            )
            return None

    def _ik_ladder(self, name, goal_pose, start_joints) -> Optional[List[float]]:
        """Solve `goal_pose` by stepping to it from where the arm starts.

        Interpolates start pose -> goal pose and solves each rung seeded by the
        previous rung's answer, so no single solve has to search far. The
        intermediate solutions are scaffolding for the seeding and nothing else:
        they are thrown away, and the caller free-plans one leg to the joints
        the last rung returns. This is not a Cartesian path -- the arm never
        follows these poses, and a rung that fails is skipped rather than
        aborting, since a midpoint can be blocked while both endpoints are fine.
        """
        start = self._moveit2.compute_fk(joint_state=list(start_joints))
        if start is None:
            self._node.get_logger().warning(
                f"no FK for the state '{name}' is planned from; skipping the IK ladder"
            )
            return None

        p0, p1 = np.asarray(position_xyz(start.pose)), np.asarray(position_xyz(goal_pose))
        r0 = Rot.from_quat(list(quat_xyzw(start.pose)))
        r1 = Rot.from_quat(list(quat_xyzw(goal_pose)))
        # Slerp takes the shortest arc, so the quaternion double cover needs no
        # handling here. from_quat on a 2x4 array rather than Rot.concatenate,
        # which is newer than the scipy this is pinned against.
        slerp = Slerp([0.0, 1.0], Rot.from_quat([r0.as_quat(), r1.as_quat()]))
        # Rungs sized from the motion actually being covered, so a near pregrasp
        # does not pay for a far one's IK calls. Rotation usually binds first.
        spacing = max(
            math.dist(p0, p1) / _LADDER_STEP_M,
            float((r0.inv() * r1).magnitude()) / _LADDER_STEP_RAD,
        )
        steps = min(_LADDER_MAX_STEPS, max(1, math.ceil(spacing)))

        seed, skipped = list(start_joints), 0
        for i in range(1, steps + 1):
            t = i / steps
            # t == 1.0 is the goal pose itself, so the last rung is the real
            # solve and no extra call is needed after the loop.
            rung = self._solve_ik(make_pose(p0 + t * (p1 - p0), slerp(t).as_quat()), seed, name)
            if rung is None:
                if i == steps:
                    self._node.get_logger().info(
                        f"IK ladder for '{name}' solved {i - 1 - skipped} of its "
                        f'{steps - 1} intermediate rungs but not the goal pose itself'
                    )
                    return None
                skipped += 1
                continue
            seed = rung
        self._node.get_logger().info(
            f"IK for '{name}' needed a {steps}-rung ladder ({skipped} rungs unsolved)"
        )
        return seed

    def _plan_linear(self, segment, start_joints):
        """Straight line in task space, via /compute_cartesian_path.
        """
        trajectory = self._moveit2.plan(
            pose=segment.command.pose_goal,
            frame_id=segment.command.header.frame_id or None,
            start_joint_state=start_joints,
            cartesian=True,
            max_step=self._linear['max_step_m'],
            cartesian_fraction_threshold=self._linear['fraction_threshold'],
        )
        return trajectory

    def _fit_to_joint_limits(self, trajectory):
        """Slow a trajectory down uniformly until it satisfies the joint limits.

        Stretching time by k scales velocity by 1/k but acceleration by 1/k^2, so
        the two limits need different roots of the same k and the binding one is
        whichever demands more. Checking velocity alone -- as this did -- leaves
        an acceleration-bound trajectory untouched, because its velocities are
        already legal and the early return fires before the 1/k^2 divide ever
        runs. That is how a Cartesian 'target' leg reached trajectory_bridge at
        joint7 acceleration 0.502 against a 0.500 limit and had the whole pick
        aborted, with velocity nowhere near its own limit.
        """
        if not self._velocity_limits and not self._acceleration_limits:
            return trajectory
        scale = 1.0
        for point in trajectory.points:
            for name, velocity in zip(trajectory.joint_names, point.velocities):
                limit = self._velocity_limits.get(name)
                if limit and abs(velocity) > limit:
                    scale = max(scale, _LIMIT_MARGIN * abs(velocity) / limit)
            for name, acceleration in zip(trajectory.joint_names, point.accelerations):
                limit = self._acceleration_limits.get(name)
                if limit and abs(acceleration) > limit:
                    scale = max(scale, _LIMIT_MARGIN * math.sqrt(abs(acceleration) / limit))
        if scale <= 1.0:
            return trajectory
        self._node.get_logger().info(
            f'slowing segment {scale:.3f}x to fit joint velocity/acceleration limits'
        )
        for point in trajectory.points:
            total = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            total *= scale
            point.time_from_start.sec = int(total)
            point.time_from_start.nanosec = int((total - int(total)) * 1e9)
            point.velocities = [v / scale for v in point.velocities]
            point.accelerations = [a / (scale * scale) for a in point.accelerations]
        return trajectory


def _quantize(joints) -> Tuple[float, ...]:
    """Joint positions as a hashable cache key.

    Rounded because a chained start state arrives as the last point of a
    trajectory rather than the literal config value, so exact equality is the
    wrong test even when the two describe the same arm pose.
    """
    return tuple(round(float(j), 9) for j in joints)


def _describe(pose) -> str:
    """One-line pose summary for a planning-failure log line.

    `reach` is the distance from the planning frame's origin, which is also the
    arm's base: world_to_base_link is an identity fixed joint in the URDF. That
    is the number that separates "the wrist cannot hold that orientation" from
    "that point is simply too far away" -- the arm's fully-extended reach is
    roughly 0.78 m (0.138 + 0.31 + 0.27 + the wrist offsets).
    """
    x, y, z = position_xyz(pose)
    roll, pitch, yaw = Rot.from_quat(list(quat_xyzw(pose))).as_euler('xyz')
    return (
        f'xyz [{x:.4f}, {y:.4f}, {z:.4f}] '
        f'rpy [{roll:.4f}, {pitch:.4f}, {yaw:.4f}] '
        f'reach {math.dist((x, y, z), (0.0, 0.0, 0.0)):.4f} m'
    )


def _load_joint_limits(path, node):
    """(velocity, acceleration) limit maps from joint_limits.yaml."""
    if not path:
        node.get_logger().warning(
            'moveit.joint_limits_yaml not set; planned segments will not be '
            'checked against joint velocity or acceleration limits'
        )
        return {}, {}
    with open(path) as f:
        limits = yaml.safe_load(f).get('joint_limits', {})
    velocity = {
        name: v['max_velocity']
        for name, v in limits.items()
        if v.get('has_velocity_limits')
    }
    acceleration = {
        name: v['max_acceleration']
        for name, v in limits.items()
        if v.get('has_acceleration_limits')
    }
    return velocity, acceleration
