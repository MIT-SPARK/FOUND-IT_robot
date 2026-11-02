"""Turn one grasp or place target into an ordered task-space waypoint sequence.

Pure geometry and config: no ROS node, no TF, no planner, no robot. That is the
point -- the shape of a pick can be inspected and unit-tested on its own, and
the same six segments serve a place with only the config block differing.

The sequence replaces what pose_commander.py used to do, which was to plan one
blind move from home straight onto the grasp:

    staging   FREE    a predefined waypoint with a known joint solution
    pregrasp  FREE    standoff, backed off along the gripper's approach axis
    target    LINEAR  the grasp/place pose itself, reached in a straight line
    retreat   LINEAR  straight back out, along the approach axis or world Z
    staging   FREE    back to the same known-good waypoint
    home      FREE    and home

Only segments 2-4 depend on the target, and they all share its orientation, so
the approach and retreat are pure translations along a fixed tool axis -- which
is exactly what makes them solvable as straight lines.
"""

from typing import Dict, List

from geometry_msgs.msg import PoseStamped, Twist
from joey_interfaces.msg import PoseCommand
from std_msgs.msg import Header

from .geometry import (
    GRIPPER_APPROACH_AXIS,
    WORLD_Z_AXIS,
    normalized_pose,
    pose_from_rpy,
    translate_along_local_axis,
    translate_along_world_axis,
)
from .segment import Grip, Motion, Segment

RETREAT_AXES = ('approach', 'world_z')


class TaskPathGenerator:
    """Builds the waypoint sequence for a pick or a place from config."""

    def __init__(self, config: Dict):
        self._frames = config['frames']
        self._named = config['named_configurations']
        self._tasks = config['tasks']
        for task, cfg in self._tasks.items():
            if cfg['retreat_axis'] not in RETREAT_AXES:
                raise ValueError(
                    f"task '{task}' has retreat_axis '{cfg['retreat_axis']}'; "
                    f'expected one of {RETREAT_AXES}'
                )
            if cfg['staging'] not in self._named:
                raise ValueError(
                    f"task '{task}' stages through '{cfg['staging']}', which is not "
                    f'a named_configuration ({sorted(self._named)})'
                )
        if 'home' not in self._named:
            raise ValueError("named_configurations must define 'home'")
        # Poses derived at runtime (by FK) for configurations that have none in
        # the file, so a backend with no joint-space interface can still reach them.
        self._derived_poses = {}

    @property
    def planning_frame(self) -> str:
        return self._frames['planning']

    def move_to_named(self, configuration: str) -> List[Segment]:
        """A one-segment sequence that just goes to a named configuration.

        A list rather than a bare Segment so it drops straight into
        backend.plan() and the node's execute path with no special-casing.
        """
        if configuration not in self._named:
            raise ValueError(
                f"unknown named configuration '{configuration}'; "
                f'expected one of {sorted(self._named)}'
            )
        return [self._named_segment(configuration, configuration)]

    def set_named_pose(self, configuration: str, pose) -> None:
        """Supply a pose for a configuration whose config entry omits one.

        The node derives these by FK at startup, since it is the half of the
        system with a move_group to ask.
        """
        if configuration not in self._named:
            raise ValueError(f"unknown named configuration '{configuration}'")
        self._derived_poses[configuration] = pose

    def generate(self, target: PoseStamped, task: str) -> List[Segment]:
        """Waypoints for `task` ('pick' or 'place') against `target`.

        `target` is the gripper pose to reach -- a grasp for a pick, a release
        pose for a place -- expressed in the planning frame.
        """
        if task not in self._tasks:
            raise ValueError(f"unknown task '{task}'; expected one of {sorted(self._tasks)}")
        cfg = self._tasks[task]

        if target.header.frame_id and target.header.frame_id != self.planning_frame:
            raise ValueError(
                f"target is in frame '{target.header.frame_id}' but the planner "
                f"works in '{self.planning_frame}'; transform it before calling"
            )

        target_pose = normalized_pose(target.pose)

        pregrasp_pose = translate_along_local_axis(
            target_pose, GRIPPER_APPROACH_AXIS, -cfg['approach_distance_m']
        )

        if cfg['retreat_axis'] == 'approach':
            retreat_pose = translate_along_local_axis(
                target_pose, GRIPPER_APPROACH_AXIS, -cfg['retreat_distance_m']
            )
        else:
            retreat_pose = translate_along_world_axis(
                target_pose, WORLD_Z_AXIS, cfg['retreat_distance_m']
            )

        staging = cfg['staging']
        return [
            self._named_segment('staging', staging, grip=_grip(cfg.get('grip_at_staging'))),
            self._pose_segment('pregrasp', Motion.FREE, pregrasp_pose),
            self._pose_segment(
                'target', Motion.LINEAR, target_pose, grip=_grip(cfg.get('grip_at_target'))
            ),
            self._pose_segment('retreat', Motion.LINEAR, retreat_pose),
            self._named_segment('staging_return', staging),
            self._named_segment('home', 'home'),
        ]

    def named_pose(self, configuration: str):
        """A named configuration's pose: recorded in config, derived, or None.

        Config wins, so a hand-captured pose is never silently overridden by a
        derived one -- the node cross-checks the two instead.
        """
        pose = self._named[configuration].get('pose')
        if pose:
            return pose_from_rpy(pose['position'], pose['rpy'])
        return self._derived_poses.get(configuration)

    def _named_segment(self, name: str, configuration: str, grip=None) -> Segment:
        """A waypoint from named_configurations, carrying both goal forms.
        """
        entry = self._named[configuration]
        pose = self.named_pose(configuration)
        return Segment(
            name=name,
            motion=Motion.FREE,
            command=self._command(pose) if pose else None,
            joints=list(entry['joints']),
            grip=grip,
        )

    def _pose_segment(self, name: str, motion: Motion, pose, grip=None) -> Segment:
        return Segment(name=name, motion=motion, command=self._command(pose), grip=grip)

    def _command(self, pose) -> PoseCommand:
        # v_goal is left at zero throughout: the arm comes to rest at every
        # waypoint. MoveIt ignores it (it does its own time parameterization
        # from joint_limits.yaml) and impedance_control holds one setpoint at a
        # time, so a non-zero pass-through velocity would be fiction today. The
        # field is carried because it becomes meaningful the moment a backend
        # streams a blended trajectory rather than stepping setpoint to setpoint.
        return PoseCommand(
            header=Header(frame_id=self.planning_frame),
            pose_goal=pose,
            v_goal=Twist(),
        )


def _grip(value) -> Grip:
    return Grip(value) if value else None
