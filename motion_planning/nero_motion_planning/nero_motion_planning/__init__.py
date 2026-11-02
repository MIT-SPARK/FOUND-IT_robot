"""Task-space manipulation sequencing for the NERO arm.

Two halves with a narrow seam between them:

  TaskPathGenerator  a grasp or place target -> an ordered list of Segments
                     (staging, pregrasp standoff, linear approach, gripper
                     action, linear retreat, staging, home). Pure geometry and
                     config; no ROS graph, no planner, no robot.

  MotionBackend      a list of Segments -> arm motion. MoveIt today; an
                     impedance controller consuming joey_interfaces/PoseCommand
                     next, behind the same ABC.  [phase 1b -- not yet written]

Segment is the seam: it carries a PoseCommand (the message impedance_control
already subscribes to) and, for waypoints that have one, a known-good joint
solution. Each backend takes what it can use.
"""

from .geometry import (
    GRIPPER_APPROACH_AXIS,
    GRIPPER_CLOSING_AXIS,
    flip_180_about_approach_axis,
    poses_close,
    top_down_pose,
    transform_point,
)
from .path_generator import TaskPathGenerator
from .segment import Grip, Motion, Segment

__all__ = [
    'GRIPPER_APPROACH_AXIS',
    'GRIPPER_CLOSING_AXIS',
    'Grip',
    'Motion',
    'Segment',
    'TaskPathGenerator',
    'flip_180_about_approach_axis',
    'poses_close',
    'top_down_pose',
    'transform_point',
]
