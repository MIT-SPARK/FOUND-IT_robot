"""The interface between path generation and motion execution.

A `Segment` is one leg of a manipulation task: where to go, how strictly the
path there matters, and what the gripper should do on arrival. It is the only
type `TaskPathGenerator` produces and the only type a `MotionBackend` consumes,
which is what lets the backend be swapped (MoveIt today, an impedance
controller next) without either side of the seam knowing about the other.

Deliberately not a custom `.msg`: this is an in-process producer/consumer pair,
and adding a ROS interface would mean converting the package to `ament_cmake`
or spawning a message package for a dataclass. The task-space goal it carries
*is* a standard message -- `joey_interfaces/PoseCommand`, the same type
impedance_control already subscribes to.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from joey_interfaces.msg import PoseCommand


class Motion(str, Enum):
    """How much the path between waypoints matters.

    Inherits from `str` so a config file can name these directly and so log
    lines read as words rather than `<Motion.LINEAR: 'linear'>`.
    """

    #: The backend may take any collision-free path it likes (CHOMP, or an
    #: impedance controller simply servoing to the setpoint).
    FREE = 'free'
    #: The end effector must travel in a straight line in task space. Used for
    #: the final approach to a grasp and the retreat away from it, where
    #: sweeping through an arc would clip the object or the surface it sits on.
    LINEAR = 'linear'


class Grip(str, Enum):
    OPEN = 'open'
    CLOSE = 'close'


@dataclass(frozen=True)
class Segment:
    """
    One leg of a manipulation task. -- `grip` is applied *after* arriving.
    """
    name: str
    motion: Motion
    command: Optional[PoseCommand] = None
    joints: Optional[List[float]] = None
    grip: Optional[Grip] = None

    def __post_init__(self):
        if self.command is None and self.joints is None:
            raise ValueError(
                f"segment '{self.name}' has neither a task-space command nor a "
                'joint-space goal, so no backend could execute it'
            )
