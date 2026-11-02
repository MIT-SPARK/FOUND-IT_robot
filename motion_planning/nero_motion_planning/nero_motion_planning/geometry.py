"""Pose arithmetic for grasp waypoints, in the gripper's own frame.

Lifted out of pose_commander.py so the geometry can be read, tested, and
reasoned about without a ROS node around it. Everything here is a pure function
over geometry_msgs types.
"""

import math
from typing import Tuple

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion, TransformStamped
from scipy.spatial.transform import Rotation as Rot

# gripper_flange's local +Z is the gripper's approach/insertion axis; +Y is the
# finger-to-finger closing axis. 
GRIPPER_APPROACH_AXIS = (0.0, 0.0, 1.0)
GRIPPER_CLOSING_AXIS = (0.0, 1.0, 0.0)

WORLD_Z_AXIS = (0.0, 0.0, 1.0)


def normalized_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float, float, float]:
    """
    Renormalize a quaternion, erroring on a degenerate one.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError(f'degenerate quaternion ({x}, {y}, {z}, {w}) cannot be normalized')
    return (x / norm, y / norm, z / norm, w / norm)


def quat_xyzw(pose: Pose) -> Tuple[float, float, float, float]:
    q = pose.orientation
    return normalized_quaternion(q.x, q.y, q.z, q.w)


def position_xyz(pose: Pose) -> Tuple[float, float, float]:
    p = pose.position
    return (p.x, p.y, p.z)


def make_pose(position, quat) -> Pose:
    x, y, z, w = normalized_quaternion(*quat)
    return Pose(
        position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
        orientation=Quaternion(x=float(x), y=float(y), z=float(z), w=float(w)),
    )


def normalized_pose(pose: Pose) -> Pose:
    """Copy of `pose` with a unit-norm orientation. See normalized_quaternion."""
    return make_pose(position_xyz(pose), quat_xyzw(pose))


def rotation_matrix(pose: Pose) -> np.ndarray:
    return Rot.from_quat(list(quat_xyzw(pose))).as_matrix()


def pose_from_rpy(position, rpy) -> Pose:
    """
    Build a Pose from a position and roll/pitch/yaw.
    """
    return make_pose(position, Rot.from_euler('xyz', list(rpy)).as_quat())


def top_down_pose(position, yaw_rad: float = 0.0) -> Pose:
    """
    Build a gripper pose whose approach axis points straight down.
    """
    # rpy (pi, 0, yaw) puts flange +Z on world -Z and flange +X at yaw in the
    # horizontal plane, leaving yaw as the family's only free parameter.
    #
    # Which yaw the arm can actually hold is a fact about this wrist rather than
    # a convention: joint7's axis is flange -Y (the closing axis, not the
    # approach axis) and joint6 spans only [-0.73, 0.95] rad, so a top-down pose
    # is not free to roll -- yaw is paid for with the whole wrist. Measured
    # against the joint limits, what it wants is flange +X pointing radially
    # outward from the base, within roughly +/-60 degrees. That direction
    # depends on where the target is, so the caller picks yaw; see
    # manipulation.yaml's place_target.yaws_deg for the ordering in use and the
    # margins behind it.
    return pose_from_rpy(position, (math.pi, 0.0, yaw_rad))


def translate_along_local_axis(pose: Pose, axis, distance: float) -> Pose:
    """Translate `distance` along `axis` expressed in the pose's *own* frame.
    """
    offset_world = rotation_matrix(pose) @ (np.asarray(axis, dtype=float) * distance)
    return make_pose(np.asarray(position_xyz(pose)) + offset_world, quat_xyzw(pose))


def translate_along_world_axis(pose: Pose, axis, distance: float) -> Pose:
    """Translate `distance` along `axis` in the *planning (world) frame*, not the tool's.
    """
    offset = np.asarray(axis, dtype=float) * distance
    return make_pose(np.asarray(position_xyz(pose)) + offset, quat_xyzw(pose))


def rotate_about_local_axis(pose: Pose, axis, angle_rad: float) -> Pose:
    """Rotate the orientation about `axis` in the pose's own frame; position held."""
    delta = Rot.from_rotvec(np.asarray(axis, dtype=float) * angle_rad)
    rotated = Rot.from_quat(list(quat_xyzw(pose))) * delta
    return make_pose(position_xyz(pose), rotated.as_quat())


def flip_180_about_approach_axis(pose: Pose) -> Pose:
    """Rotate a grasp 180 degrees about the approach (local Z) axis.
    """
    return rotate_about_local_axis(pose, GRIPPER_APPROACH_AXIS, math.pi)


def poses_close(a: Pose, b: Pose, position_tol: float, orientation_tol: float) -> bool:
    position_delta = math.dist(position_xyz(a), position_xyz(b))
    # The dot product of two unit quaternions is ±1 for identical orientations
    # (± because q and -q are the same rotation), so abs() then acos gives the
    # angle between them without a full relative-rotation computation.
    dot = sum(ac * bc for ac, bc in zip(quat_xyzw(a), quat_xyzw(b)))
    orientation_delta = math.acos(min(1.0, abs(dot))) * 2.0
    return position_delta < position_tol and orientation_delta < orientation_tol


def pose_from_transform(transform: TransformStamped) -> Pose:
    t = transform.transform.translation
    q = transform.transform.rotation
    return make_pose((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))


def transform_point(transform: TransformStamped, point) -> Tuple[float, float, float]:
    """
    Apply a TF transform to a bare xyz point.
    """
    # One matrix multiply, so it stays here rather than pulling in
    # tf2_geometry_msgs -- which would also mean this module could no longer be
    # exercised without a ROS environment around it.
    frame = pose_from_transform(transform)
    moved = rotation_matrix(frame) @ np.asarray(point, dtype=float) + np.asarray(
        position_xyz(frame)
    )
    return (float(moved[0]), float(moved[1]), float(moved[2]))
