"""Unit tests for the pure pose math -- no ROS graph, no robot, no planner."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as Rot

from nero_motion_planning.geometry import (
    GRIPPER_APPROACH_AXIS,
    GRIPPER_CLOSING_AXIS,
    position_xyz,
    quat_xyzw,
    rotation_matrix,
    top_down_pose,
    transform_point,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'manipulation.yaml'


@pytest.fixture
def place_yaws_rad():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    return [math.radians(d) for d in config['place_target']['yaws_deg']]


def make_transform(rpy=(0.0, 0.0, 0.0), xyz=(0.0, 0.0, 0.0)):
    transform = TransformStamped()
    transform.transform.translation.x = float(xyz[0])
    transform.transform.translation.y = float(xyz[1])
    transform.transform.translation.z = float(xyz[2])
    quat = Rot.from_euler('xyz', list(rpy)).as_quat()
    transform.transform.rotation.x = float(quat[0])
    transform.transform.rotation.y = float(quat[1])
    transform.transform.rotation.z = float(quat[2])
    transform.transform.rotation.w = float(quat[3])
    return transform


def test_top_down_pose_approaches_straight_down(place_yaws_rad):
    # Every configured yaw has to leave the approach axis pointing at the floor;
    # that is the whole premise of a place, and yaw is meant to be the only
    # degree of freedom the fan explores.
    for yaw in place_yaws_rad + [0.0, math.pi, -2.0, 7.0]:
        approach = rotation_matrix(top_down_pose((0.4, 0.0, 0.2), yaw)) @ np.asarray(
            GRIPPER_APPROACH_AXIS
        )
        np.testing.assert_allclose(approach, [0.0, 0.0, -1.0], atol=1e-12)


def test_top_down_pose_yaw_rotates_about_world_z():
    for degrees in (0.0, -30.0, 30.0, -60.0, 60.0, 180.0):
        yaw = math.radians(degrees)
        rotation = rotation_matrix(top_down_pose((0.4, -0.1, 0.2), yaw))
        # Flange +X sweeps the horizontal plane; the closing axis stays
        # horizontal with it, so the fingers never tilt out of level.
        np.testing.assert_allclose(
            rotation @ np.asarray((1.0, 0.0, 0.0)),
            [math.cos(yaw), math.sin(yaw), 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            (rotation @ np.asarray(GRIPPER_CLOSING_AXIS))[2], 0.0, atol=1e-12
        )


def test_top_down_pose_at_yaw_zero_is_a_half_turn_about_world_x():
    # The natural branch, and worth pinning by value: quaternion (1, 0, 0, 0).
    # See top_down_pose for why this one and not its 180-degree twin.
    pose = top_down_pose((0.45, 0.0, 0.2), 0.0)
    np.testing.assert_allclose(quat_xyzw(pose), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_top_down_pose_keeps_the_position():
    position = (0.45, -0.12, 0.31)
    np.testing.assert_allclose(
        position_xyz(top_down_pose(position, math.radians(-30.0))), position, atol=1e-12
    )


def test_transform_point_under_identity_is_the_point():
    np.testing.assert_allclose(
        transform_point(make_transform(), (0.4, -0.1, 0.2)), (0.4, -0.1, 0.2), atol=1e-12
    )


def test_transform_point_applies_rotation_then_translation():
    # A quarter turn about Z maps +X to +Y, then the offset is added -- in that
    # order, which is what a translation-first bug would get wrong.
    transform = make_transform(rpy=(0.0, 0.0, math.pi / 2), xyz=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        transform_point(transform, (1.0, 0.0, 0.0)), (1.0, 3.0, 3.0), atol=1e-12
    )


def test_transform_point_matches_the_equivalent_matrix_product():
    transform = make_transform(rpy=(0.3, -0.7, 1.1), xyz=(0.2, -0.4, 0.6))
    point = np.asarray((0.31, -0.22, 0.17))
    expected = Rot.from_euler('xyz', [0.3, -0.7, 1.1]).as_matrix() @ point + np.asarray(
        (0.2, -0.4, 0.6)
    )
    np.testing.assert_allclose(transform_point(transform, point), expected, atol=1e-12)
