"""Unit tests for TaskPathGenerator -- no ROS graph, no robot, no planner."""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as Rot

from nero_motion_planning.geometry import (
    GRIPPER_APPROACH_AXIS,
    make_pose,
    position_xyz,
    quat_xyzw,
    top_down_pose,
)
from nero_motion_planning.path_generator import TaskPathGenerator
from nero_motion_planning.segment import Grip, Motion

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'manipulation.yaml'
EXPECTED_NAMES = ['staging', 'pregrasp', 'target', 'retreat', 'staging_return', 'home']


@pytest.fixture
def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture
def generator(config):
    return TaskPathGenerator(config)


def make_target(position=(0.4, 0.1, 0.3), rpy=(math.pi / 2, 0.2, -0.7), frame='world'):
    # Deliberately tilted, so a bug that offsets along world -Z instead of the
    # tool axis cannot pass by coincidence.
    stamped = PoseStamped()
    stamped.header.frame_id = frame
    stamped.pose = make_pose(position, Rot.from_euler('xyz', rpy).as_quat())
    return stamped


def segments_by_name(segments):
    return {s.name: s for s in segments}


def test_sequence_shape(generator):
    for task in ('pick', 'place'):
        segments = generator.generate(make_target(), task)
        assert [s.name for s in segments] == EXPECTED_NAMES, task
        assert [s.motion for s in segments] == [
            Motion.FREE, Motion.FREE, Motion.LINEAR, Motion.LINEAR, Motion.FREE, Motion.FREE
        ], task


def test_named_waypoints_carry_joints(generator, config):
    by_name = segments_by_name(generator.generate(make_target(), 'pick'))
    for name, configuration in (
        ('staging', 'staging'), ('staging_return', 'staging'), ('home', 'home')
    ):
        assert by_name[name].joints == config['named_configurations'][configuration]['joints'], name
    # staging has a pose in config, home does not -- so only staging can also
    # offer a task-space goal to a backend with no joint-space interface.
    assert by_name['staging'].command is not None
    assert by_name['home'].command is None


def test_perception_waypoints_carry_no_joints(generator):
    by_name = segments_by_name(generator.generate(make_target(), 'pick'))
    for name in ('pregrasp', 'target', 'retreat'):
        assert by_name[name].joints is None, name
        assert by_name[name].command is not None, name


def test_pregrasp_is_backed_off_along_the_targets_own_approach_axis(generator, config):
    target = make_target()
    by_name = segments_by_name(generator.generate(target, 'pick'))
    distance = config['tasks']['pick']['approach_distance_m']

    delta = (np.array(position_xyz(by_name['pregrasp'].command.pose_goal))
             - np.array(position_xyz(target.pose)))
    expected = Rot.from_quat(list(quat_xyzw(target.pose))).as_matrix() @ (
        np.array(GRIPPER_APPROACH_AXIS) * -distance
    )
    np.testing.assert_allclose(delta, expected, atol=1e-9)
    assert math.isclose(np.linalg.norm(delta), distance, abs_tol=1e-9)

    # The tilted target means the standoff must NOT be a pure world-Z move --
    # this is the assertion that actually catches a local/world axis mixup.
    assert abs(delta[0]) > 1e-3 and abs(delta[1]) > 1e-3


def test_orientation_constant_across_approach_and_retreat(generator):
    target = make_target()
    by_name = segments_by_name(generator.generate(target, 'pick'))
    reference = quat_xyzw(target.pose)
    for name in ('pregrasp', 'target', 'retreat'):
        np.testing.assert_allclose(
            quat_xyzw(by_name[name].command.pose_goal), reference, atol=1e-9,
            err_msg=f'{name} rotates; the LINEAR legs would stop being pure translations',
        )


def retreat_delta(config, axis):
    """Where 'retreat' lands relative to the target, for a given retreat_axis."""
    config = copy.deepcopy(config)
    config['tasks']['pick']['retreat_axis'] = axis
    target = make_target()
    segments = TaskPathGenerator(config).generate(target, 'pick')
    retreat = segments_by_name(segments)['retreat'].command.pose_goal
    return np.array(position_xyz(retreat)) - np.array(position_xyz(target.pose))


def test_approach_retreat_axis_backs_out_along_the_tool(config):
    distance = config['tasks']['pick']['retreat_distance_m']
    expected = Rot.from_quat(list(quat_xyzw(make_target().pose))).as_matrix() @ (
        np.array(GRIPPER_APPROACH_AXIS) * -distance
    )
    np.testing.assert_allclose(retreat_delta(config, 'approach'), expected, atol=1e-9)


def test_world_z_retreat_axis_lifts_straight_up(config):
    # Whatever the gripper's orientation -- the option for lifting off a surface
    # without dragging the gripper sideways across it.
    distance = config['tasks']['pick']['retreat_distance_m']
    np.testing.assert_allclose(
        retreat_delta(config, 'world_z'), [0.0, 0.0, distance], atol=1e-9
    )


@pytest.mark.parametrize('task', ['pick', 'place'])
def test_grips_come_from_config(generator, config, task):
    by_name = segments_by_name(generator.generate(make_target(), task))
    cfg = config['tasks'][task]
    expected_staging = Grip(cfg['grip_at_staging']) if cfg['grip_at_staging'] else None
    assert by_name['staging'].grip is expected_staging
    assert by_name['target'].grip is Grip(cfg['grip_at_target'])
    # Only the two waypoints config names ever actuate the gripper.
    for name in ('pregrasp', 'retreat', 'staging_return', 'home'):
        assert by_name[name].grip is None, name


def test_staging_and_target_are_task_independent(generator, config):
    # A pick and a place aim at the same waypoints; what differs is grip,
    # retreat_axis, and the standoff distances. The standoff is deliberately
    # per-task: place's top-down envelope runs out about 0.26 m up at the
    # measured drop point, so its pregrasp has to sit lower than a pick's.
    target = make_target()
    pick = segments_by_name(generator.generate(target, 'pick'))
    place = segments_by_name(generator.generate(target, 'place'))
    for name in ('staging', 'target'):
        np.testing.assert_allclose(
            position_xyz(pick[name].command.pose_goal),
            position_xyz(place[name].command.pose_goal),
            atol=1e-12, err_msg=name,
        )

    # Each task's pregrasp is its own approach_distance_m back along the shared
    # approach axis, so the two lie on one line through the target.
    at = {task: np.asarray(position_xyz(by_name['pregrasp'].command.pose_goal))
          for task, by_name in (('pick', pick), ('place', place))}
    origin = np.asarray(position_xyz(target.pose))
    for task, position in at.items():
        distance = config['tasks'][task]['approach_distance_m']
        assert math.isclose(
            float(np.linalg.norm(position - origin)), distance, abs_tol=1e-9
        ), task
    pick_offset, place_offset = at['pick'] - origin, at['place'] - origin
    np.testing.assert_allclose(
        np.cross(pick_offset, place_offset), [0.0, 0.0, 0.0], atol=1e-12
    )


def test_a_top_down_place_approaches_and_retreats_vertically(generator, config):
    # The case the system actually runs: a place target built by top_down_pose.
    # Its approach axis is world -Z, so the standoff and the retreat both have to
    # come out as pure lifts in world Z -- which is what makes the two LINEAR
    # legs a straight drop onto the drop point and straight back up off it.
    target = PoseStamped()
    target.header.frame_id = config['frames']['planning']
    target.pose = top_down_pose((0.45, 0.0, 0.2), 0.0)

    cfg = config['tasks']['place']
    by_name = segments_by_name(generator.generate(target, 'place'))
    at = {name: np.asarray(position_xyz(by_name[name].command.pose_goal))
          for name in ('pregrasp', 'target', 'retreat')}

    np.testing.assert_allclose(at['target'], [0.45, 0.0, 0.2], atol=1e-12)
    np.testing.assert_allclose(
        at['pregrasp'] - at['target'], [0.0, 0.0, cfg['approach_distance_m']], atol=1e-12
    )
    np.testing.assert_allclose(
        at['retreat'] - at['target'], [0.0, 0.0, cfg['retreat_distance_m']], atol=1e-12
    )
    # The gripper never reorients along the way, so the drop stays a straight line.
    for name in ('pregrasp', 'retreat'):
        np.testing.assert_allclose(
            quat_xyzw(by_name[name].command.pose_goal),
            quat_xyzw(by_name['target'].command.pose_goal),
            atol=1e-12, err_msg=name,
        )


def test_velocity_goals_are_zero(generator):
    # The arm rests at every waypoint; a non-zero pass-through velocity would be
    # fiction until a backend streams a blended trajectory.
    for segment in generator.generate(make_target(), 'pick'):
        if segment.command is None:
            continue
        v = segment.command.v_goal
        assert (v.linear.x, v.linear.y, v.linear.z) == (0.0, 0.0, 0.0), segment.name
        assert (v.angular.x, v.angular.y, v.angular.z) == (0.0, 0.0, 0.0), segment.name


def test_rejects_target_in_a_foreign_frame(generator):
    with pytest.raises(ValueError, match='camera_link'):
        generator.generate(make_target(frame='camera_link'), 'pick')


def test_rejects_unknown_task(generator):
    with pytest.raises(ValueError, match='wave'):
        generator.generate(make_target(), 'wave')


def test_rejects_bad_retreat_axis(config):
    config['tasks']['pick']['retreat_axis'] = 'sideways'
    with pytest.raises(ValueError, match='retreat_axis'):
        TaskPathGenerator(config)


def test_rejects_staging_that_is_not_a_named_configuration(config):
    config['tasks']['pick']['staging'] = 'nowhere'
    with pytest.raises(ValueError, match='nowhere'):
        TaskPathGenerator(config)


def test_named_pose_round_trips(generator, config):
    # named_pose() is what the node's FK cross-check compares against.
    pose = generator.named_pose('staging')
    np.testing.assert_allclose(
        position_xyz(pose),
        config['named_configurations']['staging']['pose']['position'], atol=1e-12,
    )
    # home has no pose in config, and none derived yet on a fresh generator.
    assert generator.named_pose('home') is None


def test_move_to_named_is_a_one_segment_sequence(generator, config):
    segments = generator.move_to_named('home')
    assert len(segments) == 1
    home = segments[0]
    assert home.name == 'home'
    assert home.motion is Motion.FREE
    assert home.joints == config['named_configurations']['home']['joints']
    assert home.grip is None


def test_move_to_named_rejects_an_unknown_configuration(generator):
    with pytest.raises(ValueError, match='parked'):
        generator.move_to_named('parked')


def test_derived_pose_makes_home_reachable_by_a_task_space_backend(generator):
    # Without this, ImpedanceBackend could never home: it has no joint-space
    # interface, so a segment carrying only `joints` is unexecutable for it.
    assert generator.move_to_named('home')[0].command is None

    derived = make_pose((0.1, 0.2, 0.3), Rot.from_euler('xyz', (0.0, 0.1, 0.2)).as_quat())
    generator.set_named_pose('home', derived)

    command = generator.move_to_named('home')[0].command
    assert command is not None
    np.testing.assert_allclose(position_xyz(command.pose_goal), (0.1, 0.2, 0.3), atol=1e-12)
    # Still carries joints too, so MoveItBackend keeps preferring the exact solution.
    assert generator.move_to_named('home')[0].joints is not None


def test_a_recorded_pose_is_not_overridden_by_a_derived_one(generator, config):
    # Config wins, so the node cross-checks a hand-captured pose against FK
    # rather than silently replacing it.
    generator.set_named_pose('staging', make_pose((9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0)))
    np.testing.assert_allclose(
        position_xyz(generator.named_pose('staging')),
        config['named_configurations']['staging']['pose']['position'], atol=1e-12,
    )
