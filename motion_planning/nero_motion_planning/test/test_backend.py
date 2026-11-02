"""Tests for MoveItBackend's two subtle behaviours, with move_group faked out.

The velocity-limit fitting and the start-state chaining are the parts most
likely to be silently wrong, and neither needs a robot to check.
"""

import math

import pytest
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from joey_interfaces.msg import PoseCommand
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from nero_motion_planning.backend import _LADDER_STEP_M, _LIMIT_MARGIN, MoveItBackend
from nero_motion_planning.segment import Motion, Segment

JOINTS = ['joint1', 'joint2']
LIMIT = 0.5
# Seeds _plan_free_pose tries after the chained state and the ladder, in order.
STAGING_JOINTS = [0.11, 0.11]
HOME_JOINTS = [0.22, 0.22]
# Where the fake arm starts in task space. pose_segment()'s goal is the origin,
# so the ladder spans LADDER_RUNGS rungs at _LADDER_STEP_M; the orientation
# matches the goal's, leaving position the only thing that sets the rung count.
FK_START_X = 0.3
LADDER_RUNGS = round(FK_START_X / _LADDER_STEP_M)

# Distinguishes "this test has no canned FK" from a test that cans None to say
# the FK service itself is unavailable.
_UNSET = object()


class FakeLogger:
    def __init__(self):
        self.messages = []

    def _record(self, msg):
        self.messages.append(msg)

    info = warning = warn = error = _record


class FakeNode:
    def __init__(self):
        self._logger = FakeLogger()

    def get_logger(self):
        return self._logger


class FakeMoveIt2:
    """Records what it was asked to plan and hands back canned trajectories."""

    def __init__(self, trajectories, ik_results=None, fk_result=_UNSET):
        self._trajectories = list(trajectories)
        self._ik_results = list(ik_results) if ik_results is not None else None
        self._fk_result = fk_result
        self.start_states = []
        self.goal_states = []
        self.ik_seeds = []
        self.fk_states = []
        self.cartesian_avoid_collisions = None
        self.pipeline_id = None
        self.max_velocity = None
        self.max_acceleration = None

    def plan(self, **kwargs):
        self.start_states.append(kwargs.get('start_joint_state'))
        self.goal_states.append(kwargs.get('joint_positions'))
        return self._trajectories.pop(0) if self._trajectories else None

    def compute_ik(self, **kwargs):
        if self._ik_results is None:
            raise AssertionError('this test only uses joint-goal segments')
        self.ik_seeds.append(kwargs.get('start_joint_state'))
        return self._ik_results.pop(0) if self._ik_results else None

    def compute_fk(self, **kwargs):
        if self._fk_result is _UNSET:
            raise AssertionError('this test did not expect the IK ladder to run')
        self.fk_states.append(kwargs.get('joint_state'))
        return self._fk_result


def trajectory(*point_specs):
    """(time_s, positions, velocities) triples -> a JointTrajectory."""
    traj = JointTrajectory(joint_names=list(JOINTS))
    for t, positions, velocities in point_specs:
        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.velocities = list(velocities)
        point.accelerations = [0.25] * len(positions)
        point.time_from_start = Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
        traj.points.append(point)
    return traj


def pose_segment(name='pregrasp'):
    """A FREE segment with a task-space goal, so IK is the only way to solve it."""
    pose = Pose(orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    return Segment(name=name, motion=Motion.FREE, command=PoseCommand(pose_goal=pose))


def fk_start_pose():
    """The FK of the state a segment is planned from: FK_START_X from the goal."""
    return PoseStamped(
        pose=Pose(
            position=Point(x=FK_START_X, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )


def ik(value):
    """A canned compute_ik answer, both joints at `value`."""
    return JointState(name=list(JOINTS), position=[value, value])


def make_backend(trajectories, enforce=True, limits=True, ik_results=None, fk_result=_UNSET):
    node = FakeNode()
    moveit2 = FakeMoveIt2(trajectories, ik_results, fk_result)
    config = {
        'moveit': {
            'free_pipeline': 'chomp',
            'linear': {'max_step_m': 0.005, 'fraction_threshold': 0.99,
                       'avoid_collisions': True},
            'enforce_joint_limits': enforce,
            'velocity_scaling': 0.95,
            'acceleration_scaling': 0.95,
            'joint_limits_yaml': '',
        },
        'named_configurations': {
            'staging': {'joints': list(STAGING_JOINTS)},
            'home': {'joints': list(HOME_JOINTS)},
        },
    }
    backend = MoveItBackend(node, moveit2, config, JOINTS)
    if limits:
        backend._velocity_limits = {name: LIMIT for name in JOINTS}
        backend._acceleration_limits = {name: LIMIT for name in JOINTS}
    else:
        backend._velocity_limits, backend._acceleration_limits = {}, {}
    return backend, moveit2, node


def seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def test_start_states_chain_across_segments():
    # Each segment must start where the previous one ended, not at the live pose.
    trajectories = [
        trajectory((1.0, [0.1, 0.1], [0.0, 0.0])),
        trajectory((1.0, [0.2, 0.2], [0.0, 0.0])),
        trajectory((1.0, [0.3, 0.3], [0.0, 0.0])),
    ]
    backend, moveit2, _ = make_backend(trajectories)
    segments = [Segment(name=f's{i}', motion=Motion.FREE, joints=[0.0, 0.0]) for i in range(3)]

    planned = backend.plan(segments, start_joints=[0.0, 0.0])

    assert planned is not None
    assert moveit2.start_states == [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]
    assert planned.final_joints == [0.3, 0.3]


def test_plan_returns_none_and_stops_when_a_segment_fails():
    backend, moveit2, node = make_backend([trajectory((1.0, [0.1, 0.1], [0.0, 0.0]))])
    segments = [Segment(name=f's{i}', motion=Motion.FREE, joints=[0.0, 0.0]) for i in range(3)]

    assert backend.plan(segments, start_joints=[0.0, 0.0]) is None
    # Stopped at the failure rather than planning the rest against a bad state.
    assert len(moveit2.start_states) == 2
    assert any("segment 's1'" in m for m in node.get_logger().messages)


def test_ik_falls_back_to_a_named_configuration_seed():
    # TRAC-IK searches outward from its seed, so a pose with no solution from the
    # chained state often has one from elsewhere. Giving up on the first None is
    # what made every pick candidate fail at 'pregrasp'. fk_result=None takes the
    # ladder out of the picture, leaving the named seeds on their own.
    backend, moveit2, node = make_backend(
        [trajectory((1.0, [0.3, 0.3], [0.0, 0.0]))],
        ik_results=[None, JointState(name=list(JOINTS), position=[0.3, 0.3])],
        fk_result=None,
    )

    planned = backend.plan([pose_segment()], start_joints=[0.0, 0.0])

    assert planned is not None
    # Chained state first (branch continuity), then staging -- and it stopped as
    # soon as one worked rather than walking the rest.
    assert moveit2.ik_seeds == [[0.0, 0.0], STAGING_JOINTS]
    assert any("'staging' seed" in m for m in node.get_logger().messages)


def test_no_ik_from_any_seed_logs_the_pose_that_failed():
    backend, moveit2, node = make_backend([], ik_results=[None, None, None], fk_result=None)

    assert backend.plan([pose_segment()], start_joints=[0.0, 0.0]) is None
    assert moveit2.ik_seeds == [[0.0, 0.0], STAGING_JOINTS, HOME_JOINTS]
    # A missing FK is not fatal -- it only costs the ladder, and the named seeds
    # still get their turn.
    assert any('skipping the IK ladder' in m for m in node.get_logger().messages)
    # The pose itself must reach the log: without it there is no way to tell an
    # out-of-reach waypoint from an orientation the wrist cannot hold.
    message = next(m for m in node.get_logger().messages if 'no IK' in m)
    assert 'xyz' in message and 'rpy' in message and 'reach' in message


def test_a_direct_ik_solution_costs_one_call_and_no_ladder():
    # The common case must stay a single service round-trip: the ladder is a
    # recovery path, not a tax on every pregrasp. FakeMoveIt2 raises if compute_fk
    # is called at all here.
    backend, moveit2, _ = make_backend(
        [trajectory((1.0, [0.3, 0.3], [0.0, 0.0]))], ik_results=[ik(0.3)]
    )

    assert backend.plan([pose_segment()], start_joints=[0.0, 0.0]) is not None
    assert moveit2.ik_seeds == [[0.0, 0.0]]


def test_the_ik_ladder_seeds_each_rung_with_the_last_one_that_solved():
    # The whole point: no solve has to search far, because each starts from the
    # answer to a pose a few centimetres behind it. A rung that fails is skipped
    # rather than aborting -- a midpoint can be blocked while both ends are fine.
    rungs = [ik(0.1), ik(0.2), None, ik(0.4), ik(0.5), ik(0.6)]
    assert len(rungs) == LADDER_RUNGS
    backend, moveit2, node = make_backend(
        [trajectory((1.0, [0.6, 0.6], [0.0, 0.0]))],
        ik_results=[None] + rungs,
        fk_result=fk_start_pose(),
    )

    planned = backend.plan([pose_segment()], start_joints=[0.0, 0.0])

    assert planned is not None
    assert moveit2.fk_states == [[0.0, 0.0]]
    # The direct attempt, then a rung per step: the third rung found nothing, so
    # the fourth was seeded from the second rather than from nothing.
    assert moveit2.ik_seeds == [
        [0.0, 0.0], [0.0, 0.0], [0.1, 0.1], [0.2, 0.2], [0.2, 0.2], [0.4, 0.4], [0.5, 0.5]
    ]
    # Only the last rung's joints are planned to. The intermediate ones are
    # seeding scaffolding and must not turn the leg into a stepped path.
    assert moveit2.goal_states == [[0.6, 0.6]]
    assert any(f'{LADDER_RUNGS}-rung ladder' in m for m in node.get_logger().messages)


def test_a_ladder_that_stops_short_of_the_goal_falls_through_to_the_named_seeds():
    # Reaching the last rung is the only success. Solving every pose but the goal
    # is still no solution, and the named seeds are what is left to try.
    backend, moveit2, node = make_backend(
        [trajectory((1.0, [0.9, 0.9], [0.0, 0.0]))],
        ik_results=[None] + [ik(0.1)] * (LADDER_RUNGS - 1) + [None, ik(0.9)],
        fk_result=fk_start_pose(),
    )

    planned = backend.plan([pose_segment()], start_joints=[0.0, 0.0])

    assert planned is not None
    assert moveit2.ik_seeds[-1] == STAGING_JOINTS
    assert moveit2.goal_states == [[0.9, 0.9]]
    assert any('not the goal pose itself' in m for m in node.get_logger().messages)


def test_an_identical_joint_leg_is_solved_once_per_round():
    # pose_commander opens every candidate's sequence with the same 'staging'
    # move, from the same live start pose. Re-solving it per candidate is what
    # made a 20-candidate walk take minutes.
    backend, moveit2, _ = make_backend([trajectory((1.0, [0.1, 0.1], [0.0, 0.0]))])
    staging = [Segment(name='staging', motion=Motion.FREE, joints=list(STAGING_JOINTS))]

    first = backend.plan(staging, start_joints=[0.0, 0.0])
    second = backend.plan(staging, start_joints=[0.0, 0.0])

    assert first is not None and second is not None
    # One canned trajectory for two plans: the second never reached move_group.
    assert moveit2.start_states == [[0.0, 0.0]]
    # Reused, not shared -- _fit_to_joint_limits edits points in place.
    assert second.trajectories[0] is not first.trajectories[0]
    assert list(second.trajectories[0].points[-1].positions) == [0.1, 0.1]


def test_a_different_start_state_is_planned_afresh():
    # The cache must key on where the arm is, not just where it is going.
    trajectories = [
        trajectory((1.0, [0.1, 0.1], [0.0, 0.0])),
        trajectory((1.0, [0.2, 0.2], [0.0, 0.0])),
    ]
    backend, moveit2, _ = make_backend(trajectories)
    staging = [Segment(name='staging', motion=Motion.FREE, joints=list(STAGING_JOINTS))]

    backend.plan(staging, start_joints=[0.0, 0.0])
    backend.plan(staging, start_joints=[0.05, 0.05])

    assert moveit2.start_states == [[0.0, 0.0], [0.05, 0.05]]


def test_clearing_the_cache_forces_a_replan():
    # pose_commander clears per round, so a plan never outlives the live pose it
    # was solved from.
    trajectories = [
        trajectory((1.0, [0.1, 0.1], [0.0, 0.0])),
        trajectory((1.0, [0.2, 0.2], [0.0, 0.0])),
    ]
    backend, moveit2, _ = make_backend(trajectories)
    staging = [Segment(name='staging', motion=Motion.FREE, joints=list(STAGING_JOINTS))]

    backend.plan(staging, start_joints=[0.0, 0.0])
    backend.clear_plan_cache()
    planned = backend.plan(staging, start_joints=[0.0, 0.0])

    assert moveit2.start_states == [[0.0, 0.0], [0.0, 0.0]]
    assert list(planned.trajectories[0].points[-1].positions) == [0.2, 0.2]


def test_a_failed_joint_leg_is_not_cached():
    # CHOMP is seeded from a straight line and can fail on a problem it would
    # solve on a retry, so a failure must not poison the rest of the round.
    backend, moveit2, _ = make_backend([None, trajectory((1.0, [0.1, 0.1], [0.0, 0.0]))])
    staging = [Segment(name='staging', motion=Motion.FREE, joints=list(STAGING_JOINTS))]

    assert backend.plan(staging, start_joints=[0.0, 0.0]) is None
    assert backend.plan(staging, start_joints=[0.0, 0.0]) is not None
    assert moveit2.start_states == [[0.0, 0.0], [0.0, 0.0]]


def test_plans_are_timed_below_the_hard_limits():
    # MoveIt times plans to whatever scaling the request asks for, and its TOTG
    # resampling then reports accelerations slightly past that. Riding the limit
    # exactly (pymoveit2's 0.0 default, which MoveIt reads as full speed) is what
    # made a plain home move come back at 0.502 against a 0.500 limit and get
    # rejected by trajectory_bridge, so the request must leave headroom.
    _, moveit2, _ = make_backend([])
    assert 0.0 < moveit2.max_velocity < 1.0
    assert 0.0 < moveit2.max_acceleration < 1.0


def test_over_limit_trajectory_is_slowed_until_it_fits():
    over = trajectory(
        (0.0, [0.0, 0.0], [0.0, 0.0]),
        (1.0, [0.5, 0.2], [1.5, 0.4]),      # joint1 at 3x its 0.5 limit
    )
    backend, _, _ = make_backend([])
    fitted = backend._fit_to_joint_limits(over)

    # 3x for the limit itself, plus _LIMIT_MARGIN to land just inside it.
    stretch = 3.0 * _LIMIT_MARGIN
    assert seconds(fitted.points[-1]) == pytest.approx(stretch)
    for point in fitted.points:
        for v in point.velocities:
            assert abs(v) <= LIMIT
    # Time stretched by k scales acceleration by 1/k^2.
    assert fitted.points[-1].accelerations[0] == pytest.approx(0.25 / stretch ** 2)
    # Positions are the path itself and must not move.
    assert list(fitted.points[-1].positions) == [0.5, 0.2]


def test_within_limit_trajectory_is_untouched():
    ok = trajectory((0.0, [0.0, 0.0], [0.0, 0.0]), (2.0, [0.4, 0.4], [0.2, 0.2]))
    backend, _, _ = make_backend([])
    fitted = backend._fit_to_joint_limits(ok)

    assert seconds(fitted.points[-1]) == pytest.approx(2.0)
    assert list(fitted.points[-1].velocities) == [0.2, 0.2]


def test_fitting_is_a_noop_without_limits_loaded():
    # No joint_limits_yaml means no checking -- better than scaling by a made-up
    # limit, and trajectory_bridge still rejects anything genuinely unsafe.
    over = trajectory((0.0, [0.0, 0.0], [0.0, 0.0]), (1.0, [0.5, 0.2], [1.5, 0.4]))
    backend, _, _ = make_backend([], limits=False)
    assert seconds(backend._fit_to_joint_limits(over).points[-1]) == pytest.approx(1.0)


def test_fractional_times_survive_the_stretch():
    # The sec/nanosec split is easy to get wrong; 0.75s at 2x must be 1.5s,
    # and 2x here is the 1.0/0.5 velocity overshoot times _LIMIT_MARGIN.
    over = trajectory((0.75, [0.0, 0.0], [1.0, 0.0]))
    backend, _, _ = make_backend([])
    fitted = backend._fit_to_joint_limits(over)
    expected = 0.75 * 2.0 * _LIMIT_MARGIN
    assert seconds(fitted.points[-1]) == pytest.approx(expected)
    assert fitted.points[-1].time_from_start.sec == 1
    assert fitted.points[-1].time_from_start.nanosec == pytest.approx(
        (expected - 1.0) * 1e9, abs=1e3
    )


def test_empty_velocities_do_not_crash_the_fitter():
    bare = JointTrajectory(joint_names=list(JOINTS))
    point = JointTrajectoryPoint()
    point.positions = [0.1, 0.1]
    point.time_from_start = Duration(sec=1)
    bare.points.append(point)
    backend, _, _ = make_backend([])
    assert seconds(backend._fit_to_joint_limits(bare).points[-1]) == pytest.approx(1.0)


def test_acceleration_over_limit_is_slowed_even_when_velocity_fits():
    # The real abort: a Cartesian 'target' leg whose velocities were all legal but
    # whose joint7 acceleration came back 0.502 against a 0.500 limit. Checking
    # velocity alone left scale at 1.0 and returned the trajectory untouched, and
    # trajectory_bridge then rejected the whole goal.
    over = trajectory((0.0, [0.0, 0.0], [0.0, 0.0]), (1.0, [0.2, 0.2], [0.1, 0.1]))
    over.points[-1].accelerations = [0.502, 0.1]
    backend, _, _ = make_backend([])

    fitted = backend._fit_to_joint_limits(over)

    # Time stretched by k divides acceleration by k^2, so k = sqrt(0.502/0.5).
    assert seconds(fitted.points[-1]) == pytest.approx(math.sqrt(0.502 / LIMIT) * _LIMIT_MARGIN)
    for point in fitted.points:
        for a in point.accelerations:
            assert abs(a) <= LIMIT
    assert list(fitted.points[-1].positions) == [0.2, 0.2]


def test_acceleration_within_limit_is_untouched():
    ok = trajectory((0.0, [0.0, 0.0], [0.0, 0.0]), (2.0, [0.4, 0.4], [0.2, 0.2]))
    ok.points[-1].accelerations = [0.5, 0.5]
    backend, _, _ = make_backend([])
    assert seconds(backend._fit_to_joint_limits(ok).points[-1]) == pytest.approx(2.0)


def test_fitting_is_idempotent():
    # _plan_segment fits every segment, including one handed back from
    # _plan_joint's cache, so a second pass must not stretch an already-fit
    # trajectory again.
    over = trajectory((0.0, [0.0, 0.0], [0.0, 0.0]), (1.0, [0.5, 0.2], [1.5, 0.4]))
    backend, _, _ = make_backend([])
    once = seconds(backend._fit_to_joint_limits(over).points[-1])
    twice = seconds(backend._fit_to_joint_limits(over).points[-1])
    assert once == pytest.approx(twice)
