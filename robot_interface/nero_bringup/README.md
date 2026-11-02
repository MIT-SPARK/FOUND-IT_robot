# nero_bringup

Launch files that compose the NERO manipulation stack, and the entry point for
everything below. Start here; the other packages are libraries and config.

This extends the existing bring-up/visualization workflow (see
`analytic_grasp_ros/README.md`) rather than replacing it — it adds `move_group`,
a controller, and task sequencing on top of the `robot_state_publisher`/RViz that
`agx_arm_description` already provides.

## Package map

| Package | Holds |
|---|---|
| `nero_description` | Frozen URDF + meshes, snapshotted from `agx_arm_description` so nothing here needs the `nero_aloha` workspace |
| `nero_moveit_config` | MoveIt config only: TRAC-IK, CHOMP, SRDF, RViz layouts, and a `move_group`-only launch file |
| `nero_motion_planning` | `pose_commander` — watches for grasp candidates, plans, previews, executes |
| `nero_joint_control` | `trajectory_bridge` — validated `FollowJointTrajectory` server onto `agx_arm_ctrl` |
| `impedance_control` | Torque-level impedance controller (an *alternative* to `nero_joint_control`) |
| `nero_bringup` | This package: the launch files that wire the above together |

`nero_joint_control` and `impedance_control` are **mutually exclusive**. Each owns
the arm's command path and publishes `puppet/joint_states`, so exactly one runs at
a time — which is why the choice is two launch files rather than a runtime flag.
Running both means two CAN owners and two publishers on one feedback topic.

`move_group.launch.py` in `nero_moveit_config` starts *only* `move_group`. It
deliberately starts no controller and no task logic: composing those is this
package's job, and it is also what keeps the dependency graph acyclic, since
`nero_motion_planning` depends on `nero_moveit_config` for its planner config.

## Launch (three terminals)

```bash
# 1. Real arm state publisher. mode=1 is publish-only (no commanding) -- start
#    here for a new environment/arm. mode=0 also enables the command channel
#    trajectory_bridge streams to; see Scope for what that requires.
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=0 auto_enable:=true

# 2. RViz + robot_state_publisher: MoveIt-enabled rviz config, with kinematics also
#    loaded locally onto rviz2 (needed for RobotInteraction's drag markers)
ros2 launch agx_arm_description display.launch.py namespace:=nero_right \
  rvizconfig:=$(ros2 pkg prefix nero_moveit_config)/share/nero_moveit_config/rviz/moveit_display.rviz \
  kinematics_config:=$(ros2 pkg prefix nero_moveit_config)/share/nero_moveit_config/config/kinematics.yaml

# 3. move_group: TRAC-IK + CHOMP planning against the live arm state (also brings
#    up trajectory_bridge and pose_commander, see below)
ros2 launch nero_bringup pick_place.launch.py namespace:=nero_right
```

Use `namespace:=nero_left` on commands 2 and 3 for the left arm instead. Command 1
always brings up both arms; commands 2 and 3 are per-side.

## Headless (no RViz)

Step 2 above only exists to give you `robot_state_publisher` (which `move_group`
needs, for `/tf`) plus the RViz GUI. If you're driving MoveIt from a script
instead (`sample_pose_commander.py`, pymoveit2, your own `MoveGroupInterface`
client) and don't want the RViz window at all, skip step 2 and pass
`start_robot_state_publisher:=true` to step 3 instead — it starts its own
`robot_state_publisher` internally, so `pick_place.launch.py` becomes fully
self-contained:

```bash
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=0 auto_enable:=true
ros2 launch nero_bringup pick_place.launch.py namespace:=nero_right \
  start_robot_state_publisher:=true
```

Leave `start_robot_state_publisher` at its default (`false`) whenever
`display.launch.py` is also running — it starts its own `robot_state_publisher`,
and two of them on the same namespace would publish the same TF frames twice.

## Using RViz

1. In the **MotionPlanning** panel's Planning tab, an orange interactive marker
   should appear at the gripper. If it's not visible, check `Displays →
   MotionPlanning → Planning Request → Query Goal State` is checked.
2. Drag the marker to the pose you want the gripper to reach.
3. Click **Plan** to compute a TRAC-IK/CHOMP trajectory from the arm's current live pose
   to that goal. On success, RViz automatically animates the planned path in the
   same 3D view (the `Planned Path` sub-display, purple ghost robot). Check
   `Loop Animation` under that same section to replay it, or just click Plan again.
4. Click **Execute** — see Scope below for what has to be true first for this to
   actually move the real arm.

## Scope: joint-space execution via trajectory_bridge

`pick_place.launch.py` also launches `trajectory_bridge` (from `nero_joint_control`), which implements the
`FollowJointTrajectory` action `moveit_simple_controller_manager` calls when you
click **Execute**. `agx_arm_ctrl` has no controller_manager and no validation on
its own command path (`control/joint_states` → vendor SDK `move_j()`/`move_js()`
directly, straight to CAN) — `trajectory_bridge` is what actually makes this safe:
it rejects (never clamps) any goal with out-of-range positions/velocities/
accelerations, or whose first point doesn't match the arm's current live position
within `chomp_planning.yaml`'s `start_state_max_bounds_error` tolerance.

`trajectory_bridge` has one gate of its own, `require_armed`, which defaults to
**`false`** now that live execution has been validated on this arm — a bare
`pick_place.launch.py` launch with `agx_arm_ctrl` at `mode:=0` drives the real
arm on Execute, no extra steps. Bring it back (e.g. for a new environment, a
different arm, or a trajectory shape you have not watched execute yet) with
`require_armed:=true`: the bridge then starts disarmed and rejects every goal
until you deliberately arm it:
`ros2 service call /nero_right/arm_trajectory_bridge std_srvs/srv/SetBool "{data: true}"`
(disarm the same way with `data: false`).

There is **no dry-run mode**. `trajectory_bridge` always publishes to
`control/joint_states`; whether anything acts on that depends entirely on
`agx_arm_ctrl` being up and in `mode:=0`. The task-level gate is
`execution.require_confirmation` in `nero_motion_planning`'s
`config/manipulation.yaml`, which decides whether a planned sequence waits for
`execute_on_enter.py` or runs the moment it is planned.

Separately, `agx_arm_ctrl`'s own `mode` param gates whether it even has a command
subscriber at all — `mode:=1` (publish-only) means the arm is physically
uncommandable regardless of `trajectory_bridge`'s state, independent of the two
gates above. Keep a hand near the physical e-stop / CAN power whenever `mode:=0`
is active: **the software `emergency_stop` service on
`agx_arm_ctrl` currently only holds the arm's current position, it is not a true
cutoff** (its `electronic_emergency_stop()` call is disabled upstream with a
`# TODO: broken` note) — treat physical power as the real e-stop.

This first pass covers joint-space execution only (matching the `arm` planning
group and the CHOMP pipeline already in place); Cartesian execution
(`move_l`/`move_p`) is a separate follow-up.

## Offline development (no hardware)

Record a session during a normal bring-up, then bring the same stack up against
it later to tune `analytic_grasp_ros`'s grasp proposer and this package's motion
planning against a fixed scene, in RViz, with no arm/camera/CAN connected.

**Recording** (while `agx_arm_ctrl`, `agx_arm_description`, and
`analytic_grasp_ros` are already up, per the normal bring-up above):

```bash
scripts/record_session.sh test1   # Ctrl-C after a grasp attempt
```

An alias makes this a one-liner from any terminal during bring-up, e.g. in
`~/.bashrc`:

```bash
alias log_grasp='~/lorenzo/grasp_ws/src/robot_interface/nero_bringup/scripts/record_session.sh'
```

This captures `/tf_static` (camera extrinsics + `calibration_publisher`'s
hand-eye output), the arm's `puppet/joint_states`, and the `object_pcd` point
cloud into `~/nero_recordings/test1/` -- outside the package, since bags are
hundreds of MB of session data, not source. Override the location with
`$NERO_RECORDINGS_DIR`; `scripts/launch_offline.sh` resolves the same variable,
so the recorder and playback always agree.
`/tf` itself is deliberately not recorded; offline regenerates it from the
frozen joint state instead, via the frozen robot description below, so nothing
double-publishes the same frames.

The raw capture lands in `test1.raw/` and is then rewritten into `test1/` by
`scripts/prune_session.py`, which drops the `noN/gripper_*` frames from
`/tf_static`. Those belong to the live RViz candidate-gripper publishers, and
offline runs its own copies of those nodes -- keeping the recorded ones would
mean two publishers on the same frames. The raw capture is kept rather than
replaced, because a session is unrepeatable and a wrong filter should never be
able to destroy the only copy; delete it once you are happy. Set
`NERO_RECORD_PRUNE=0` to skip the rewrite entirely (playback filters the same
frames anyway, so an unpruned bag still works).

**Robot description**: `agx_arm_description`'s URDF/meshes are resolved live
via `xacro` from the separate `nero_aloha` workspace on every normal launch
(there's no SDF anywhere in this stack, despite `agx_arm`'s bring-up docs
sometimes calling it that). For offline dev to not need that workspace
present at all, `nero_description` holds a frozen, self-contained snapshot,
produced by (re-run whenever the upstream URDF changes):

```bash
# with nero_aloha's install/setup.bash sourced
python3 ../nero_description/scripts/freeze_description.py
python3 ../nero_description/scripts/freeze_description.py --model nero/urdf/gripper_only.urdf
```

The second model is the gripper on its own, which `offline_viz.launch.py` draws
once per grasp candidate.

**Playback**:

```bash
./launch_offline.sh test1          # from ~, or grasp_ws/scripts/
./launch_offline.sh test1 --noviz  # without the RViz window
```

A tmuxp session mirroring `launch_right_arm.sh` pane for pane, minus everything
that needs hardware. One `ros` window with four panes -- `session_publisher.py`,
`grasp_selection`, `pick_place.launch.py`, `execute_on_enter.py` -- and an `rviz`
window running `offline_viz.launch.py` on the same `view_grasp2.rviz` the real
bring-up opens. A pane each is the point: Ctrl-C and up-arrow re-runs the one
node you are tuning without disturbing the rest of the scene.

**The scene is frozen, not replayed.** `session_publisher.py` reads the bag once
and publishes an unchanging snapshot on the wall clock: `/tf_static` latched as a
single accumulated message, the object cloud restamped and republished slowly,
and the first recorded joint state republished at 50 Hz. Nothing in a recording
is actually dynamic -- the object does not move, and the arm cannot, since
offline nothing subscribes to `control/joint_states`.

Do not reach for `ros2 bag play --loop` instead. It rewinds the clock every time
the bag restarts, which makes every tf2 buffer in the system dump itself with
`TF_OLD_DATA`, resets MoveIt's `CurrentStateMonitor`, and loses whatever was
being planned. There is no `use_sim_time` anywhere in this stack for the same
reason: with the snapshot on the wall clock, no `/clock` is needed or published,
and turning it on would simply stall every node waiting for a tick.

`grasp_selection` gets `process_once:=True` offline, which it does not have live.
Against a cloud that never changes, resampling would redraw the candidate set
every second and `pose_commander` would replan continuously; with it, candidates
are sampled once and the scene is deterministic. Restart the pane to resample.

Execute is safe to press as long as no real arm is up -- offline nothing is
subscribed to `control/joint_states`. Note that the arm therefore never moves:
every plan starts from the frozen recorded pose, whatever `home_arm` reports.

## Implementation notes

- **Mimic-joint fix**: `agx_arm_description`'s full gripper xacro
  (`nero_with_gripper_description.xacro`) originally had `gripper_joint1`/
  `gripper_joint2` both mimicking a joint named `"gripper"` that didn't exist
  anywhere in the model — silently ignored by plain URDF display, but rejected
  outright by `moveit_core`'s stricter RobotModel loader. Fixed upstream so
  `gripper_joint2` mimics `gripper_joint1` directly (a standard symmetric-gripper
  pattern, reproducing the same original range).
- **Known upstream URDF/hardware limit mismatch**: some joints on the real arm can
  rest slightly past the position limits declared in `nero_description.urdf` (e.g.
  `joint7`'s real position has been observed ~0.13 rad past its declared `±1.5708`
  rad limit; `joint2`'s ~0.02 rad past its declared `±1.74` rad limit). MoveIt
  tolerates small start-state violations via `chomp_planning.yaml`'s
  `start_state_max_bounds_error` (currently `0.15` rad), and `trajectory_bridge`
  applies that same tolerance to a goal's first point. If planning fails
  immediately with "start tree could not be initialized," check the `move_group`
  log for "outside bounds by a significant margin" and widen this value, or if
  `trajectory_bridge` rejects a goal's start point specifically, check
  `ros2 topic echo /nero_right/puppet/joint_states` for how far past the limit
  that joint is actually resting and consider whether the URDF's declared limits
  for that joint should be revisited upstream instead of just widening tolerance
  further.
- **CHOMP can plan to an out-of-bounds goal**: `default_planning_pipeline` is
  `chomp` (`nero_moveit_config/launch/move_group.launch.py`), which treats
  joint limits as a soft cost on its straight-line seed, not a hard constraint
  the way a bounds-checked sampling planner would — so it can report a
  successful plan whose goal sits past a joint's declared URDF limit (e.g. IK
  landing a hair past a joint's range on this redundant 7-DOF arm).
  `trajectory_bridge` is the only thing that catches this before hardware —
  there's no clamp underneath it: the vendor SDK's own joint-limit clamp
  (`pyAgxArm`'s `joint_limits_enabled`) defaults off and `agx_arm_ctrl` never
  turns it on. `goal_position_tolerance` (default `0.15` rad, passed to
  `pick_place.launch.py`) is the verified-on-hardware margin `trajectory_bridge`
  allows past each joint's declared limit for every point *after* the start
  (the start point keeps its own separate `start_state_tolerance` above).
  Raise it further only after re-verifying the new margin doesn't reach a real
  hard stop on the physical arm.
