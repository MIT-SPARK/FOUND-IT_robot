"""The MoveIt pick/place stack: move_group + joint-space controller + task logic.

Composes the three packages that used to all live inside nero_moveit_config:

  nero_moveit_config    move_group (TRAC-IK + CHOMP)
  nero_joint_control    trajectory_bridge -- validated FollowJointTrajectory
                        server streaming onto agx_arm_ctrl's control/joint_states
  nero_motion_planning  pose_commander -- watches for grasp candidates, plans the
                        waypoint sequence, previews it, executes on request

This is the MoveIt branch. Its counterpart, pick_place_impedance.launch.py,
swaps nero_joint_control for impedance_control -- the two are mutually
exclusive, since each is an owner of the arm's command path and of the
puppet/joint_states feedback topic. Never run both.

Assumes agx_arm_ctrl is already up (mode:=0 to allow commanding) and that
something is publishing /tf -- either agx_arm_description's display.launch.py,
or start_robot_state_publisher:=true here for a headless bring-up.

    ros2 launch nero_bringup pick_place.launch.py namespace:=nero_right
"""

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import yaml

from nero_moveit_config.launch_utils import (
    DEFAULT_FROZEN_URDF,
    DEFAULT_MODEL,
    resolve_robot_description,
)


def launch_setup(context, *args, **kwargs):
    namespace = LaunchConfiguration('namespace').perform(context)
    robot_description_source = LaunchConfiguration('robot_description_source').perform(context)
    custom_model = LaunchConfiguration('custom_model').perform(context)
    frozen_description_path = LaunchConfiguration('frozen_description_path').perform(context)

    moveit_config_path = get_package_share_path('nero_moveit_config')
    config_path = moveit_config_path / 'config'
    with open(config_path / 'chomp_planning.yaml') as f:
        chomp_yaml = yaml.safe_load(f)
    with open(config_path / 'moveit_controllers.yaml') as f:
        moveit_controllers_yaml = yaml.safe_load(f)

    # trajectory_bridge parses this with an XML parser to read position limits,
    # so it needs the resolved string, not a lazy substitution. Same resolution
    # move_group.launch.py performs for itself -- shared so the two cannot drift.
    robot_description = resolve_robot_description(
        context,
        source=robot_description_source,
        custom_model=custom_model,
        frozen_path=frozen_description_path,
    )

    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(moveit_config_path / 'launch' / 'move_group.launch.py')
        ),
        launch_arguments={
            'namespace': namespace,
            'custom_model': custom_model,
            'robot_description_source': robot_description_source,
            'frozen_description_path': frozen_description_path,
            'start_robot_state_publisher': LaunchConfiguration(
                'start_robot_state_publisher'
            ).perform(context),
        }.items(),
    )

    trajectory_bridge_node = Node(
        package='nero_joint_control',
        executable='trajectory_bridge.py',
        namespace=namespace,
        output='screen',
        parameters=[
            {
                'joint_names': moveit_controllers_yaml['moveit_simple_controller_manager'][
                    'arm_controller'
                ]['joints'],
                'control_rate': float(LaunchConfiguration('control_rate').perform(context)),
                'require_armed': LaunchConfiguration('require_armed')
                .perform(context)
                .lower()
                == 'true',
                # Same leniency CHOMP's own FixStartStateBounds adapter applies,
                # read from the pipeline config so the two cannot drift apart.
                'start_state_tolerance': chomp_yaml['start_state_max_bounds_error'],
                'goal_position_tolerance': float(
                    LaunchConfiguration('goal_position_tolerance').perform(context)
                ),
                'joint_limits_yaml': str(config_path / 'joint_limits.yaml'),
                'robot_description': robot_description,
            }
        ],
        # Same live-feedback topic move_group plans against; command_pub's target
        # topic (control/joint_states) needs no remap since it already matches
        # agx_arm_ctrl's namespaced subscription.
        remappings=[('joint_states', 'puppet/joint_states')],
    )

    pose_commander_node = Node(
        package='nero_motion_planning',
        executable='pose_commander.py',
        namespace=namespace,
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('manipulation_config').perform(context)}],
    )

    return [move_group, trajectory_bridge_node, pose_commander_node]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'namespace',
                default_value='nero_right',
                description=(
                    'Must match the namespace passed to display.launch.py and the '
                    'side (nero_right/nero_left) whose real feedback topic to plan '
                    'against.'
                ),
            ),
            DeclareLaunchArgument('custom_model', default_value=DEFAULT_MODEL),
            DeclareLaunchArgument(
                'robot_description_source',
                default_value='live',
                choices=['live', 'frozen'],
            ),
            DeclareLaunchArgument(
                'frozen_description_path', default_value=DEFAULT_FROZEN_URDF
            ),
            DeclareLaunchArgument(
                'start_robot_state_publisher',
                default_value='false',
                choices=['true', 'false'],
                description=(
                    'Set true for a headless bring-up with no display.launch.py. '
                    'Leave false when display.launch.py is running, or two '
                    'robot_state_publishers will publish the same TF frames.'
                ),
            ),
            DeclareLaunchArgument(
                'require_armed',
                default_value='false',
                description=(
                    'When true, trajectory_bridge starts disarmed and rejects all '
                    'goals until the arm_trajectory_bridge service is called with '
                    "data=true, independent of agx_arm_ctrl's own enable/mode "
                    'gating.'
                ),
            ),
            DeclareLaunchArgument(
                'control_rate',
                default_value='50.0',
                description='Rate (Hz) at which trajectory_bridge resamples and streams goals.',
            ),
            DeclareLaunchArgument(
                'goal_position_tolerance',
                default_value='0.15',
                description=(
                    'Slack (rad), applied uniformly to all joints, that '
                    'trajectory_bridge allows a goal position to sit past its '
                    'declared URDF limit before rejecting it. Exists because CHOMP '
                    'treats joint limits as a soft cost, not a hard constraint, so it '
                    'can report a successful plan whose goal is slightly out of '
                    'bounds. This is a physical-safety margin verified on hardware, '
                    'not floating-point noise absorption -- see the README before '
                    'raising it further.'
                ),
            ),
            DeclareLaunchArgument(
                'manipulation_config',
                default_value=str(
                    get_package_share_path('nero_motion_planning') / 'config' / 'manipulation.yaml'
                ),
                description=(
                    'Task description for pose_commander: named configurations, '
                    'approach/retreat distances, gripper widths, and the '
                    'execution.require_confirmation gate.'
                ),
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
