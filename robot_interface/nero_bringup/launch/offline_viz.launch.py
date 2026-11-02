"""Open RViz on the offline scene, showing what the real bring-up shows.

The offline twin of analytic_grasp_ros/launch/view.launch.py, which cannot be
reused here for two reasons: it xacros the gripper out of a hardcoded
/home/agilex/nero_aloha/... path, and it includes agx_arm_description's
display.launch.py -- which would start a second robot_state_publisher for the
arm on top of the one pick_place.launch.py already provides, and reintroduce
exactly the workspace dependency offline dev exists to avoid. Everything else
matches it node for node, so the two views are identical.

    ros2 launch nero_bringup offline_viz.launch.py

Started by scripts/launch_offline.sh's rviz window. Run it by hand to reopen
RViz after closing it, without restarting the rest of the session.
"""

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# view.launch.py shows the top five candidates as gripper meshes, one
# robot_state_publisher each. Matching that number keeps the two views
# identical; view_grasp2.rviz has a RobotModel display per gripper, so raising
# it means adding displays there too -- and grasp_selection would publish
# noN/gripper_flange frames whose grasp_candidate_NN parent does not exist for
# any candidate it did not actually rank.
CANDIDATE_GRIPPER_COUNT = 5


def frozen_gripper_description():
    """The gripper-only URDF the candidate meshes are drawn from."""
    # view.launch.py xacros this straight out of agx_arm_description, in the
    # separate nero_aloha workspace. Offline reads nero_description's frozen
    # snapshot of the same model instead -- the same reason robot_description
    # does. See nero_description/scripts/freeze_description.py, which produced
    # it with --model nero/urdf/gripper_only.urdf.
    return (
        get_package_share_path('nero_description') / 'urdf' / 'gripper_only.urdf'
    ).read_text()


def generate_launch_description():
    gripper_description = frozen_gripper_description()

    # One robot_state_publisher + joint_state_publisher per candidate, exactly as
    # view.launch.py runs them live: the frame_prefix places the gripper's links
    # under the noN/ frames grasp_selection broadcasts, and the description goes
    # out on gripperN, which view_grasp2.rviz's GraspN displays read. Unnamespaced
    # and on the global /joint_states, again matching the live nodes -- everything
    # in pick_place.launch.py is namespaced and remapped to puppet/joint_states,
    # so there is nothing to collide with.
    candidate_grippers = []
    for i in range(CANDIDATE_GRIPPER_COUNT):
        candidate_grippers.append(
            Node(
                name=f'gripper_state_publisher{i}',
                package='robot_state_publisher',
                executable='robot_state_publisher',
                parameters=[
                    {
                        'robot_description': gripper_description,
                        'frame_prefix': f'no{i}/',
                    }
                ],
                remappings=[('robot_description', f'gripper{i}')],
            )
        )
        candidate_grippers.append(
            Node(
                name=f'gripper_joint_publisher{i}',
                package='joint_state_publisher',
                executable='joint_state_publisher',
                # Jaws drawn part-open, so a candidate reads as a gripper rather
                # than a closed block.
                parameters=[{'zeros': {'gripper_joint1': 0.05}}],
                remappings=[('robot_description', f'gripper{i}')],
            )
        )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        namespace=LaunchConfiguration('namespace'),
        # The config's Fixed Frame and TF Prefix entries are already namespaced,
        # so display.launch.py's rewriting of those is a no-op on this file and
        # nothing is lost by launching rviz2 directly.
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('analytic_grasp_ros'),
                'rviz',
                LaunchConfiguration('config'),
            ]),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('namespace', default_value='nero_right'),
            DeclareLaunchArgument(
                'config',
                default_value='view_grasp2.rviz',
                description=(
                    'RViz config in analytic_grasp_ros/rviz. Same argument name '
                    'and default the real bring-up passes view.launch.py, so the '
                    'two rviz panes differ only in which package they name.'
                ),
            ),
            *candidate_grippers,
            rviz,
        ]
    )
