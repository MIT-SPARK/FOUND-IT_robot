from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, Shutdown
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    launch_grasp = LaunchConfiguration('launch_grasp')

    return LaunchDescription([
        DeclareLaunchArgument(
            'launch_grasp',
            default_value='True'
        ),
        ####
        # Publish all robot tfs in URDF (TODO: needed if you don't want to rely on rviz)
        ####
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('agx_arm_description'), 'launch', 'publish.launch.py']),
            launch_arguments={'namespace': 'nero_right',
                              'arm_type': 'nero',
                              'follow': 'true',
                            #   'kinematics_config':'false',
                              }.items()
        ),
        ####
        # Calibration publisher (static tfs from calibration)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='calibration_publisher',
            name='calibration_publisher',
            parameters=[{"transforms": "/home/agilex/lorenzo/calibration/save_Sept11"}], #save3 is good too
            on_exit=Shutdown()
        ),
        ####
        # Point cloud publisher (point cloud from found-it)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='pcd_publisher',
            name='pcd_publisher',
            parameters=[{'pcd_file': '/home/agilex/lorenzo/analytic/pickmeup.pcd'}],
            condition=IfCondition(launch_grasp)
        ),
        ####
        # Grasp selection (selects and publishes grasps)
        ####
        Node(
            package='analytic_grasp_ros',
            executable='grasp_selection',
            name='grasp_selection',
            parameters=[{"publish_gripper_tfs": True,
                         "top_k": 20}],
            condition=IfCondition(launch_grasp),
            on_exit=Shutdown()
        )
    ])