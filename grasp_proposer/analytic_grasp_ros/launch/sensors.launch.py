from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, Shutdown
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        ####
        # Start the realsense camera
        ####
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py']),
            launch_arguments={'serial_no': '"115422250397"',
                              }.items()
        ),
        ####
        # Start tracer base
        ####
        # IncludeLaunchDescription(
        #     PathJoinSubstitution([FindPackageShare('tracer_base'), 'launch', 'tracer_base.launch.py']),
        #     launch_arguments={}.items()
        # ),
        Node(
            package='tracer_base',
            executable='tracer_base_node',
            output='screen',
            emulate_tty=True,
            parameters=[{
                    'use_sim_time': False,
                    'port_name': 'can0',                
                    'odom_frame': 'odom',
                    'base_frame': 'base_link',
                    'odom_topic_name': 'odom',
                    'is_tracer_mini': False,
                    'simulated_robot': False,
                    'control_rate': 50,
            }],
            on_exit=Shutdown()
        ),
        ####
        # Point cloud publisher (point cloud from found-it)
        ####
        # ros2 run image_transport republish raw compressed --ros-args --remap in:=/camera_f/color/image_raw --remap out/compressed:=/camera_f/color/compressedd
        Node(
            package='image_transport',
            executable='republish',
            name='image_republisher',
            arguments=['raw', 'compressed'],
            remappings=[
                ('in', '/camera_f/color/image_raw'),
                ('out/compressed', '/camera_f/color/compressedd'),
            ],
            on_exit=Shutdown()
        ),
    ])