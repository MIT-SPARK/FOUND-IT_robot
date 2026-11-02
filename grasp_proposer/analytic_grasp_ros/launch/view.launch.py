from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution, Command
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    gripper_description = ParameterValue(Command(['xacro ', "third_party/agx_arm_description/agx_arm_urdf/nero/urdf/gripper_only.urdf"]), value_type=str)
    gripper_nodes = [
        Node(
            name=['gripper_state_publisher', str(i)],
            package='robot_state_publisher', 
            executable='robot_state_publisher',
            parameters=[{'robot_description': gripper_description,
                         'frame_prefix': f"no{i}/",
                        }],
            remappings=[('robot_description', f"gripper{i}")]
        )
        for i in range(5)
        # TODO: make this a launch parameter (the 5)
    ]

    gsp_nodes = [
        Node(
            name=['gripper_joint_publisher', str(i)],
            package='joint_state_publisher', 
            executable='joint_state_publisher',
            parameters=[{'zeros': {
                            'gripper_joint1': 0.05,
                            }
                        }],
            remappings=[('robot_description', f"gripper{i}")]
        )
        for i in range(5)
    ]
    rviz_config = LaunchConfiguration('config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value='view_pcd.rviz',
            description='rviz config name'
        ),
        ####
        # Launch file for rviz with robot arm
        ####
        IncludeLaunchDescription(
            PathJoinSubstitution([FindPackageShare('agx_arm_description'), 'launch', 'display.launch.py']),
            launch_arguments={'namespace': 'nero_right',
                              'arm_type': 'nero',
                              'rvizconfig': PathJoinSubstitution(["grasp_proposer/analytic_grasp_ros/rviz/",rviz_config]),
                              'follow': 'true',
                              }.items()
        ),
        ####
        # Gripper publisher
        ####
        *gripper_nodes,
        *gsp_nodes
    ])