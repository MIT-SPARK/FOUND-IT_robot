import os

from setuptools import setup

package_name = 'nero_joint_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('lib', package_name), ['scripts/trajectory_bridge.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aryan Naveen',
    maintainer_email='aryannav@mit.edu',
    description=(
        "Validated FollowJointTrajectory bridge from MoveIt onto agx_arm_ctrl's "
        "move_j command path. Mutually exclusive with impedance_control."
    ),
    license='Apache-2.0',
)
