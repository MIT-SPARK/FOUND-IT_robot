import os
from glob import glob

from setuptools import setup

package_name = 'nero_motion_planning'

setup(
    name=package_name,
    version='0.1.0',
    # The library half (segment/geometry/path_generator/backend) is a real python
    # package at the source root, so it installs to site-packages and imports
    # cleanly both from the installed scripts below and from pytest run here.
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('lib', package_name), [
            'scripts/pose_commander.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aryan Naveen',
    maintainer_email='aryannav@mit.edu',
    description=(
        'Task-space pick/place waypoint generation for the NERO arm, behind a '
        'swappable motion backend (MoveIt today, impedance control next).'
    ),
    license='Apache-2.0',
)
