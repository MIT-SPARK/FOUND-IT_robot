import os
from glob import glob

from setuptools import setup

package_name = 'nero_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('lib', package_name), glob('scripts/*.sh') + glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aryan Naveen',
    maintainer_email='aryannav@mit.edu',
    description=(
        'Launch files composing the NERO manipulation stack (move_group + a '
        'controller + task sequencing), and the offline-dev session recorder.'
    ),
    license='Apache-2.0',
)
