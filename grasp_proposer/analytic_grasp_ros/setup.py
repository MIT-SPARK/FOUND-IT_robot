from setuptools import find_packages, setup
import glob
import os

package_name = 'analytic_grasp_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*')),
        # offline_viz.launch.py resolves these through FindPackageShare, which
        # otherwise serves whatever stale copy an earlier build left behind.
        (os.path.join('share', package_name, 'rviz'), glob.glob('rviz/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lorenzo',
    maintainer_email='lorenzos@mit.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pcd_publisher = analytic_grasp_ros.pcd_publisher:main',
            'grasp_selection = analytic_grasp_ros.grasp_selection:main',
            'calibration_publisher = analytic_grasp_ros.calibration_publisher:main',
            'calibrate = analytic_grasp_ros.calibrate:main',
            'grasp_step = analytic_grasp_ros.grasp_step:main'
        ],
    },
)
