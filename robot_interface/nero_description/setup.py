import os
from glob import glob

from setuptools import setup

package_name = 'nero_description'


def _data_files_for_tree(src_root, dest_root):
    """(dest_dir, [files]) tuples for every file under src_root, preserving its
    subdirectory structure under dest_root -- data_files itself has no
    recursive-directory form, and meshes/ is arbitrarily nested (it mirrors the
    package:// layout of whatever upstream packages the meshes came from; see
    scripts/freeze_description.py)."""
    entries = []
    for dirpath, _dirnames, filenames in os.walk(src_root):
        if not filenames:
            continue
        rel_dir = os.path.relpath(dirpath, src_root)
        dest_dir = os.path.join(dest_root, rel_dir) if rel_dir != '.' else dest_root
        entries.append((dest_dir, [os.path.join(dirpath, f) for f in filenames]))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('lib', package_name), ['scripts/freeze_description.py']),
        # The URDF's package:// URIs resolve against this share tree, so the
        # nesting under meshes/ must be preserved exactly.
        *_data_files_for_tree('meshes', os.path.join('share', package_name, 'meshes')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aryan Naveen',
    maintainer_email='aryannav@mit.edu',
    description=(
        "Frozen snapshot of the NERO arm's resolved URDF and meshes, so the "
        "planning stack can describe the robot with no agx_arm_description present."
    ),
    license='Apache-2.0',
)
