# FOUND-IT++ Supplemental Robot Code

This is the release of supporting code to run [FOUND-IT++](https://arxiv.org/abs/2610.23622) on a robot. It is designed for the [AgileX Cobot Magic](https://global.agilex.ai/products/cobot-magic-bimanual-teleoperation-platform-mobile-aloha) with one [Nero](https://global.agilex.ai/products/nero) arm, but can be adapted to any robot platform.

This release includes:
- Two arm controllers (impedance control and joint control). Impedance control is used in the paper.
- An analytic grasp proposer, which also includes a ROS calibration script for hand-eye calibration.
- A motion planning interface, designed for use with the joint controller (not used in the paper).
- Several supporting packages.

To run this code on your robot, place this repo inside the `src` directory of a ROS2 workspace and run `colcon build`. You will also need to install [FOUND-IT](https://github.com/MIT-SPARK/FOUND-IT); use the `run_ros2.sh` script and run FOUND-IT in a ROS workspace with the same ROS_DOMAIN_ID. See [launch_foundit_robot.sh](launch_foundit_robot.sh) for a quick script.

This code is tested on ROS2 Humble and Ubuntu 22.04.

See [the paper](https://arxiv.org/abs/2610.23622) for more!
