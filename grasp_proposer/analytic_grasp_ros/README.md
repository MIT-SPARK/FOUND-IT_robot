# Analytic Grasping with Nero Arms

## Standard Launch Procedure
First, start the arm controller and camera:
```bash
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=0 auto_enable:=true &
ros2 launch orbbec_camera multi_camera.launch.py \
  front_serial:="CC1WC5200NX" \
  left_serial:="CC1WC5201L8" \
  right_serial:="CC1WC520126"
```
(make sure to type `fg` when closing to kill both of these launches!)

Optionally, launch rviz to view what is going on:
```bash
ros2 launch analytic_grasp_ros view.launch.py config:=view_pcd.rviz
# ros2 launch analytic_grasp_ros view.launch.py config:=view_r.rviz
```

Then, launch all the grasping infrastructure:
TODO: currently this does not work without rviz! (due to some frame weirdness with gripper_palm...)
```bash
ros2 launch analytic_grasp_ros analytic.launch.py
```

Lastly, to execute a given grasp:
```bash
ros2 param set grasp_selection execute_number 0
```

## Robot calibration
First, start the arm controller and camera:
```bash
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=1 auto_enable:=true &
ros2 launch orbbec_camera multi_camera.launch.py \
  left_serial:="CC1WC5201L8" \
  right_serial:="CC1WC520126" &
ros2 launch realsense2_camera rs_launch.py \
  serial_no:="'115422250397'"
```

TODO: launch the realsense.

Plug the leader arm in, so that you have full control over the arm position. Then, move the calibration board to a place where it can be seen by the front camera.

Now we are ready to calibrate:
```bash
cd ~/lorenzo/grasp_ws/src/grasp_proposer/analytic_grasp_ros/analytic_grasp_ros
ipython calibrate.py
```

To calibrate, use `ros2 param set /hand_eye_calibration mode "cal_f"` (or cal_a). To view in rviz:
```bash
ros2 launch analytic_grasp_ros view.launch.py config:=view_cal.rviz
```



## TODO
- fix the weird publishing to "gripper" when the robot is in teach mode (Aryan may have done this for us!)
- LOAD THE NEW CALIBRATION!!



## Older Stuff
### Visualize the arm in rviz
First, launch the arm state publisher (mode=1 to only publish)
```
ros2 launch agx_arm_ctrl start_nero_aloha.launch.py mode:=1 auto_enable:=true
```

Second, launch rviz and map it to the urdf:
```
ros2 launch agx_arm_description display.launch.py namespace:=nero_right
```

The default settings connect to the right gripper; change this to `nero_left` for the left gripper. See the launch file for more config options.

### Nodes
Point cloud publisher:
```
ros2 run analytic_grasp_ros pcd_publisher --ros-args -p pcd_file:=/home/agilex/lorenzo/analytic/pickmeup.pcd
```
Publishes a saved point cloud in the camera frame!

Static tf publisher:
```
ros2 run analytic_grasp_ros calibration_publisher
```
Publishes static transforms from calibration.

Grasp selector:
```
ros2 run analytic_grasp_ros grasp_selection
```
Reads a point cloud message and computes analytic grasps. Publishes them to the tf tree.

### Gripper visualization
To add in gripper visualizations, first run a robot state publisher for the gripper:
```
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args \
  -p robot_description:="$(cat third_party/agx_arm_description/agx_arm_urdf/nero/urdf/gripper_only.urdf)" \
  -p frame_prefix:=no0/ \
  -r robot_description:=robot2
```
And then a static tf:
```
ros2 run tf2_ros static_transform_publisher \
    0 0 0 0 0 0 \
    grasp_candidate_00 no0/gripper_flange
```






TODO:
1. Need a static tf for the gripper grab parts (done)
2. Need some tf logic for the transform: shouldn't be gripper_flange! (done)
3. Make a few launch files! (done)
4. Fix calibration problems

This was very helpful: https://github.com/MetroRobots/rosetta_launch


