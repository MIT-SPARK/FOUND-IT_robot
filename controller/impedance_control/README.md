Run the URDF publisher/RVIZ:
```bash
ros2 launch agx_arm_description display.launch.py namespace:=nero_right
```

Run impedance control:
```bash
ros2 run impedance_control impedance_control
```

Run a test script (currently some issues with angle interpolation):
```bash
ros2 run impedance_control test_impedance
```

By default `test_impedance` plans on a grasp/place target but waits for a
confirmation before moving, which `execute_on_enter` supplies:
```bash
ros2 run impedance_control execute_on_enter   # press Enter to execute
```

To run hands-free instead — the arm moves as soon as a target arrives, with no
human in the loop:
```bash
ros2 run impedance_control test_impedance --ros-args -p require_confirmation:=false
```

The parameter is re-read on every target, so it can also be flipped on a
running node:
```bash
ros2 param set /my_node require_confirmation false
```

`grasp_ws/scripts/impedance_launch.sh` brings the whole stack up hands-free by
default; pass `--safe` to require Enter and start `execute_on_enter` in the
last tmux pane.
