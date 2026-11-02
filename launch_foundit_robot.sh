#!/bin/bash
# Bring up the right arm in a tmux session
set -euo pipefail

VIZ=true
SAFE=false
for arg in "$@"; do
    case "$arg" in
        --noviz) VIZ=false ;;
        --safe)  SAFE=true ;;
    esac
done

if [ "$SAFE" = true ]; then
    # Enter-gated: test_impedance plans on a target but waits for the
    # execute_grasp_impedance service, which execute_on_enter supplies.
    CONFIRM_ARG="-p require_confirmation:=true"
    GATE_PANE="ros2 run impedance_control execute_on_enter"
else
    # Hands-free: the arm moves as soon as a grasp or place target arrives,
    # with no human in the loop. The last pane is left as a spare shell --
    # the tmux layout below is pinned to five panes, so it stays.
    CONFIRM_ARG="-p require_confirmation:=false"
    GATE_PANE="echo 'hands-free -- re-run with --safe to require Enter before each move'"
fi

CONFIG_FILE="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG_FILE"' EXIT

# Unquoted heredoc: $CONFIRM_ARG and $GATE_PANE below are interpolated. The
# layout string contains no $ or backticks, so nothing else expands.
cat > "$CONFIG_FILE" << EOF
session_name: nero_right_arm
suppress_history: false
start_directory: "~"
windows:
  - window_name: ros
    layout: 0f85,98x58,0,0[98x16,0,0{49x16,0,0,9,48x16,50,0,16},98x24,0,17,13,98x16,0,42{49x16,0,42,14,48x16,50,42,18}]
    panes:
      - shell_command:
        - ros2 launch analytic_grasp_ros sensors.launch.py
      - shell_command:
        - echo "launch FOUND-IT here"
      - shell_command:
        - ros2 launch analytic_grasp_ros analytic.launch.py
      - shell_command:
        - ros2 run impedance_control impedance_control
      - shell_command:
        - $GATE_PANE &
        - ros2 run impedance_control test_impedance --ros-args $CONFIRM_ARG
EOF

if [ "$VIZ" = true ]; then
    cat >> "$CONFIG_FILE" << 'EOF'
  - window_name: rviz
    panes:
      - ros2 launch analytic_grasp_ros view.launch.py config:=view_grasp3.rviz
EOF
fi

# google-chrome --log-level=3 http://10.90.0.4:8080/ & disown

exec tmuxp load -y "$CONFIG_FILE"
