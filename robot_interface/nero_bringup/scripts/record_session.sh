#!/usr/bin/env bash
# Records one offline-dev session: the static TF tree (camera extrinsics,
# calibration_publisher's hand-eye output) plus the arm's live joint state
# and the object point cloud grasp_selection is consuming. /tf itself is
# deliberately not recorded -- the offline robot_state_publisher regenerates it
# from the frozen joint state session_publisher.py republishes, so two
# publishers never fight over the same frames. See scripts/launch_offline.sh in
# the workspace root for playback.
#
# Usage: scripts/record_session.sh [session_name] [namespace] [pcd_topic]
#
# Run this during a normal bring-up (agx_arm_ctrl + agx_arm_description +
# analytic_grasp_ros already up), through a real grasp attempt, then Ctrl-C.
# Confirm topic names with `ros2 topic list` first if you've changed the
# defaults anywhere -- pcd_topic in particular has a documented mismatch
# between analytic_grasp_ros's live default ("object_pcd") and its
# canned-file demo path ("pointcloud").
set -euo pipefail

# Same two-step resolution scripts/launch_offline.sh uses, so the recorder and
# playback always agree on where sessions live. Not derived from this script's
# own path: it is installed into lib/ as well as living in scripts/, and bags
# are large gitignored session data that belong outside the package either way.
RECORDINGS_DIR="${NERO_RECORDINGS_DIR:-${HOME}/nero_recordings}"
NAME="${1:-$(date +%Y%m%d_%H%M%S)}"
NAMESPACE="${2:-nero_right}"
PCD_TOPIC="${3:-object_pcd}"

mkdir -p "${RECORDINGS_DIR}"
echo "Recording to ${RECORDINGS_DIR}/${NAME} -- Ctrl-C to stop."

# Captured under .raw and pruned into place below, so ${NAME} always names a
# session that is ready to replay. Ctrl-C is how a recording normally ends and
# it leaves ros2 bag record exiting 130, which `set -e` would otherwise treat as
# a failure and abort on before the prune ever runs -- hence the `|| true`.
ros2 bag record -o "${RECORDINGS_DIR}/${NAME}.raw" \
  /tf_static \
  "/${NAMESPACE}/puppet/joint_states" \
  "/${PCD_TOPIC}" \
  "/place_obj_location" \
  || true

if [ ! -d "${RECORDINGS_DIR}/${NAME}.raw" ]; then
  echo "Nothing was recorded." >&2
  exit 1
fi

if [ "${NERO_RECORD_PRUNE:-1}" = "1" ]; then
  # The noN/ frames are the live RViz candidate grippers' own static tree.
  # Offline runs its own copies of those publishers, so keeping the recorded
  # ones would mean two publishers on the same frames. session_publisher.py
  # filters them at playback too -- this is belt and braces, and it is why the
  # raw capture is kept rather than replaced: a session is unrepeatable, so a
  # wrong filter should never be able to destroy the only copy.
  ros2 run nero_bringup prune_session.py "${RECORDINGS_DIR}/${NAME}.raw" \
    --output "${RECORDINGS_DIR}/${NAME}"
  echo "Raw capture kept at ${RECORDINGS_DIR}/${NAME}.raw -- delete when happy."
else
  mv "${RECORDINGS_DIR}/${NAME}.raw" "${RECORDINGS_DIR}/${NAME}"
fi
