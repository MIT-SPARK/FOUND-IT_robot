#!/usr/bin/env python3
"""Rewrite a recorded session with the grasp proposer's own frames dropped.

The recorder captures /tf_static wholesale, which sweeps up the noN/gripper_*
tree published by the live RViz candidate-gripper nodes. Offline runs its own
copies of those publishers, so a replayed recording double-publishes frames that
were never really part of the scene. session_publisher.py filters them at
playback too, so this is belt and braces -- its value is that recordings stop
carrying frames that are a lie about what was observed. It keeps the topic name,
so nothing downstream needs to know whether a bag was pruned, and it is safe to
run on an existing session.

    ros2 run nero_bringup prune_session.py ~/nero_recordings/apple.raw \
      --output ~/nero_recordings/apple
"""

import argparse
import os
import re
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from tf2_msgs.msg import TFMessage

# Kept in step with session_publisher.py's exclude_child_frames default. The two
# are duplicated rather than shared because nero_bringup installs loose scripts
# and has no importable module to hang a single constant off.
DEFAULT_EXCLUDE = r'^no\d+/'


def prune(bag_path, output, tf_static_topic, exclude):
    """Copy bag_path to output, dropping excluded children from tf_static_topic."""
    metadata = rosbag2_py.Info().read_metadata(bag_path, '')

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path),
        rosbag2_py.ConverterOptions('', ''),
    )
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(
            uri=output, storage_id=metadata.storage_identifier
        ),
        rosbag2_py.ConverterOptions('', ''),
    )
    # Copied whole, so offered_qos_profiles comes across with them. Rebuilding
    # these by hand would silently drop transient-local durability from the
    # cloud, and a replay of the pruned bag would then never reach
    # grasp_selection's transient-local subscription.
    for topic in reader.get_all_topics_and_types():
        writer.create_topic(topic)

    kept_total = dropped_total = messages = 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != tf_static_topic:
            # Raw bytes, never deserialized: the rewrite stays I/O bound rather
            # than CPU bound, which is what keeps it cheap once a recording
            # grows camera topics.
            writer.write(topic, data, stamp)
            continue

        messages += 1
        transforms = deserialize_message(data, TFMessage).transforms
        kept = [t for t in transforms if not exclude.search(t.child_frame_id)]
        dropped_total += len(transforms) - len(kept)
        kept_total += len(kept)
        if not kept:
            # An empty TFMessage says nothing; the five all-noN messages in a
            # session recorded with the viz up would otherwise become noise.
            continue
        writer.write(topic, serialize_message(TFMessage(transforms=kept)), stamp)

    return kept_total, dropped_total, messages


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('bag_path', help='recorded session to read')
    parser.add_argument(
        '--output', required=True, help='session directory to write; must not exist'
    )
    parser.add_argument('--tf-static-topic', default='/tf_static')
    parser.add_argument(
        '--exclude-child-frames',
        default=DEFAULT_EXCLUDE,
        help=f'regex matched against child_frame_id (default {DEFAULT_EXCLUDE})',
    )
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.bag_path)
    output = os.path.expanduser(args.output)
    if os.path.exists(output):
        print(f'error: {output} already exists', file=sys.stderr)
        return 1

    kept, dropped, messages = prune(
        bag_path, output, args.tf_static_topic, re.compile(args.exclude_child_frames)
    )
    print(
        f'kept {kept} and dropped {dropped} transforms across {messages} '
        f'{args.tf_static_topic} messages -> {output}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
