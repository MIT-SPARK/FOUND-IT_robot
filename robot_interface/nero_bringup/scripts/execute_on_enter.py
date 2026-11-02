#!/usr/bin/env python3
"""Run in its own terminal alongside pose_commander. Press Enter to execute
the most recently previewed grasp plan; Ctrl+C to quit.

    ros2 run nero_bringup execute_on_enter.py
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def main():
    rclpy.init()
    node = Node("execute_on_enter", namespace="nero_right")
    client = node.create_client(Trigger, "execute_grasp")

    print("Press Enter to execute the current plan (Ctrl+C to quit)...")
    try:
        while rclpy.ok():
            input()
            if not client.wait_for_service(timeout_sec=2.0):
                print("execute_grasp service unavailable -- is pose_commander running?")
                continue
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(node, future)
            print(future.result().message)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
