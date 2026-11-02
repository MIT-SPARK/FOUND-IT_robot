#!/usr/bin/env python3
# run with:
# ros2 run analytic_grasp_ros pcd_publisher --ros-args -p pcd_file:=/absolute/path/to/cloud.pcd

import struct

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class PcdPublisher(Node):
    def __init__(self):
        super().__init__("pcd_publisher")

        self.declare_parameter("pcd_file", "/home/agilex/lorenzo/analytic/pickmeup.pcd")
        self.declare_parameter("frame_id", "camera_f_color_optical_frame")
        self.declare_parameter("publish_period_sec", 10.0)

        pcd_file = self.get_parameter("pcd_file").value
        self.frame_id = self.get_parameter("frame_id").value
        publish_period_sec = self.get_parameter("publish_period_sec").value

        if not pcd_file:
            raise ValueError(
                "The 'pcd_file' parameter is required. "
                "Example: -p pcd_file:=/absolute/path/to/cloud.pcd"
            )

        self.publisher = self.create_publisher(
            PointCloud2,
            "pointcloud",
            10,
        )

        self.cloud_msg = self.load_pcd(pcd_file)

        self.timer = self.create_timer(
            publish_period_sec,
            self.publish_cloud,
        )

        self.publish_cloud()

    def load_pcd(self, pcd_file: str) -> PointCloud2:
        """Load a PCD file and convert it to sensor_msgs/PointCloud2."""
        cloud = o3d.io.read_point_cloud(pcd_file)

        if cloud.is_empty():
            raise RuntimeError(f"Could not load point cloud, or cloud is empty: {pcd_file}")

        points = np.asarray(cloud.points, dtype=np.float32)
        has_colors = cloud.has_colors()

        header = Header()
        header.frame_id = self.frame_id

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.is_dense = True

        if has_colors:
            # Open3D stores colors as floats in [0, 1].
            colors = np.asarray(cloud.colors)
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

            # Fields: x, y, z, packed RGB
            msg.fields = [
                PointField(name="x", offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="rgb", offset=12,
                           datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 16
            msg.row_step = msg.point_step * msg.width

            buffer = bytearray(msg.row_step)

            for i, ((x, y, z), (r, g, b)) in enumerate(zip(points, colors)):
                # RViz convention: RGB is packed into a float32 field.
                rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
                rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]

                struct.pack_into(
                    "<ffff",
                    buffer,
                    i * msg.point_step,
                    float(x),
                    float(y),
                    float(z),
                    rgb_float,
                )

            msg.data = bytes(buffer)

        else:
            # XYZ-only cloud.
            msg.fields = [
                PointField(name="x", offset=0,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4,
                           datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8,
                           datatype=PointField.FLOAT32, count=1),
            ]
            msg.point_step = 12
            msg.row_step = msg.point_step * msg.width
            msg.data = points.astype(np.float32).tobytes()

        self.get_logger().info(
            f"Loaded {msg.width} points from: {pcd_file} "
            f"(RGB: {'yes' if has_colors else 'no'})"
        )

        return msg

    def publish_cloud(self):
        self.cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.cloud_msg)


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = PcdPublisher()
        rclpy.spin(node)
    except Exception as exc:
        rclpy.logging.get_logger("pcd_publisher").error(str(exc))
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()