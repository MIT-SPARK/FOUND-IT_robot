#!/usr/bin/env python3
# Publish calibration static transforms

import os
import sys
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
# from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized ROS-order [x, y, z, w]."""
    trace = np.trace(R)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)

def matrix_to_transform_msg(matrix: np.ndarray, parent: str, child: str, stamp) -> TransformStamped:
    """Inverse of transform_msg_to_matrix."""
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    qx, qy, qz, qw = rotation_matrix_to_quaternion(matrix[:3, :3])
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw
    return msg


def transform_msg_to_matrix(msg: TransformStamped) -> np.ndarray:
    """Convert a ROS TransformStamped into a 4x4 homogeneous transform."""
    t = msg.transform.translation
    q = msg.transform.rotation

    # ROS quaternion order is x, y, z, w.
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Received TF transform with a zero-length quaternion.")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    R = np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )

    p = np.array([t.x, t.y, t.z], dtype=np.float64)
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p
    return X


class CalibrationPublisher(Node):
    def __init__(self):
        super().__init__('calibration_publisher')

        self.declare_parameter("transforms", "/home/agilex/lorenzo/calibration/save3")


        self.yaml_file = self.get_parameter("transforms").value
        print(self.yaml_file)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.broadcaster = StaticTransformBroadcaster(self)

        try:
            self.timer = self.create_timer(1.0, self.load_transforms)
            # transforms = self.load_transforms(yaml_file)
        except Exception as exc:
            self.get_logger().fatal(f'Failed to load transforms: {exc}')
            raise

    def optical_transform(self, link, optical, cache_name, cached):
        """link->optical from TF, cached to .npy so a dead camera is not fatal.

        On a cache hit the transform is also appended to `cached` for
        publishing, since with no driver running nothing else provides it.
        """
        path = os.path.join(self.yaml_file, cache_name)
        try:
            # lookup_transform(target, source, time) gives T_g0_g2:
            # pose of g2 expressed in g0
            msg = self.tf_buffer.lookup_transform(link, optical, rclpy.time.Time())
        except Exception as exc:
            if not os.path.exists(path):
                self.get_logger().info(
                    f'Waiting for {link} -> {optical} (no cache at {path}): {exc}')
                return None
            matrix = np.load(path)
            self.get_logger().warn(
                f'{link} -> {optical} unavailable; using cached {cache_name}. '
                'Republish it so downstream frames still resolve.')
            cached.append(matrix_to_transform_msg(
                matrix, link, optical, self.get_clock().now().to_msg()))
            return matrix

        matrix = transform_msg_to_matrix(msg)
        np.save(path, matrix)
        return matrix

    def load_transforms(self):
        # These come from the camera drivers, so an unplugged camera used to
        # block bring-up entirely. Cache them whenever they are available and
        # fall back to the cached copy when they are not.
        cached = []
        T_camf0_F = self.optical_transform(
            'camera_f_link', 'camera_f_color_optical_frame', 'T_camf0_F.npy', cached)
        T_camr0_A = self.optical_transform(
            'camera_r_link', 'camera_r_color_optical_frame', 'T_camr0_A.npy', cached)
        if T_camf0_F is None or T_camr0_A is None:
            return
        

        # Load calibration data
        # YAML (deprecated)
        # with open(self.yaml_file, 'r') as file:
        #     data = yaml.safe_load(file)
        # from folder (current)
        # tf_AcamBoard = np.load(f"{self.yaml_file}/T_AcamBoard.npy")
        tf_AcamGripper = np.load(f"{self.yaml_file}/T_AcamGripper.npy")
        tf_FcamBase = np.load(f"{self.yaml_file}/T_FcamBase.npy")
        # tf_FcamBoard = np.load(f"{self.yaml_file}/tf_FcamBoard.npy")
        data = {"tf_AG": tf_AcamGripper,
                "tf_FC": tf_FcamBase}

        result = list(cached)
        for key, entry in data.items():
            try:
                if key == "tf_AG":
                    parent = "nero_right/gripper_base"
                    child = "camera_r_link"
                    parent2 = "nero_left/gripper_base"
                    child2 = "camera_l_link"

                    # G = gripper flange
                    # tf_AG = np.array(entry)
                    tf_AG = entry
                    tf_G_camr0 = np.linalg.inv(T_camr0_A @ tf_AG)
                    matrix = tf_G_camr0

                    # LAZY
                    translation = matrix[:3, 3] 
                    rotation = matrix[:3, :3]
                    qx, qy, qz, qw = rotation_matrix_to_quaternion( rotation )

                    transform = TransformStamped()

                    transform.header.stamp = self.get_clock().now().to_msg()
                    transform.header.frame_id = parent2
                    transform.child_frame_id = child2

                    transform.transform.translation.x = float(translation[0])
                    transform.transform.translation.y = float(translation[1])
                    transform.transform.translation.z = float(translation[2])

                    transform.transform.rotation.x = qx
                    transform.transform.rotation.y = qy
                    transform.transform.rotation.z = qz
                    transform.transform.rotation.w = qw
                    result.append(transform)

                elif key == "tf_FC":
                    parent = "nero_right/world"
                    child = "camera_f_link"

                    # C = base_link
                    # tf_FC = np.array(entry)
                    # tf_CF = np.array([[-0.036,  0.542, -0.839,  0.005],
                    #                   [0.999,  0.031, -0.023, -0.306],
                    #                   [0.014, -0.840, -0.543,  0.503],
                    #                   [0.000,  0.000,  0.000,  1.000]])
                    # tf_FC = np.linalg.inv(tf_CF)
                    tf_FC = entry
                    tf_C_camf0 = np.linalg.inv(T_camf0_F @ tf_FC)
                    matrix = tf_C_camf0
                else:
                    continue
                
                translation = matrix[:3, 3] 
                rotation = matrix[:3, :3]
                qx, qy, qz, qw = rotation_matrix_to_quaternion( rotation )

                transform = TransformStamped()

                transform.header.stamp = self.get_clock().now().to_msg()
                transform.header.frame_id = parent
                transform.child_frame_id = child

                transform.transform.translation.x = float(translation[0])
                transform.transform.translation.y = float(translation[1])
                transform.transform.translation.z = float(translation[2])

                transform.transform.rotation.x = qx
                transform.transform.rotation.y = qy
                transform.transform.rotation.z = qz
                transform.transform.rotation.w = qw

                result.append(transform)

            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f'Invalid transform at index {key}: {exc}'
                ) from exc
            

            # publish
            self.broadcaster.sendTransform(result)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CalibrationPublisher()
        rclpy.spin(node)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())