# Convert between ROS and numpy
import numpy as np
from typing import Union

from impedance_control.utils.TransformHelpers import *

### Transforms
from geometry_msgs.msg import Transform
def Tmsg_to_T(msg: Transform):
    p = msg.translation
    p = np.array([p.x, p.y, p.z]).reshape([3,1])
    q = msg.rotation
    quat = np.array([float(q.w), float(q.x), float(q.y), float(q.z)])
    R = R_from_quat(quat)
    return T_from_Rp(R, p)

def T_to_Tmsg(T) -> Transform:
    transform = Transform()
    p = p_from_T(T)
    transform.translation.x = float(p[0])
    transform.translation.y = float(p[1])
    transform.translation.z = float(p[2])
    q = quat_from_R(R_from_T(T))
    transform.rotation.w = float(q[0])
    transform.rotation.x = float(q[1])
    transform.rotation.y = float(q[2])
    transform.rotation.z = float(q[3])
    return transform

from geometry_msgs.msg import Pose
def Posemsg_to_T(msg: Pose):
    p = msg.position
    p = np.array([p.x, p.y, p.z]).reshape([3,1])
    q = msg.orientation
    quat = np.array([float(q.w), float(q.x), float(q.y), float(q.z)])
    R = R_from_quat(quat)
    return T_from_Rp(R, p)

def T_to_Posemsg(T) -> Pose:
    pose = Pose()
    p = p_from_T(T)
    pose.position.x = float(p[0])
    pose.position.y = float(p[1])
    pose.position.z = float(p[2])
    q = quat_from_R(R_from_T(T))
    pose.orientation.w = float(q[0])
    pose.orientation.x = float(q[1])
    pose.orientation.y = float(q[2])
    pose.orientation.z = float(q[3])
    return pose

from geometry_msgs.msg import Point
def Pointmsg_to_p(p: Point):
    return np.array([p.x, p.y, p.z]).reshape([3,1])


### Time
import builtin_interfaces.msg
# from builtin_interfaces.msg import Time
def float_to_rostime(timestamp: float) -> builtin_interfaces.msg.Time:
    ros_time = builtin_interfaces.msg.Time()
    ros_time.sec = int(timestamp)
    ros_time.nanosec = int((timestamp - ros_time.sec) * 1e9)
    return ros_time

def float_to_rosduration(timestamp: float) -> builtin_interfaces.msg.Duration:
    ros_time = builtin_interfaces.msg.Duration()
    ros_time.sec = int(timestamp)
    ros_time.nanosec = int((timestamp - ros_time.sec) * 1e9)
    return ros_time