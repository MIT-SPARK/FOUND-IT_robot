#!/usr/bin/env python3
######################
# Impedance controller for the NERO arm.
# This completely replaces the nero arm control node.
# Publish:
# - "/joint_states": JointState
#       current joint state of robot (including gripper).
# Subscribe:
# - "/pose_cmd": PoseCommand
#       pose and twist command with duration for the robot.
# - "/joint_cmd": JointStates
#       joint with velocity command for the robot.
#       Gripper commands do not use impedance.
#       NOT CURRENTLY WORKING EXCEPT FOR GRIPPER
#
import numpy as np
import time
from threading import Lock
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from joey_interfaces.msg import PoseCommand
from tf2_ros import Buffer, TransformListener

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, resolve_firmware_profile
from agx_arm_ctrl.effector import AgxGripperWrapper

from .utils.TransformHelpers import *
from .utils.Convert import *
from .utils.KinematicChain import KinematicChain

# helper function to set up the NERO arm
def setup_agxarm(can_port):
    profile = resolve_firmware_profile(ArmModel.NERO, "1.21")
    config = create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=profile, comm="can", channel=can_port
    )
    arm = AgxArmFactory.create_arm(config)
    arm.connect()
    arm.set_follower_mode()
    for i in range(100):
        test = arm.enable()
        if test:
            break
        time.sleep(0.01)
    if not test:
        return None, None

    gripper = AgxGripperWrapper(arm)
    if not gripper.initialize():
        gripper = None
    return arm, gripper


class ControllerNode(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        # parameters
        base_frame = "world"
        tip_frame = "gripper_palm"
        self.tip_frame = f"nero_right/{tip_frame}"
        topic = "/nero_right/robot_description"
        can_port = "can_right"

        # set up kinematic chain
        self.chain = KinematicChain(base_frame, tip_frame, topic=topic)

        # set up arm control
        self.arm, self.gripper = setup_agxarm(can_port)
        if self.gripper is None:
            self.get_logger().error("Gripper failed!")
        self.get_logger().info("Arm connected!")
        # for i in range(self.chain.dofs):
        #     self.arm.set_joint_angle_vel_limits(i+1,
        #         max_joint_spd=2.0)
        self.gripper_default_effort = 1.0
        self.gripper.move(width=0.1, force=0.5)

        ############################
        # Publishers and subscribers
        # use separate threads for the controller (fast)
        # and the telemetry (slow).

        # threading
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.telemetry_group = MutuallyExclusiveCallbackGroup()
        self.lock = Lock()

        # variables
        self.q = None
        self.qdot = None
        self.efforts = None
        self.js_time = None
        self.T_goal = None
        self.v_goal = None
        self.Te_buf = None
        self.start_time = 0.
        self.movetime_goal = None
        self.q_goal = None
        self.qdot_goal = None
        self.gripper_goal_width = None
        self.gripper_goal_force = None
        self.timeout = 8.0

        # tf setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscribers
        self.sub_goalpose = self.create_subscription(
            PoseCommand, "/pose_cmd", self.cb_posecmd, 1,
            callback_group=self.telemetry_group
        )
        self.sub_goaljoint = self.create_subscription(
            JointState, "/joint_cmd", self.cb_jointcmd, 1,
            callback_group=self.telemetry_group
        )

        # publishers
        self.pub_jointstates = self.create_publisher(
            JointState, "/nero_right/puppet/joint_states", 1
        )
        
        # timers
        self.control_timer = self.create_timer(
            0.009, # this may be the fastest we can run
            self.cb_control,
            callback_group=self.control_group,
        )

        self.publish_timer = self.create_timer(
            0.01,
            self.cb_publish,
            callback_group=self.telemetry_group,
        )

    # TIMER CALLBACKS
    # high-rate control timer (impedance control)
    def cb_control(self):
        # 1) get joint states
        if not self.get_joint_states():
            return

        # 2) fkin at current joint angles
        p_cur, R_cur, Jv, Jw = self.chain.fkin(self.q)
        grav = self.chain.g()

        # 3) Compute joint torques
        # gravity compensation
        tau = grav

        # TODO: THIS IS FOR TESTING
        # if self.T_goal is None:
        #     self.T_goal = T_from_Rp(R_cur, p_cur)
        #     self.movetime_goal = 10000.
        #     self.start_time = self.get_clock().now()

        # TASK SPACE
        if self.T_goal is not None:
            now = self.get_clock().now()
            if (now - self.start_time).nanoseconds*1e-9 > self.movetime_goal:
                self.T_goal = None
                self.Te_buf = None
                # self.gripper_goal_width = None
                # self.gripper_goal_force = None # TODO: 0?
                self.get_logger().info("Trajectory end. Returning to floating mode.")
                return

            tau += self.task_space_torque(p_cur, R_cur, Jv, Jw)
            tau += self.centering_torque(Jv, Jw)
            tau += self.overcome_friction()
            # tau = np.clip(tau, -5., 5.)

        # JOINT SPACE
        elif self.q_goal is not None:
            # TODO: check if at goal, then stop
            now = self.get_clock().now()
            if (now - self.start_time).nanoseconds*1e-9 > self.timeout:
                self.q_goal = None
            return

        # 4) Send joint torques
        for i in range(self.chain.dofs):
            self.arm.move_mit(i+1, p_des=0, v_des=0, kp=0, kd=0, 
                t_ff=tau[i])
            
        # gripper
        if self.gripper_goal_width is not None:
            self.gripper.move(width=self.gripper_goal_width, force=self.gripper_goal_force)
        

    # slower callback to publish joint state messages.
    def cb_publish(self):
        with self.lock:
            q = self.q
            qdot = self.qdot
            efforts = self.efforts
            t = self.js_time

        if q is None:
            return

        # add these to message
        msg = JointState()
        msg.header.stamp = float_to_rostime(t)

        joints_data = []
        # arm
        joints_data.extend(
            (joint_name, joint_state, velocity, effort)
            for joint_name, joint_state, velocity, effort 
            in zip(self.chain.joint_names, q.tolist(), qdot.tolist(), efforts.tolist())
        )
        # gripper
        joints_data.extend(self.get_gripper_joint_data())
        if joints_data:
            msg.name, msg.position, msg.velocity, msg.effort =map(list, zip(*joints_data))
            self.pub_jointstates.publish(msg)

    # SUBSCRIBER CALLBACKS
    # save the pose command and move time
    def cb_posecmd(self, msg: PoseCommand):
        if msg.header.frame_id != self.tip_frame:
            self.get_logger().warn(f"Received pose command in {msg.header.frame_id}, must be in {self.tip_frame}. Ignoring...")
            return
        v = msg.v_goal

        T_goal = Posemsg_to_T(msg.pose_goal)
        v_goal = np.array([v.linear.x, v.linear.y, v.linear.z,
                           v.angular.x, v.angular.y, v.angular.z])
        movetime_goal = msg.move_time.sec + 1e-9*msg.move_time.nanosec
        with self.lock:
            self.T_goal = T_goal
            self.v_goal = v_goal
            self.movetime_goal = movetime_goal
            self.start_time = self.get_clock().now()

    def cb_jointcmd(self, msg: JointState):
        # ARM JOINT COMMANDS
        joint_name_idx = {name: i for i, name in enumerate(msg.name)}
        # check if there are any arm commands
        if len(joint_name_idx) > 1:
            self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
            q = np.zeros(len(self.joint_names))
            qdot = np.zeros_like(q)
            for i, name in enumerate(self.joint_names):
                if name not in joint_name_idx:
                    # just keep the current position
                    q[i] = self.q[i]
                    qdot[i] = self.qdot[i]
                    break
                
                idx = joint_name_idx[name]
                q[i] = msg.position[idx]
                qdot[i] = msg.velocity[idx]

            # MOVE THE ARM
            # self.arm.set_speed_percent(15)
            # self.arm.set_speed_percent(15)
            # self.arm.set_speed_percent(15)
            # self.arm.move_j(q.tolist())
            # self.arm.move_j(q.tolist())

            # save
            with self.lock:
                self.q_goal = q
                self.qdot_goal = qdot
                self.start_time = self.get_clock().now()
                self.T_goal = None


        # GRIPPER COMMANDS
        joint_pos = { name: msg.position[idx] for idx, name in enumerate(msg.name) }
        joint_effort = { name: msg.effort[idx] for idx, name in enumerate(msg.name) }
        gripper_joint_map = {
            "gripper":   1.0,
            "gripper_joint1":    2.0,
            "gripper_joint2":    2.0,
        }
        matched = next(((name, scale) for name, scale in gripper_joint_map.items() if name in joint_pos), None)
        if matched is None:
            return
        joint_name, scale = matched
        width = abs(joint_pos[joint_name]) * scale
        # Use default force if effort is 0 or not specified
        force = joint_effort.get(joint_name, self.gripper_default_effort) or self.gripper_default_effort
        self.gripper_goal_width = np.clip(width, 0., 0.1)
        self.gripper_goal_force = np.clip(force, 0.5, 5.0)


    # OTHER HANDY FUNCTIONS
    # read joint states from the robot.
    # this runs with the controller, not telemetry.
    def get_joint_states(self):
        q = []
        qdot = []
        efforts = []
        js_time = 0.
        for joint_index in range(1, self.chain.dofs+1):
            ms = self.arm.get_motor_states(joint_index)
            if ms is None:
                return False
            q.append(ms.msg.position)
            qdot.append(ms.msg.velocity)
            efforts.append(ms.msg.torque)
            js_time = ms.timestamp

        # modify velocities to match URDF
        qdot[6] = -qdot[6]
        qdot[4] = -qdot[4]
        qdot[3] = -qdot[3]
        qdot[2] = -qdot[2]
        qdot[1] = -qdot[1]
        qdot[0] = -qdot[0]

        
        q = np.array(q)
        qdot = np.array(qdot)
        efforts = np.array(efforts)
        with self.lock:
            self.q = q
            self.qdot = qdot
            self.efforts = efforts
            self.js_time = js_time

        return True

    # read gripper data from robot.
    def get_gripper_joint_data(self):
        if self.gripper is None or not self.gripper.is_ok():
            return []
        status = self.gripper.get_status()
        if status is None:
            return []
        gripper_joint_map = {
            "gripper_joint1":     0.5,
            "gripper_joint2":    -0.5,
        }
        return [
            (name, status.width * scale, 0.0, status.force)
            for name, scale in gripper_joint_map.items()
        ]

    # IMPEDANCE FUNCTIONS
    # compute the torque to go to a task space position
    def task_space_torque(self, p_cur, R_cur, Jv, Jw):
        if self.v_goal is None:
            self.v_goal = np.zeros(6)

        # proportional control
        Kp = [140., 4.0]
        Fv_p = Kp[0]*ep(p_from_T(self.T_goal), p_cur)
        Fw_p = Kp[1]*eR(R_from_T(self.T_goal), R_cur)

        # derivative control (velocities)
        Kd = [6.1, 0.65]
        Fv_d = Kd[0]*ep(self.v_goal[:3], Jv @ self.qdot).reshape([3,1])
        Fw_d = Kd[1]*ep(self.v_goal[3:], Jw @ self.qdot).reshape([3,1])

        # impedance equation
        tau_goal  = Jv.T @ (Fv_p + Fv_d)
        tau_goal += Jw.T @ (Fw_p + Fw_d)

        # integral control
        if self.Te_buf is None:
            buffer_size = 10
            self.Te_buf = deque(maxlen=buffer_size)
        else:
            Ki = [0.28, 0.05]
            Fv_i = Ki[0]*sum(self.Te_buf)[:3]
            Fw_i = Ki[1]*sum(self.Te_buf)[3:]
            tau_goal += Jv.T @ Fv_i + Jw.T @ Fw_i
        self.Te_buf.append(np.vstack([ep(p_from_T(self.T_goal), p_cur), eR(R_from_T(self.T_goal), R_cur)]))

        return tau_goal

    def centering_torque(self, Jv, Jw):
        J = np.vstack([Jv, Jw]) # 6 x DOFs
        dofs = J.shape[1]
        q = self.q.reshape([dofs,1])

        Kq = 0.65
        tau_center = -Kq * q
        J_dampedinv = J.T @ np.linalg.inv(J @ J.T + 1e-6*np.eye(J.shape[0]))
        # nullspace_projector = np.eye(dofs) - np.linalg.pinv(J) @ J
        nullspace_projector = np.eye(dofs) - J_dampedinv @ J

        return nullspace_projector.T @ tau_center

    def overcome_friction(self):
        # tau_static = np.ones([7,1])*0.35
        # tau_static = np.vstack([np.ones([4,1])*0.5, np.ones([3,1])*0.15])
        tau_static = np.array([0.25, 0.4, 0.3, 0.4, 0.2, 0.15, 0.15])
        tau_static = tau_static.reshape([7,1])
        eps = 0.025
        return tau_static * np.tanh(self.qdot.reshape([7,1]) / eps) # essentially just sign(self.qdot)


    # SAFE NODE DESTROY
    def destroy_node(self):
        super().destroy_node()
        self.arm.set_leader_mode()
        self.arm.disconnect()
        

def main():
    rclpy.init()
    node = ControllerNode()

    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()