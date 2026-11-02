#!/usr/bin/env python3
"""KinematicChainGravity.py

   chain = KinematicChain(node, basefame, tipframe, expectedjointnames)

      Initialize the kinematic chain, read from the URDF message on
      the topic /robot_description, sent from the robot_state_publisher.
      Extract the joints moving from the baseframe to the tipframe.
      Expect the active joints to match the given names.

   chain.name               Name of the robot
   chain.dofs               Active DOFS (# of active joints)
   chain.joints             List of joints (active and fixed) from URDF XML

   chain.data               Data for the active joints:
   chain.data.type[DOFS]    List of types per joint
   chain.data.e[DOFS]       List of joint axes w.r.t. world per joint
   chain.data.T[DOFS]       List of T matrices w.r.t. world per joint
   chain.data.Ttip          T matrix for tip w.r.t. world
   chain.data.grav          Vector of gravity torques

   chain.setjoints(q)       Update the chain.data to the joint position (q)

   chain.ptip()             Determine/return the tip position
   chain.Rtip()             Determine/return the tip orientation
   chain.Ttip()             Determine/return the tip transform
   chain.g()                Return the gravity vector
   chain.Jv()               Determine/return the tip translational Jacobian
   chain.Jw()               Determine/return the tip rotational Jacobian

   (ptip, Rtip, Jv, Jw) = chain.fkin(q)

      Update the kinematic chain.data and report the results.


   Note, internally, the URDF XML file is converted into a python list
   of Joint elements.  These contain

   Joint            URDF <joint> element
   Joint.name       Name of the joint
   Joint.parent     Parent link/frame
   Joint.child      Child  link/frame
   Joint.type       Type = 'fixed'                    fixed
                           'continuous', 'revolute'   rotational
                           'prismatic'                translational
   Joint.origin     Joint's fixed T matrix shift
   Joint.axis       Joint's axis of movement (if applicable)


   Node:        /kintest or as given
   Subscribe:   /robot_description      std_msgs/String

"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String
from urdf_parser_py.urdf import Robot

from .TransformHelpers import *

class KinematicChainData:
    def __init__(self):
        self.type = []  # List of 'revolute' or 'prismatic'
        self.e = []  # List of 3x1 joint axes
        self.T = []  # List of 4x4 transforms
        self.Ttip = None  # 4x4 tip transform
        self.grav = [] # Nx1 vector of gravity torques


class KinematicChain():
    """
    Low-level impedance control
    """
    def info(self, string):
        self.node.get_logger().info("KinematicChain: " + string)
    def error(self, string):
        self.node.get_logger().error("KinematicChain: " + string)
        raise Exception(string)


    def __init__(self, baseframe, tipframe, topic="/robot_description"):
        self.node = Node(f"kinematic_chain_{baseframe}_{tipframe}")

        ############
        ## Load URDF
        self.info("Waiting for URDF to be published...")
        self.urdf = None
        
        # create temporary subscriber with TRANSIENT_LOCAL quality
        # this means we see the last previously published message.
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        def cb(msg: String):
            self.urdf = msg.data
        sub_urdf = self.node.create_subscription(String, topic, cb, quality)
        while self.urdf is None:
            rclpy.spin_once(self.node)
        self.node.destroy_subscription(sub_urdf)

        # convert URDF string to robot object
        robot = Robot.from_xml_string(self.urdf.encode("utf-8"))
        self.name = robot.name
        self.info("URDF Robot '%s'" % self.name)

        # load joint list
        self.create_joint_list(robot, baseframe, tipframe)
        

    #############################
    ## URDF parsing and Jacobians
    def create_joint_list(self, robot, baseframe, tipframe):
        self.joints = []
        self.inertias = []
        # load joints from URDF
        # searching backwards from tip to base
        frame = tipframe
        while frame != baseframe:
            joint = next((j for j in robot.joints if j.child ==frame), None)
            link = next((l for l in robot.links if l.name == frame), None)
            if joint is None:
                self.error(f"Unable to find joint connecting to {frame}")
            if joint.parent == frame:
                self.error(f"Joint {joint.name} connects {frame} to itself")
            self.joints.insert(0, joint)
            self.inertias.insert(0, link.inertial)
            frame = joint.parent
        
        # report
        self.dofs = sum(1 for j in self.joints if j.type !="fixed")
        self.info(f"{len(self.joints)} total joints in URDF, {self.dofs} active DOFs:")
        dof = 0
        for i in range(len(self.joints)):
            joint = self.joints[i]
            inertia = self.inertias[i]
            if joint.type == "fixed":
                string = "Joint #%d fixed      '%s'" % (i, joint.name)
            elif joint.type == "continuous" or joint.type == "revolute":
                string = "Joint #%d rot DOF #%d '%s'" % (i, dof, joint.name)
                dof = dof + 1
            elif joint.type == "prismatic":
                string = "Joint #%d lin DOF #%d '%s'" % (i, dof, joint.name)
                dof = dof + 1
            else:
                self.error(
                    "Joint '%s' has unknown type '%s'" % (joint.name, joint.type)
                )
            string += "%-20s" % ("'" + joint.name + "'")
            if inertia is not None:
                string += "mass %6.3fkg" % inertia.mass
            else:
                string += "unknown mass"
            self.info(string)

        # save joint names
        self.joint_names = [j.name for j in self.joints if j.type != "fixed"]

    def set_joints(self, q):
        if q is None:
            return
        
        # check number of joints
        if len(q) != self.dofs:
            self.error(f"Number of joints angles ({len(q)}) does not match URDF ({self.dofs})")
        
        self.data = KinematicChainData()
        T = np.eye(4)
        grav = np.zeros((self.dofs, 1))

        # walk up the kinematic chain
        dof = 0
        for (joint, inertia) in zip(self.joints, self.inertias):
            if joint.type == "fixed":
                T = T @ T_from_URDF_origin(joint.origin)

            elif (joint.type == "continuous") or (joint.type == "revolute"):
                # Grab the joint axis in the local frame.
                elocal = e_from_URDF_axis(joint.axis)

                # First append the fixed transform, then the rotational
                # transform.  The joint angle comes from q-vector.
                T = T @ T_from_URDF_origin(joint.origin)
                T = T @ T_from_Rp(Rote(elocal, q[dof]), pzero())

                # Compute the joint axis w.r.t. world frame.
                e = R_from_T(T) @ elocal

                # Save the transform and advance the active DOF counter.
                self.data.type.append("revolute")
                self.data.e.append(e)
                self.data.T.append(T)
                dof += 1

            elif joint.type == "prismatic":
                # Grab the joint axis in the local frame.
                elocal = e_from_URDF_axis(joint.axis)

                # First append the fixed transform, then the translational
                # transform.  The joint displacement comes from q-vector.
                T = T @ T_from_URDF_origin(joint.origin)
                T = T @ T_from_Rp(Reye(), elocal * q[dof])

                # Compute the joint axis w.r.t. world frame.
                e = R_from_T(T) @ elocal

                # Save the transform and advance the active DOF counter.
                self.data.type.append("prismatic")
                self.data.e.append(e)
                self.data.T.append(T)
                dof += 1

            else:
                # There shouldn't be any other types...
                self.error("Unknown Joint Type: %s", joint.type)

            # Also look at the child link to get the mass and gravity.
            # Proceed if there is an inertial element.
            if inertia is not None:
                # Check whether to shift to the child link's center of mass.
                if inertia.origin is not None:
                    Tlink = T @ T_from_URDF_origin(inertia.origin)
                else:
                    Tlink = T

                # Extract the mass times g and center of mass location.
                mg = inertia.mass * 9.81
                pc = p_from_T(Tlink)

                # Add all the torques (below the current dof).  Note
                # the gravity vector is vertically up in world frame,
                # hence we simply take the 3rd element of the vector.
                for k in range(dof):
                    grav[k] += (
                        mg
                        * np.cross(
                            self.data.e[k], pc - p_from_T(self.data.T[k]), axis=0
                        )[2]
                    )

        # Also save the tip transform and gravity vector.
        self.data.Ttip = T
        self.data.grav = grav

    def ptip(self):
        return p_from_T(self.data.Ttip)

    def Rtip(self):
        return R_from_T(self.data.Ttip)

    def Ttip(self):
        return self.data.Ttip
    
    def g(self):
        return self.data.grav

    # Extract the Jacobian data.
    def Jv(self):
        J = np.zeros((3, self.dofs))
        for dof in range(self.dofs):
            if self.data.type[dof] == "revolute":
                dp = p_from_T(self.data.Ttip) - p_from_T(self.data.T[dof])
                J[:, dof : dof + 1] = cross(self.data.e[dof], dp)
            else:
                J[:, dof : dof + 1] = self.data.e[dof]
        return J

    def Jw(self):
        J = np.zeros((3, self.dofs))
        for dof in range(self.dofs):
            if self.data.type[dof] == "revolute":
                J[:, dof : dof + 1] = self.data.e[dof]
            else:
                J[:, dof : dof + 1] = np.zeros((3, 1))
        return J

    # All-in-one call.
    def fkin(self, q):
        # Compute the chain data.
        self.set_joints(q)

        # Extract and return the data.
        return (self.ptip(), self.Rtip(), self.Jv(), self.Jw())
    



def main(args=None):
    rclpy.init(args=args)

    chain = KinematicChain("world", "link7", topic = "/nero_right/robot_description")

    q_test = np.zeros(7)
    ptip, Rtip, Jv, Jw = chain.fkin(q_test)
    print(ptip)
    print(Rtip)


if __name__ == "__main__":
    main()

