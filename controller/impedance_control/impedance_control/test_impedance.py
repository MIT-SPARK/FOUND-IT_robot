# goto_point.py
#
# Simple node to direct the robot to a position
# 
# PUBLISHES:
# /goal_pose        joey_interfaces.msg/PoseCommand

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Twist, PointStamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, TransformException
from scipy.spatial.transform import Rotation as Rot
from std_srvs.srv import Trigger
from std_msgs.msg import Bool
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy

from joey_interfaces.msg import PoseCommand
from impedance_control.utils.TransformHelpers import *
from impedance_control.utils.Convert import *
from impedance_control.utils.Segments import GotoRP, HoldRP, StayRP, Segment, SetGripper, GraspSeg

import math
import time
import sys

# T_stage = T_from_Rp(np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]]), np.array([-0.316, 0., 0.423]).reshape([3,1]))
T_stage = np.array([[-0.163, -0.291, -0.943, -0.167],
                    [-0.131,  0.953, -0.272, -0.041],
                    [0.978 , 0.079 ,-0.193 , 0.402],
                    [0.000 , 0.000 , 0.000 , 1.000],])
T_home = np.array([[-0.038, -0.146, -0.989,  0.120],
                   [-0.146,  0.979, -0.139, -0.023],
                   [ 0.989,  0.139, -0.059,  0.440],
                   [ 0.000,  0.000,  0.000,  1.000]])

q_stage = np.array([0.077, -0.64, 0.015, 2.18, -0.059, -0.223, 0.202])

PLACE_OFFSET_Z = -0.11 # meters offset

def get_pregrasp_pose(T_grasp):
    T_offset = np.eye(4)
    T_offset[2,3] = -0.15 # amount to move back by
    return T_grasp @ T_offset

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.tip_frame = "nero_right/gripper_palm"

        # parameters
        # True  -- wait for the execute_grasp_impedance service before moving
        #          (run execute_on_enter and press Enter).
        # False -- execute as soon as a grasp or place target arrives, so the
        #          arm moves on a perception event with no human in the loop.
        # Read at point of use rather than cached, so `ros2 param set` takes
        # effect without restarting the node.
        self.declare_parameter('require_confirmation', True)

        self.T_grasp = None
        self.T_place = None
        self.T_where_placed = None

        self.pub_cmd = self.create_publisher(
            PoseCommand, "/pose_cmd", 1
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.traj = None
        self.start_time = rclpy.duration.Duration()

        # grasp command subscriber
        self.create_subscription(PoseStamped, "nero_right/control/move_p", self.cb_grasp, 1)
        self.create_subscription(PointStamped, 'place_obj_location', self.cb_place, 1)
        self.create_subscription(JointState, "/nero_right/puppet/joint_states", self.cb_joint_states, 1)
        self.create_service(Trigger, 'nero_right/execute_grasp_impedance', self.on_execute_requested)

        # communication with found it
        self.pub_grasp_attempted = self.create_publisher(Bool, '/grasp_attempted', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.pub_grasp_success = self.create_publisher(Bool, '/grasp_success', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))

        # publish timer
        self.dt = 0.009
        self.timer = self.create_timer(self.dt, self.cb_timer)
        self.pub_grippercmd = self.create_publisher(JointState, "/joint_cmd", 1)

        self.get_logger().info(
            "waiting for targets; "
            + ("run execute_on_enter and press Enter to execute"
               if self.get_parameter('require_confirmation').value
               else "executing automatically on each target")
        )


    def maybe_auto_execute(self):
        """Fire the execute gate ourselves unless a human confirmation is required."""
        if self.get_parameter('require_confirmation').value:
            return
        # Go through on_execute_requested rather than calling create_*_traj
        # directly: its both-targets-set and trajectory-busy guards are exactly
        # the ones auto-execution needs, and there is no second copy to drift.
        self.on_execute_requested(Trigger.Request(), Trigger.Response())

    def on_execute_requested(self, request, response):
        if self.T_grasp is not None and self.T_place is not None:
            self.get_logger().warn("Recieved place & pick at the same time! Cannot proceed.")
            response.success = False
            return response

        # traj is None until the first cb_timer tick, and stays None for as long
        # as TF is unavailable -- treat "not ready yet" as busy. Reachable now
        # that a target callback can request execution during startup.
        if self.traj is None or len(self.traj) > 1:
            self.get_logger().warn("Please wait for trajectory to finish before starting next one!")
            response.success = False
            return response

        print(self.T_grasp)
        if self.T_grasp is not None:
            self.create_pick_traj()
        elif self.T_place is not None:
            self.create_place_traj()


        self.execute_requested = True
        response.success = True
        response.message = 'execute requested'
        return response
    
    def create_pick_traj(self):
        T_grasp = self.T_grasp
        T_pregrasp = get_pregrasp_pose(self.T_grasp)
        T_cur = self.get_tip_pose()

        self.traj = [HoldRP(T_cur, 1.0),
                     SetGripper(0.1),
                     GotoRP(T_cur, T_stage, 3.0),
                     GotoRP(T_stage, T_pregrasp, 3.0),
                     SetGripper(0.1),
                     HoldRP(T_pregrasp, 1.0),
                     GotoRP(T_pregrasp, T_grasp, 2.0),
                     HoldRP(T_grasp, 1.0),
                     GraspSeg(),
                     HoldRP(T_grasp, 1.0),
                     GotoRP(T_grasp, T_pregrasp, 2.0),
                    #  GotoRP(T_pregrasp, T_stage, 8.0, tag="joint"),
                     GotoRP(T_pregrasp, T_stage, 15.0),
                     HoldRP(T_stage, 2.0),
                     GotoRP(T_stage, T_home, 3.0),
                     StayRP(T_home, space="pick")]
                    #  StayRP(T_stage)]
        self.T_grasp = None
        self.published_tofoundit = False
        self.start_time = self.get_clock().now()
        self.get_logger().info("Starting pick trajectory")

    def create_place_traj(self):
        if self.T_where_placed is None:
            T_place = self.T_place
        else:
            T_place = self.T_where_placed
            T_place[:3,3] = self.T_place[:3,3]
        T_preplace = get_pregrasp_pose(T_place)
        T_cur = self.get_tip_pose()
        
        self.traj = [HoldRP(T_cur, 1.0),
                     SetGripper(0.0),
                     GotoRP(T_cur, T_stage, 3.0),
                     SetGripper(0.0),
                     GotoRP(T_stage, T_preplace, 3.0),
                     GotoRP(T_preplace, T_place, 1.5),
                     HoldRP(T_place, 1.0),
                     SetGripper(0.1),
                     HoldRP(T_place, 1.0),
                     GotoRP(T_place, T_stage, 3.0),
                     HoldRP(T_stage, 2.0),
                     GotoRP(T_stage, T_home, 3.0),
                     StayRP(T_home, space="place")]
                    #  StayRP(T_stage)]
        self.T_place = None
        self.T_where_placed = None
        self.published_tofoundit = False
        self.start_time = self.get_clock().now()
        self.get_logger().info("Starting place trajectory")



    # listen for grasp message but don't start or plan trajectory
    def cb_grasp(self, msg: PoseStamped):
        self.get_logger().info("Received grasp!")
        X_BO = Posemsg_to_T(msg.pose)
        self.T_grasp = X_BO
        self.maybe_auto_execute()

    def cb_place(self, msg: PointStamped):
        self.get_logger().info("Received place!")
        while True:
            try:
                transform = self.tf_buffer.lookup_transform(
                        'nero_right/world', msg.header.frame_id, rclpy.time.Time()
                    )
                break
            except Exception as exp:
                self.get_logger().warn(exp)

        T = Tmsg_to_T(transform.transform)
        place_p = Pointmsg_to_p(msg.point)
        place_world = transform_point(T, place_p)
        rotX = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]]) # TODO: make this bias towards arm base
        X_BO = T_from_Rp(rotX, place_world)
        place_T_offset = T_from_Rp(np.eye(3), np.array([[0, 0, -PLACE_OFFSET_Z]]).T)
        self.T_place = X_BO @ place_T_offset
        self.maybe_auto_execute()

    # trajectory publishing
    def cb_timer(self):
        if self.traj is None:
            try:
                self.start_time = self.get_clock().now()
                self.traj = []
                # self.get_logger().info("Starting trajectory")
            except TransformException as ex:
                self.get_logger().info(
                    f'Waiting for transform: {ex}'
                )
                return


        now = self.get_clock().now()
        if len(self.traj) == 0:
            # trajectory hasn't started yet
            return
        t = (now - self.start_time).nanoseconds * 1e-9

        seg: Segment = self.traj[0]
        if seg.space == "gripper":
            self.pub_grippercmd.publish(seg.evaluate())
            if seg.is_grasp():
                self.T_where_placed = self.get_tip_pose()
            t = math.inf
            
        if not self.published_tofoundit and type(seg) == StayRP:
            self.pub_grasp_attempted.publish(Bool(data=seg.space == "pick"))
            if seg.space == "pick":
                if self.cur_gripper_width <= 0.007:
                    self.pub_grasp_success.publish(Bool(data=False))
                else:
                    self.pub_grasp_success.publish(Bool(data=True))
            self.published_tofoundit = True

        if t > seg.get_duration():
            # trajectory finished
            self.traj.pop(0)

            # compute error
            if seg.space != "gripper":
                T_tip = self.get_tip_pose()
                T_cmd = seg.evaluate(t)[0]
                deg_err = math.degrees(np.linalg.norm(eR(R_from_T(T_cmd), R_from_T(T_tip))))
                p_err = np.linalg.norm(ep(p_from_T(T_cmd), p_from_T(T_tip)))*1000
                self.get_logger().info(f"error: {p_err: .2f} mm, {deg_err: .2f} deg")
            
            if len(self.traj) == 0:
                self.start_time = rclpy.duration.Duration()
                self.get_logger().info("Finished trajectory")
                self.T_grasp = None
                return
            else:
                self.start_time = now
                t = 0.
                seg = self.traj[0]
                if seg.space == "gripper":
                    return
                
                if seg.tag == "joint":
                    self.send_joint_command(q_stage, np.zeros_like(q_stage))
                    self.traj.pop(0)
                    time.sleep(seg.get_duration())
                    next_seg = self.traj.pop(0)
                    tf0 = self.get_tip_pose()
                    tff = next_seg.evaluate(next_seg.get_duration())[0]
                    seg = GotoRP(tf0, tff, next_seg.get_duration())
                    self.traj.insert(0, seg)

                self.get_logger().info("Next segment")

        # get & publish the pose command
        T_goal, v_goal = seg.evaluate(t)
        self.send_pose_command(T_goal, v_goal)


    def send_pose_command(self, T_goal, v_goal, frame=None):
        msg = PoseCommand()
        if frame is None:
            msg.header.frame_id = self.tip_frame
        else:
            msg.header.frame_id = frame

        msg.pose_goal = T_to_Posemsg(T_goal)
        msg.move_time = float_to_rosduration(2*self.dt)
        msg.v_goal = Twist()
        msg.v_goal.linear.x = v_goal[0]
        msg.v_goal.linear.y = v_goal[1]
        msg.v_goal.linear.z = v_goal[2]
        msg.v_goal.angular.x = v_goal[3]
        msg.v_goal.angular.y = v_goal[4]
        msg.v_goal.angular.z = v_goal[5]

        self.pub_cmd.publish(msg)

    
    def send_joint_command(self, q_goal, qd_goal):
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        msg = JointState()
        names = self.joint_names
        positions = q_goal.tolist()
        velocity = qd_goal.tolist()
        effort = np.zeros_like(q_goal).tolist()

        msg.name = names
        msg.position = positions
        msg.velocity = velocity
        msg.effort = effort

        self.pub_grippercmd.publish(msg)

    def get_tip_pose(self):
        Tmsg_cur = self.tf_buffer.lookup_transform(
            'nero_right/base_link',
            self.tip_frame,
            # self.get_clock().now(),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0)
        )
        T_cur = Tmsg_to_T(Tmsg_cur.transform)
        return T_cur

    def get_link3_pose(self):
        Tmsg_cur = self.tf_buffer.lookup_transform(
            'nero_right/base_link',
            'nero_right/link3',
            # self.get_clock().now(),
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0)
        )
        T_cur = Tmsg_to_T(Tmsg_cur.transform)
        return T_cur

    def cb_joint_states(self, msg: JointState):
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
        self.cur_gripper_width = width
    

def main(args=None):
    rclpy.init(args=args)

    # rclpy.spin(MyNode())
    try:
        node = MyNode()
        rclpy.spin(node)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    main()
