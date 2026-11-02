#!/usr/bin/env python3
# Calibrate the arm camera relative to the gripper
import sys
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Transform
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo


## chessboard setup
CHESS_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Number of INNER corners, not number of squares
BOARD_SIZE = (9, 6)
# Physical size of one square [m]
SQUARE_SIZE = 0.0217
# 3D coordinates of chessboard corners.
# Board lies in the Z=0 plane.
CHESS_OBJ_PTS = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
CHESS_OBJ_PTS[:, :2] = np.mgrid[
    0:BOARD_SIZE[0],
    0:BOARD_SIZE[1]
].T.reshape(-1, 2)
CHESS_OBJ_PTS *= SQUARE_SIZE


### HELPERS
def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.asarray(t).ravel(); return M

def Tmsg_to_T(msg: TransformStamped):
    t = msg.transform.translation
    t = np.array([t.x, t.y, t.z])
    q = msg.transform.rotation
    quat = np.array([float(q.x), float(q.y), float(q.z), float(q.w)])
    R = Rot.from_quat(quat).as_matrix()
    return T(R, t)

def T_to_Tmsg(T) -> Transform:
    transform = Transform()
    transform.translation.x = float(T[0,3])
    transform.translation.y = float(T[1,3])
    transform.translation.z = float(T[2,3])
    q = Rot.as_quat(Rot.from_matrix(T[:3,:3]))
    transform.rotation.x = float(q[0])
    transform.rotation.y = float(q[1])
    transform.rotation.z = float(q[2])
    transform.rotation.w = float(q[3])
    return transform



# Plan:
# 1) Calibrate front camera relative to fixed board and publish transform
#   (here, you can check by comparing the point clouds in rviz)
# 2) Calibrate arm camera relative to fixed board and publish transform
#   (here, you can check by looking at the point clouds in rviz)
# 3) Calibrate arm camera to gripper using hand-eye calibration stuffs
# 4) Save



class CalibrationNode(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration')

        # parameters
        self.declare_parameter("mode", "none") # cal_f, cal_a
        self.declare_parameter("tf_parent_board", False)
        self.declare_parameter("solve_robotworldhandeye", False)

        self.mode = self.get_parameter("mode").value
        self.tf_parent_board = self.get_parameter("tf_parent_board").get_parameter_value().bool_value
        self.solve_robotworldhandeye = self.get_parameter("solve_robotworldhandeye").get_parameter_value().bool_value
        self.total_calib = 30

        # tf setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.broadcaster = TransformBroadcaster(self)
        self.bridge = CvBridge()

        # the transformers we are solving for
        self.T_FcamBoard = np.eye(4)
        self.T_AcamBoard = np.eye(4)
        self.T_AcamGripper = None
        self.T_BaseBoard = None
        self.K = {}
        self.D = {}

        # helpers for the camera
        self.poses = {"AcamBoard": [], "BaseGripper": []}
        self.handeye_updated = False
        self.calibrate_handeye = False

        # image subscriber
        self.sub_fcam_info = self.create_subscription(
            CameraInfo, "camera_f/color/camera_info", self.cb_fcam_info, 1)
        self.sub_acam_info = self.create_subscription(
            CameraInfo, "camera_r/color/camera_info", self.cb_acam_info, 1)
        self.sub_fcam = self.create_subscription(
            Image, "camera_f/color/image_raw", self.cb_fcam, qos_profile_sensor_data)
        self.sub_acam = self.create_subscription(
            Image, "camera_r/color/image_raw", self.cb_acam, qos_profile_sensor_data)
        
        # a publish timer
        publish_period_sec = 1.0
        self.tf_pub_timer = self.create_timer(publish_period_sec, self.pub_tfs)

        self.get_logger().info("Node started. Use ros2 param set /hand_eye_calibration mode 'cal_f'/'cal_a'/'cal_handeye' to start.")


    def pub_tfs(self):
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.tf_parent_board = self.get_parameter("tf_parent_board").get_parameter_value().bool_value

        transforms = []
        if self.tf_parent_board:
            transforms = self.make_board_parent_tfs()
        else:
            transforms = self.make_base_parent_tfs()

        # publish!
        if len(transforms) > 0:
            self.broadcaster.sendTransform(transforms)


    def make_board_parent_tfs(self):
        transforms = []

        if self.T_FcamBoard is not None:
            # publish as board -> camera_f_link
            Tmsg_FcamlinkFcam = self.tf_buffer.lookup_transform(
                'camera_f_link',
                'camera_f_color_optical_frame',
                rclpy.time.Time()
            )
            T_FcamlinkFcam = Tmsg_to_T(Tmsg_FcamlinkFcam)
            T_BoardFcamlink = np.linalg.inv(T_FcamlinkFcam @ self.T_FcamBoard)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "calibration_board"
            transform.child_frame_id = "camera_f_link"
            transform.transform = T_to_Tmsg(T_BoardFcamlink)
            transforms.append(transform)


        if self.T_AcamBoard is not None:
            # publish as board -> camera_r_link
            Tmsg_AcamlinkAcam = self.tf_buffer.lookup_transform(
                'camera_r_link',
                'camera_r_color_optical_frame',
                rclpy.time.Time()
            )
            T_AcamlinkAcam = Tmsg_to_T(Tmsg_AcamlinkAcam)
            T_BoardAcamlink = np.linalg.inv(T_AcamlinkAcam @ self.T_AcamBoard)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "calibration_board"
            transform.child_frame_id = "camera_r_link"
            transform.transform = T_to_Tmsg(T_BoardAcamlink)
            transforms.append(transform)


        if self.T_AcamGripper is not None:
            T_GripperbaseBoard_throughAcam = np.linalg.inv(self.T_AcamGripper) @ self.T_AcamBoard

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "nero_right/gripper_base"
            transform.child_frame_id = "calibration_board"
            transform.transform = T_to_Tmsg(T_GripperbaseBoard_throughAcam)
            transforms.append(transform)

        return transforms

    def make_base_parent_tfs(self):
        transforms = []

        if self.T_FcamBoard is not None:
            # publish as board -> camera_f_link
            Tmsg_FcamlinkFcam = self.tf_buffer.lookup_transform(
                'camera_f_link',
                'camera_f_color_optical_frame',
                rclpy.time.Time()
            )
            T_FcamlinkFcam = Tmsg_to_T(Tmsg_FcamlinkFcam)
            T_BoardFcamlink = np.linalg.inv(T_FcamlinkFcam @ self.T_FcamBoard)

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "calibration_board"
            transform.child_frame_id = "camera_f_link"
            transform.transform = T_to_Tmsg(T_BoardFcamlink)
            transforms.append(transform)


        if self.T_BaseBoard is not None:
            # publish as ArmBase -> calibration_board

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "nero_right/world"
            transform.child_frame_id = "calibration_board"
            transform.transform = T_to_Tmsg(self.T_BaseBoard)
            transforms.append(transform)


        if self.T_AcamGripper is not None:
            # publish as GripperBase -> camera_r_link
            Tmsg_AcamAcamlink = self.tf_buffer.lookup_transform(
                'camera_r_color_optical_frame',
                'camera_r_link',
                rclpy.time.Time()
            )
            T_AcamAcamlink = Tmsg_to_T(Tmsg_AcamAcamlink)
            T_GripperbaseAcamlink = np.linalg.inv(self.T_AcamGripper) @ T_AcamAcamlink

            # publish
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "nero_right/gripper_base"
            transform.child_frame_id = "camera_r_link"
            transform.transform = T_to_Tmsg(T_GripperbaseAcamlink)
            transforms.append(transform)

        return transforms


    ## Camera intrinsics!
    def cb_info(self, cam_name, msg: CameraInfo):
        self.K[cam_name] = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D[cam_name] = np.array(msg.d, dtype=np.float64)
        self.get_logger().info(f"Got {cam_name} intrinsics")
    def cb_fcam_info(self, msg: CameraInfo):
        self.cb_info("cam_f", msg)
        # Tear down the subscription so we never pay for it again.
        self.destroy_subscription(self.sub_fcam_info)
        self.cb_fcam_info = None
    def cb_acam_info(self, msg: CameraInfo):
        self.cb_info("cam_a", msg)
        # Tear down the subscription so we never pay for it again.
        self.destroy_subscription(self.sub_acam_info)
        self.cb_acam_info = None

    ## Camera image subscribers!
    def cb_fcam(self, msg: Image):
        if self.mode != "cal_f":
            return
        
        if self.T_FcamBoard is not None:
            self.get_logger().info("Calibrating T_FcamBoard...")
            self.T_FcamBoard = None # turn off publishing while calibrating
            self.obj_points = []
            self.img_points = []

        # termination condition
        if len(self.obj_points) > self.total_calib:
            self.T_FcamBoard = self.get_cam_pose("cam_f", self.obj_points, self.img_points)
            self.mode = "none"
            self.set_parameters([rclpy.parameter.Parameter('mode', rclpy.Parameter.Type.STRING, 'none')])
            cv.destroyAllWindows()
            # also save T_FcamBoard
            print(self.T_FcamBoard)
            self.get_logger().info("Calibrated T_FcamBoard. Saving...")
            np.save("T_FcamBoard.npy", self.T_FcamBoard)
            return

        img, pts = self.detect_board(msg)
        if img is None:
            return
        # show images
        cv.imshow("calibration", img)
        cv.waitKey(1)

        # save calibration points
        if pts is not None:
            self.obj_points.append(pts[0])
            self.img_points.append(pts[1])
        

    def cb_acam(self, msg: Image):
        if self.mode != "cal_a":
            return

        if not self.calibrate_handeye:
            # self.get_logger().info("Press c to take next calibration point")
            # self.T_AcamBoard = None # turn off publishing while calibrating
            self.obj_points = []
            self.img_points = []

        # termination condition for PnP
        if len(self.obj_points) > self.total_calib:
            # get current gripper pose
            if self.tf_buffer.can_transform(
                    'nero_right/base_link',
                    'nero_right/gripper_base',
                    msg.header.stamp
            ):
                Tmsg_BaseGripper = self.tf_buffer.lookup_transform(
                    'nero_right/base_link',
                    'nero_right/gripper_base',
                    msg.header.stamp
                )
                T_BaseGripper = Tmsg_to_T(Tmsg_BaseGripper)
            else:
                self.get_logger().debug(f'Gripper fkin failed... trying again')
                return

            # get current camera pose
            self.T_AcamBoard = self.get_cam_pose("cam_a", self.obj_points, self.img_points)
            self.get_logger().info("Recorded a calibration point!")

            # save these
            self.poses["AcamBoard"].append(self.T_AcamBoard)
            self.poses["BaseGripper"].append(T_BaseGripper)
            self.calibrate_handeye = False

            print(len(self.poses["AcamBoard"]))
            self.handeye_updated = False
            return
        
        # solve hand-eye and publish if able
        if len(self.poses["AcamBoard"]) >= 3 and (not self.handeye_updated):

            self.solve_robotworldhandeye = self.get_parameter("solve_robotworldhandeye").get_parameter_value().bool_value
            if self.solve_robotworldhandeye:
                T_AcamGripper, T_FcamBase =  self.solve_robothandeye()
            else:
                T_AcamGripper, T_FcamBase = self.solve_handeye()

            # save
            self.get_logger().info("Calibrated T_AcamGripper. Saving...")
            np.save("T_AcamGripper.npy", T_AcamGripper)
            np.save("T_FcamBase.npy", T_FcamBase)
            return
    
        # detect the board and gather board points
        img, pts = self.detect_board(msg)
        if img is None:
            return
        # show images
        cv.imshow("calibration", img)
        key = cv.waitKey(1) & 0xFF

        # only calibrate if the key was c
        if key == ord('c'):
            self.get_logger().info("c pressed!")
            self.calibrate_handeye = True
        
        if self.calibrate_handeye:
            # save pnp calibration points
            if pts is not None:
                self.obj_points.append(pts[0])
                self.img_points.append(pts[1])

    """
    Hand-eye calibration
    """
    def solve_handeye(self):
        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []
        for T_AB, T_CG in zip(self.poses["AcamBoard"], self.poses["BaseGripper"]):
            # my convention is the reverse of the opencv one
            # T_BA  means transform arm2board (arm=cam, board=target)
            R_gripper2base.append(T_CG[:3,:3])
            t_gripper2base.append(T_CG[:3,3].reshape([3,1]))
            R_target2cam.append(T_AB[:3,:3])
            t_target2cam.append(T_AB[:3,3].reshape([3,1]))

        # compute calibration
        R_cam2grip, t_cam2grip = cv.calibrateHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            method=cv.CALIB_HAND_EYE_TSAI
        )
        T_GripperAcam = T(R_cam2grip, t_cam2grip)

        # print residuals!
        Tref_BoardBase = np.linalg.inv(self.poses["BaseGripper"][0] @ T_GripperAcam @ self.poses["AcamBoard"][0])
        for Ti_AcamBoard, Ti_BaseGripper in zip(self.poses["AcamBoard"], self.poses["BaseGripper"]):
            Ti_BaseBoard = Ti_BaseGripper @ T_GripperAcam @ Ti_AcamBoard
            E = Ti_BaseBoard @ Tref_BoardBase  # should be identity
            dt = np.linalg.norm(E[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
            print(f"resid: {dt*1000:.1f} mm, {dr:.2f} deg")

        # transforms to publish
        self.T_AcamGripper = np.linalg.inv(T_GripperAcam)
        T_BaseGripper = self.poses["BaseGripper"][-1]
        self.T_BaseBoard = T_BaseGripper @ np.linalg.inv(self.T_AcamGripper) @ self.T_AcamBoard
        self.handeye_updated = True

        # transforms for saving
        T_FcamBase = self.T_FcamBoard @ np.linalg.inv(self.T_AcamBoard) @ self.T_AcamGripper @ np.linalg.inv(T_BaseGripper)

        return self.T_AcamGripper, T_FcamBase


    """
    Hand-eye calibration
    """
    def solve_robothandeye(self):
        R_world2cam, t_world2cam = [], []
        R_base2gripper, t_base2gripper = [], []
        for T_AB, T_CG in zip(self.poses["AcamBoard"], self.poses["BaseGripper"]):
            # my convention is the reverse of the opencv one
            # T_BA  means transform arm2board (arm=cam, board=target)
            T_GC = np.linalg.inv(T_CG)
            R_base2gripper.append(T_GC[:3,:3])
            t_base2gripper.append(T_GC[:3,3].reshape([3,1]))
            R_world2cam.append(T_AB[:3,:3])
            t_world2cam.append(T_AB[:3,3].reshape([3,1]))

        # compute calibration
        R_base2world, t_base2world, R_gripper2cam, t_gripper2cam = cv.calibrateRobotWorldHandEye(
            R_world2cam, t_world2cam, R_base2gripper, t_base2gripper,
            method=cv.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )
        T_AcamGripper = T(R_gripper2cam, t_gripper2cam)
        T_BoardBase = T(R_base2world, t_base2world)


        # print residuals!
        # AX = ZB
        # A: world2cam    -- T_AcamBoard
        # X: base2world   -- T_BoardBase
        # Z: gripper2cam  -- T_AcamGripper
        # B: base2gripper -- T_GripperBase
        for Ti_AcamBoard, Ti_BaseGripper in zip(self.poses["AcamBoard"], self.poses["BaseGripper"]):
            AX = Ti_AcamBoard @ T_BoardBase
            ZB = T_AcamGripper @ np.linalg.inv(Ti_BaseGripper)
            E = AX @ ZB  # should be identity
            dt = np.linalg.norm(E[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))
            print(f"resid: {dt*1000:.1f} mm, {dr:.2f} deg")

        # transforms to publish
        self.T_AcamGripper = T_AcamGripper
        self.T_BaseBoard = np.linalg.inv(T_BoardBase)
        self.handeye_updated = True

        # transforms for saving
        T_FcamBase = self.T_FcamBoard @ T_BoardBase

        return self.T_AcamGripper, T_FcamBase


    def detect_board(self, msg: Image):
        """
        Get current camera image and detect
        calibration board.

        Returns None if board not detected
        """
        img = self.bridge.imgmsg_to_cv2(msg) #, encoding="passthrough")
        if img is None:
            return None, None

        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # find board markers
        ret, corners = cv.findChessboardCorners(grey, BOARD_SIZE, None)
        if not ret:
            return img, None
        
        # collect image/corner points
        obj_points = CHESS_OBJ_PTS
        img_points = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), CHESS_CRITERIA)
        img = cv.drawChessboardCorners(img, BOARD_SIZE, img_points, ret)

        return img, (obj_points, img_points)

    def get_cam_pose(self, cam_name, objpoints, imgpoints):
        """
        Solve PnP for camera pose relative to board
        """
        all_object_points = np.concatenate(
            [p.reshape(-1, 3) for p in objpoints],
            axis=0,
        ).astype(np.float32)

        all_image_points = np.concatenate(
            [p.reshape(-1, 2) for p in imgpoints],
            axis=0,
        ).astype(np.float32)

        ok, rvec, tvec, err = cv.solvePnPGeneric(
            all_object_points,
            all_image_points,
            self.K[cam_name],
            self.D[cam_name],
            flags=cv.SOLVEPNP_ITERATIVE #cv.SOLVEPNP_SQPNP
        )
        if not ok:
            raise RuntimeError("Could not estimate the fixed board-camera pose")
        print("Reprojection errors:")
        print(err)

        # more errors (just for fun)
        img_pred, _ = cv.projectPoints(
            all_object_points,
            rvec[0],
            tvec[0],
            self.K[cam_name],
            self.D[cam_name]
        )
        err = img_pred.reshape(-1, 2) - all_image_points.reshape(-1, 2)
        rmse = np.sqrt(np.mean(np.sum(err**2, axis=1)))
        mean_err = np.mean(np.linalg.norm(err, axis=1))
        max_err = np.max(np.linalg.norm(err, axis=1))
        print(f"rmse: {rmse} px")
        print(f"mean_err: {mean_err} px")
        print(f"max_err: {max_err} px")


        R_CamBoard, _ = cv.Rodrigues(rvec[0])

        T_CamBoard = np.eye(4)
        T_CamBoard[:3,:3] = R_CamBoard
        T_CamBoard[:3,3] = tvec[0].flatten()
        return T_CamBoard


    def calibrate_camera(self, cam_name):
        print("-----------------")
        print(f"Calibrate {cam_name}")
        print("Fix the calibration board in view of the camera.")
        input(f"Begin?")

        obj_points = []
        img_points = []
        while True:
            if len(obj_points) > self.total_calib:
                break

            img, pts = self.detect_board(cam_name)
            if img is None:
                continue
            # show images
            cv.imshow("calibration", img)
            cv.waitKey(1)

            if pts is None:
                continue
            # save
            obj_points.append(pts[0])
            img_points.append(pts[1])

        cv.destroyAllWindows()

        # compute pose using pnp
        T_CamBoard = self.get_cam_pose(cam_name, obj_points, img_points)
        return T_CamBoard



def main(args=None):
    rclpy.init(args=args)

    try:
        node = CalibrationNode()
        rclpy.spin(node)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    finally:
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
    # main()