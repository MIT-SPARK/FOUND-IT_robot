from __future__ import annotations

import threading

import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import UInt32
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point  # DEBUG-VIZ
from std_msgs.msg import ColorRGBA  # DEBUG-VIZ
from visualization_msgs.msg import Marker, MarkerArray  # DEBUG-VIZ
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from std_msgs.msg import Bool

# ============================================================================
# Keep all constants and grasp-generation functions from the original script:
#
#   CROP_MIN, CROP_MAX, P_GS_G, ...
#   make_transform
#   invert_transform
#   transform_points
#   rotation_x
#   grasp_candidate_cost
#   generate_antipodal_grasp_candidate
#   sample_grasps
# ============================================================================


# ---------------------------------------------------------------------------
# Gripper geometry (Schunk WSG-50, in gripper frame G). Units: meters.
#
# `CROP_MIN`/`CROP_MAX` describe the closing volume swept between the two
# fingers -- a cloud point inside this box will end up between the fingers when
# the hand closes. These constants are taken verbatim from the reference
# notebook.
# ---------------------------------------------------------------------------
CROP_MIN = np.array([-0.05, 0.01, -0.00625])
CROP_MAX = np.array([0.05, 0.1125, 0.00625])

# Offset, expressed in the gripper frame, from the gripper origin G to the
# sample point S that we align the grasp against (also from the reference).
P_GS_G = np.array([0.054 - 0.01, 0.10625/2, 0.0])

# Roll angles (about the surface normal) tried for each sample, ordered from
# the center outward, exactly as in the reference notebook.
MIN_ROLL = -np.pi / 3.0
MAX_ROLL = np.pi / 3.0
ROLL_ALPHA = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
# NOTE: this evaluates to MAX_ROLL * (ROLL_ALPHA - 1), i.e.
#   [-30, -21, -39, -12, -48, 0, -60] degrees
# -- every roll is <= 0, so the positive half of [MIN_ROLL, MAX_ROLL] is never
# explored and the neutral 0 roll is tried 6th of 7. Given the comment above,
# the intent was probably MIN_ROLL + ROLL_ALPHA * (MAX_ROLL - MIN_ROLL), which
# maps alpha=0.5 to 0. Left as-is because changing it changes which grasps the
# production sampler returns; `grasp_step` shows the per-roll outcomes so the
# effect can be judged before touching it.
ROLL_ANGLES = MIN_ROLL + ROLL_ALPHA * MAX_ROLL

# Analytic collision model of the gripper, as a set of axis-aligned boxes in
# the gripper frame G, each given as (min_corner, max_corner). These roughly
# bound the two fingers and the palm/backplate of the WSG-50 so that we can
# reject grasps whose solid parts would intersect the observed point cloud
# (Drake did this with the true mesh + signed-distance queries).
#
# The fingers sit just outside the closing region along +/-x and extend from
# the palm (small y) out to the fingertips (past the crop box in y). The palm
# spans the gap behind the fingers.
# x: span between fingertips
# y: depth from palm to fingertips
_FINGER_HALF_THICKNESS = 0.012  # half-width of a finger along x
_FINGER_HALF_DEPTH = 0.0225  # half-width of a finger along z
_FINGER_Y_MIN = 0.02  # palm side of the fingers
_FINGER_Y_MAX = 0.092  # fingertips
_FINGER_X_INNER = 0.05  # inner face of finger == edge of closing region
_PALM_HALF_WIDTH = 0.1  # palm extent along x
_PALM_HALF_DEPTH = 0.021  # palm extent along z
_PALM_Y_MIN = 0.0
_PALM_Y_MAX = 0.02

APPROACH_DIST = 0.1
GRIPPER_COLLISION_THRESHOLD = 0.0
MARGINS = [0.005, 0.01, 0.025]

# Flat retraction applied to every published grasp, along the gripper's own
# approach axis (local +y). Stacks on top of the palm backoff chosen in
# grasp_candidate_cost. Set to 0.0 to publish the sampler's pose verbatim.
GRASP_PULLBACK = 0.025  # meters

BLEND_MIN_DIST = 0.45 # meters (below this we should grasp vertically)

GRIPPER_COLLISION_BOXES = [
    # left finger (-x side)
    (
        np.array([-_FINGER_X_INNER - 2 * _FINGER_HALF_THICKNESS, _FINGER_Y_MIN, -_FINGER_HALF_DEPTH]),
        np.array([-_FINGER_X_INNER, _FINGER_Y_MAX, _FINGER_HALF_DEPTH]),
    ),
    # right finger (+x side)
    (
        np.array([_FINGER_X_INNER, _FINGER_Y_MIN, -_FINGER_HALF_DEPTH]),
        np.array([_FINGER_X_INNER + 2 * _FINGER_HALF_THICKNESS, _FINGER_Y_MAX, _FINGER_HALF_DEPTH]),
    ),
    # palm / backplate spanning the gap behind the fingers
    (
        np.array([-_PALM_HALF_WIDTH, _PALM_Y_MIN, -_PALM_HALF_DEPTH]),
        np.array([_PALM_HALF_WIDTH, _PALM_Y_MAX, _PALM_HALF_DEPTH]),
    ),
]


# ---------------------------------------------------------------------------
# Small rigid-transform helpers (4x4 homogeneous matrices), so we don't need
# Drake's RigidTransform / RotationMatrix.
# ---------------------------------------------------------------------------
def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector."""
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p
    return X


def invert_transform(X: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform."""
    R = X[:3, :3]
    p = X[:3, 3]
    Xinv = np.eye(4)
    Xinv[:3, :3] = R.T
    Xinv[:3, 3] = -R.T @ p
    return Xinv


def transform_points(X: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply X to an (N, 3) array of points, returning (N, 3)."""
    return pts @ X[:3, :3].T + X[:3, 3]


def rotation_x(theta: float) -> np.ndarray:
    """Rotation matrix about the x axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def pull_back(X_G: np.ndarray, dist: float = GRASP_PULLBACK) -> np.ndarray:
    """Retract a gripper pose along its own approach axis, returning a new matrix.

    +y is the approach direction (see antipodal_grasp_frame), so subtracting moves
    the hand away from the object -- same sign as the palm backoff in
    grasp_candidate_cost and the pre-grasp pose it collision-checks.
    """
    X = X_G.copy()
    X[:3, 3] -= dist * X_G[:3, 1]
    return X


def preferred_approach(p_W: np.ndarray):
    """Preferred gripper approach direction at a base-frame position.

    Straight down close to the base, radially outward far from it, blended over
    BLEND_MIN_DIST -- so the hand reaches over nearby objects and in at distant
    ones. Returns (pref_approach, u_radial), both unit; accepts a (3,) point or
    an (N, 3) array and matches that shape.

    A point exactly on the base z axis has no radial direction and yields nan
    (the pure-down term is poisoned by 0.0 * nan); callers working over a whole
    cloud should drop non-finite rows.
    """
    p = np.asarray(p_W, dtype=float)

    if p.ndim == 1:
        # Kept as an explicit scalar path rather than falling through the
        # vectorized one: numpy's axis-wise norm reassociates the sum, which
        # perturbs the cost in the 12th decimal and would make this refactor
        # measurably change the sampler's output.
        r_xy = np.array([p[0], p[1], 0.0])
        d = np.linalg.norm(r_xy)
        u_radial = r_xy / d
        w_radial = np.clip((d - BLEND_MIN_DIST) / BLEND_MIN_DIST, 0.0, 1.0)
        w_down = 1.0 - w_radial
        pref = w_down * np.array([0.0, 0.0, -1.0]) + w_radial * u_radial
        pref /= np.linalg.norm(pref)
        return pref, u_radial

    r_xy = np.stack([p[:, 0], p[:, 1], np.zeros(len(p))], axis=1)
    d = np.linalg.norm(r_xy, axis=1, keepdims=True)
    u_radial = r_xy / d
    w_radial = np.clip((d - BLEND_MIN_DIST) / BLEND_MIN_DIST, 0.0, 1.0)
    w_down = 1.0 - w_radial
    pref = w_down * np.array([0.0, 0.0, -1.0]) + w_radial * u_radial
    pref /= np.linalg.norm(pref, axis=1, keepdims=True)
    return pref, u_radial


# Why a candidate was rejected. grasp_candidate_cost and
# generate_antipodal_grasp_candidate return one of these as their third element
# in place of the cost breakdown, so a caller can report the reason instead of
# just dropping the candidate.
REJECT_NO_POINTS = "no points between the fingers"
REJECT_CLOUD_COLLISION = "gripper collides with the point cloud"
REJECT_GROUND = "gripper collides with the ground"
REJECT_NORMAL_PARALLEL = "normal parallel to preferred approach"
REJECT_NO_ROLL = "no feasible roll"


def is_collision_free(
    X_G: np.ndarray,
    points: np.ndarray,
    tol: float = -0.008,
) -> bool:
    """Returns True if the gripper pose X_G has zero collisions with the cloud or hull."""
    X_GW = invert_transform(X_G)
    p_GC = transform_points(X_GW, points)

    # 1. Surface Point Cloud Check
    for lo, hi in GRIPPER_COLLISION_BOXES:
        if np.any(np.all((p_GC >= lo) & (p_GC <= hi), axis=1)):
            return False

    return True

# ---------------------------------------------------------------------------
# Grasp cost (port of GraspCandidateCost)
# ---------------------------------------------------------------------------
def grasp_candidate_cost(
    X_G: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    margin: float,
    adjust_X_G: bool = False,
    verbose: bool = not False,
):
    """Score a grasp pose against the point cloud.

    Lower is better. Returns (cost, X_G, info) where X_G may have been
    recentered along the finger-closing axis when `adjust_X_G` is True (mirrors
    the reference behavior). An infinite cost means the candidate is infeasible
    (nothing to grasp between the fingers, or a collision), and `info` is then
    the REJECT_* string saying which; otherwise `info` is the cost breakdown.
    """
    X_GW = invert_transform(X_G)
    p_GC = transform_points(X_GW, points)  # cloud in gripper frame, (N, 3)

    # Points that fall inside the closing region between the fingers.
    inside = np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)
    



    # Previously centered grasped objects. This is a partially observed adaptation of this to maximize the amount of space for unobserved points by moving the observed face as near to the gripper as possible
    if adjust_X_G:
        p_GC_in = p_GC[inside]
        n_GC_x = (normals[inside] @ X_GW[:3, :3].T)[:, 0]
        outward_is_plus_x = n_GC_x.mean() >= 0.0 #this check is because of the gripper rotation earlier for comfort but in the gripper coordinate frame is the normals pointing towards the camera or away

        #NEAR_FINGER_MARGIN = 0.005#0.025  # frontmost point sits just inside the finger
        if outward_is_plus_x:
            shift_x = p_GC_in[:, 0].max() - (CROP_MAX[0] - margin) # we want the nearest point to the camera to be sitting at 0.05 - 0.01 within the gripper
        else:
            shift_x = p_GC_in[:, 0].min() - (CROP_MIN[0] + margin) # we want the nearest point to the camera to be sitting at 0.05 - 0.01 within the gripper (min is nearest point in gripper frame because of rotation)

        shift_G = np.array([shift_x, 0.0, 0.0])
        new_p = X_G[:3, :3] @ shift_G + X_G[:3, 3]
        X_G = X_G.copy()
        X_G[:3, 3] = new_p
        X_GW = invert_transform(X_G)
        p_GC = transform_points(X_GW, points)

        BACKPLATE_MARGIN = 0.005
        MAX_BACKOFF = APPROACH_DIST  # never retract more than the pre-grasp gap

        in_palm_xz = (
            (np.abs(p_GC[:, 0]) <= _PALM_HALF_WIDTH)
            & (np.abs(p_GC[:, 2]) <= _PALM_HALF_DEPTH)
            & (p_GC[:, 1] < _PALM_Y_MAX + BACKPLATE_MARGIN)
            & (p_GC[:, 1] > _PALM_Y_MIN - 0.05)  # ignore points well behind the palm
        )
        if np.any(in_palm_xz):
            deepest_y = p_GC[in_palm_xz, 1].min()
            backoff = min(_PALM_Y_MAX + BACKPLATE_MARGIN - deepest_y, MAX_BACKOFF)
            if backoff > 0.0:
                X_G = X_G.copy()
                X_G[:3, 3] = X_G[:3, 3] - backoff * X_G[:3, 1]
                X_GW = invert_transform(X_G)
                p_GC = transform_points(X_GW, points)

        inside = np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)

    n_inside = int(np.sum(inside))
    if n_inside == 0:
        if verbose:
            print(f"cost: inf  ({REJECT_NO_POINTS})")
        return np.inf, X_G, REJECT_NO_POINTS

    # Collision check: any cloud point inside a solid part of the gripper makes
    # this grasp infeasible (Drake used signed-distance to the WSG mesh).
    for lo, hi in GRIPPER_COLLISION_BOXES:
        if np.any(np.all((p_GC >= lo - GRIPPER_COLLISION_THRESHOLD) & (p_GC <= hi + GRIPPER_COLLISION_THRESHOLD), axis=1)):
            if verbose:
                print(f"cost: inf  ({REJECT_CLOUD_COLLISION})")
            return np.inf, X_G, REJECT_CLOUD_COLLISION


    # Ground plane collision check: any solid part of the gripper goes below
    # the lowest part of the object (with some tolerance)
    cost_breakdown = {"debug": []}
    # check for collision with pre grasp position as well.
    X_pre = X_G.copy()
    X_pre[:3, 3] -= APPROACH_DIST * X_G[:3, 1]

    tol = 0.0001 # TODO: adjust this
    p_minz_W = points[np.argmin(points[:,2]), 2] - tol
    for X_check in [X_G, X_pre]:
        for lo, hi in GRIPPER_COLLISION_BOXES:
            lo_W = transform_points(X_check, lo)
            hi_W = transform_points(X_check, hi)
            cost_breakdown["debug"].append({"lo_W[2]": lo_W[2], "hi_W[2]": hi_W[2], "p_minz_W": p_minz_W})
            if lo_W[2] < p_minz_W or hi_W[2] < p_minz_W:
                if verbose:
                    print(f"cost: inf  ({REJECT_GROUND})")
                return np.inf, X_G, REJECT_GROUND

    # Normals of the grasped points, expressed in the gripper frame.
    n_GC = normals[inside] @ X_GW[:3, :3].T  # (n_inside, 3)

    # # Penalize deviation of the gripper from a straight-down approach.
    # # In the reference this is 20 * R_G[2, 1]: the world-z component of the
    # # gripper's +y (approach) axis. For a top-down grasp the approach axis
    # # points toward world -z, giving R_G[2,1] = -1 and a reward of -20.




    # Note this is evaluated at the gripper origin, and after adjust_X_G has
    # shifted it -- not at the seed point generate_antipodal_grasp_candidate
    # used to build the frame.
    pref_approach, _ = preferred_approach(X_G[:3, 3])

    u_radial = pref_approach.copy()
    u_radial[2] = 0
    comfort = float(np.dot(X_G[:3, 1], u_radial))
    print(f"COMFORT: {comfort}")
    if comfort < 0.0:
        return np.inf, X_G, None



    # PREVENT COMING FROM BEHIND GRASPS
    alignment = float(np.dot(X_G[:3, 1], pref_approach))
    error = 1.0 - alignment
    cost = 1000.0 * error 
    cost_breakdown["comfort"] = cost

    # Reward antipodal alignment: surface normals aligned with the finger
    # closing axis (gripper x). Sum of squared dot products.
    cost -= 15.0 * float(np.mean(n_GC[:, 0] ** 2)) # used to be sum trying to normalize it.
    cost_breakdown["surface normals"] = -15.0 * float(np.mean(n_GC[:, 0] ** 2))


    cost -= 1000.0 * np.abs(max(MARGINS) - margin)

    if verbose:
        print(f"cost: {cost:.4f}  (points grasped: {n_inside})")
    return cost, X_G, cost_breakdown


# ---------------------------------------------------------------------------
# Candidate generation (port of GenerateAntipodalGraspCandidate)
# ---------------------------------------------------------------------------
_DEBUG_LAST_SEED, DEBUG_SEEDS = None, []  # DEBUG-VIZ: seed cloud index, then ranked

def antipodal_grasp_frame(points, normals, index):
    """Build the un-rolled gripper frame seeded at one cloud point.

    The gripper convention: +x is the finger-closing axis and takes the surface
    normal, +y is the approach direction and takes the preferred approach
    projected orthogonal to it, +z is the wrist/roll reference.

    Returns (p_WS, R_WG, pref_approach), or None when the normal is
    (anti)parallel to the preferred approach and the frame is ill-defined.
    """
    p_WS = points[index]
    n_WS = normals[index]

    norm = np.linalg.norm(n_WS)
    if not np.isclose(norm, 1.0):
        n_WS = n_WS / norm

    # Gripper x axis aligns with the surface normal (the finger-closing axis
    # presses along the normal for an antipodal grasp).
    Gx = n_WS
    # Make an orthonormal y axis aligned as much as possible with world down.
    pref_approach, u_radial = preferred_approach(p_WS)


    if np.abs(np.dot(pref_approach, Gx)) > 1.0 - 1e-6:
        # Normal points straight up/down; a from-above grasp is ill-defined.
        return None
    Gy = pref_approach - np.dot(pref_approach, Gx) * Gx
    Gy /= np.linalg.norm(Gy)
    Gz = np.cross(Gx, Gy)


    v_ref = u_radial + np.array([0.0, 0.0, 1.0])
    v_ref /= np.linalg.norm(v_ref)
    # make wrist always comfortable for robot
    if np.dot(Gz, v_ref) < 0.0:
        Gx = -Gx  
        Gz = -Gz  
    
    R_WG = np.column_stack((Gx, Gy, Gz))
    return p_WS, R_WG, pref_approach


def generate_antipodal_grasp_candidate(
    points,
    normals,
    index,
    roll_sink: list | None = None,
    stop_at_first: bool = True,
    verbose: bool = not False,
):
    """Try to align a top-down antipodal grasp at the given surface point.

    Returns (cost, X_G, info). cost is inf if no feasible roll was found, and
    info is then a REJECT_* string; X_G is None only when the seed frame itself
    is ill-defined.

    The defaults reproduce the sampler's behavior: stop at the first feasible
    roll and discard the rest. `grasp_step` passes stop_at_first=False and a
    `roll_sink` list to collect (theta, cost, X_G, info) for every roll tried,
    which is what lets it explain a rejection instead of dropping it.
    """

    global _DEBUG_LAST_SEED; _DEBUG_LAST_SEED = int(index)  # DEBUG-VIZ

    frame = antipodal_grasp_frame(points, normals, index)
    if frame is None:
        return np.inf, None, REJECT_NORMAL_PARALLEL
    p_WS, R_WG, _ = frame

    # Try rolls about the normal, from the center out; keep the first feasible.
    first_ok = None
    fallback_X_G = None


    for theta in ROLL_ANGLES:
        for margin in MARGINS:
            R_WG2 = R_WG @ rotation_x(theta)
            # Place G so that the sample point S lands at P_GS_G in the gripper.
            p_WG = p_WS - R_WG2 @ P_GS_G
            X_G = make_transform(R_WG2, p_WG)
            cost, X_G, info = grasp_candidate_cost(
                X_G, points, normals, margin, adjust_X_G=True, verbose=verbose
            )
            if roll_sink is not None:
                roll_sink.append((float(theta), cost, X_G, info))
            if fallback_X_G is None:
                # The first roll tried, and the one the sampler would have taken:
                # the most representative pose to show when every roll fails.
                fallback_X_G = X_G
            if np.isfinite(cost) and first_ok is None:
                first_ok = (cost, X_G, info)
                if stop_at_first:
                    return first_ok

    if first_ok is not None:
        return first_ok
    return np.inf, fallback_X_G, REJECT_NO_ROLL


def sample_grasps(points, normals, num_candidates=200, seed=None):
    """Sample many candidates and return them sorted by cost (best first)."""
    rng = np.random.default_rng(seed)
    costs, poses, breakdowns = [], [], []
    seeds = []  # DEBUG-VIZ

    # Sample seed points without replacement: each cloud point is tried at most
    # once. If the cloud is smaller than num_candidates we simply try them all.
    n_points = points.shape[0]
    n_draw = min(num_candidates, n_points)
    indices = rng.choice(n_points, size=n_draw, replace=False)

    for index in indices:
        cost, X_G, cost_breakdown = generate_antipodal_grasp_candidate(points, normals, index)
        if np.isfinite(cost):
            costs.append(cost)
            poses.append(X_G)
            breakdowns.append(cost_breakdown)
            seeds.append(_DEBUG_LAST_SEED)  # DEBUG-VIZ
    order = np.argsort(costs)
    DEBUG_SEEDS[:] = [seeds[i] for i in order]  # DEBUG-VIZ
    return [(costs[i], poses[i]) for i in order]



# ----------------------------------------------------------------
# ROS utils

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
    return make_transform(R, p)


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


def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Read finite XYZ points from a ROS PointCloud2 message."""
    raw_points = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )

    # Newer ROS 2 releases return a structured NumPy array.
    if isinstance(raw_points, np.ndarray) and raw_points.dtype.names is not None:
        xyz = np.stack(
            (
                np.asarray(raw_points["x"]).reshape(-1),
                np.asarray(raw_points["y"]).reshape(-1),
                np.asarray(raw_points["z"]).reshape(-1),
            ),
            axis=1,
        ).astype(np.float64)
    else:
        # Compatible with ROS 2 releases where read_points returns an iterator.
        xyz = np.asarray(list(raw_points), dtype=np.float64)

        if xyz.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        xyz = xyz.reshape((-1, 3))

    return xyz[np.all(np.isfinite(xyz), axis=1)]


def preprocess_xyz_cloud(
    points_B: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    """Downsample a base-frame cloud and estimate outward-facing normals."""
    if points_B.shape[0] < 4:
        raise ValueError("Point cloud has fewer than four valid points.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_B)

    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    points = np.asarray(pcd.points)

    if points.shape[0] < 4:
        raise ValueError(
            "Too few points remain after voxel downsampling; reduce voxel_size."
        )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.05,
            max_nn=min(30, points.shape[0]),
        )
    )

    # Match the normal-orientation behavior in the original script.
    pcd.orient_normals_consistent_tangent_plane(
        k=min(30, points.shape[0] - 1)
    )

    normals = np.asarray(pcd.normals)
    centroid = points.mean(axis=0)

    outward = points - centroid
    needs_flip = np.sum(normals * outward, axis=1) < 0.0
    normals[needs_flip] *= -1.0

    pcd.normals = o3d.utility.Vector3dVector(normals)

    return points, normals, pcd


class GraspSelection(Node):
    def __init__(self, node_name: str = "grasp_selection"):
        # Parameterized so GraspStepDebug can register under its own name:
        # two nodes called "grasp_selection" would both offer /clear_grasps and
        # both broadcast grasp_candidate_00, and RViz would show the mesh
        # flickering between the two poses.
        super().__init__(node_name)

        # --------------------------------------------------------------------
        # Parameters
        # --------------------------------------------------------------------
        self.declare_parameter("pcd_topic", "object_pcd") # from file uses /pointcloud
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("candidates", 200)
        self.declare_parameter("top_k", 5)
        self.declare_parameter("voxel_size", 0.00)
        self.declare_parameter("seed", 0)
        self.declare_parameter("namespace", "nero_right")
        self.declare_parameter("process_once", False) # default to resample grasps with every point cloud
        self.declare_parameter("publish_period", 0.25)
        self.declare_parameter("publish_gripper_tfs", True)
        self.declare_parameter("execute_number", 0) # which rank grasp to publish for execution (set to -1 for none)

        # Optional: publishes each candidate as PoseStamped too.
        # TF itself remains published on /tf.
        self.declare_parameter("publish_pose_topics", False)

        self.pcd_topic = str(self.get_parameter("pcd_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value).lstrip("/")
        self.candidates = int(self.get_parameter("candidates").value)
        self.top_k = int(self.get_parameter("top_k").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.seed = int(self.get_parameter("seed").value)
        self.namespace = str(self.get_parameter("namespace").value).strip("/")
        self.process_once = bool(self.get_parameter("process_once").value)
        self.publish_period = max(
            0.01,
            float(self.get_parameter("publish_period").value),
        )
        self.publish_gripper_tfs = bool(self.get_parameter("publish_gripper_tfs").value)
        self.publish_pose_topics = bool(
            self.get_parameter("publish_pose_topics").value
        )

        if not self.namespace:
            self.namespace = ""
        else:
            # add namespace to base_frame
            self.base_frame = f"{self.namespace}/{self.base_frame}"

        # --------------------------------------------------------------------
        # Internal state
        # --------------------------------------------------------------------
        self._lock = threading.Lock()
        self._processing_cloud = False
        self._processed_once = False

        self.points_B: np.ndarray | None = None
        self.normals_B: np.ndarray | None = None
        self.pcd: o3d.geometry.PointCloud | None = None
        self.ranked_grasps: list[tuple[float, np.ndarray]] = []
        self.grasp_published = False
        # Identifies which sampling run the current ranking came from. Consumers
        # cannot tell one batch from the next by TF alone -- the frames are
        # rebroadcast at publish_period forever and carry no batch identity -- so
        # this is what lets them retire a candidate set they have already acted on.
        self._batch_seq = 0
        self._published_seq: int | None = None

        # TF listener: looks up T_base_cloud.
        self.tf_buffer = Buffer()
        # spin_thread=True: the blocking lookup_transform calls in the timer
        # callbacks would otherwise occupy the single spin() thread that has to
        # deliver /tf_static into the buffer, so the lookup could never succeed.
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # --------------------------------------------------------------------
        # ROS interfaces
        # --------------------------------------------------------------------
        self.setup_publishers()
        self.setup_subscribers()
        self.create_service(Trigger, "clear_grasps", self._on_clear_grasps)

        self.get_logger().info(
            f"Waiting for PointCloud2 messages on '{self.pcd_topic}'."
        )

    def _on_clear_grasps(self, request, response):
        """Empty the current ranking. Does NOT retire the candidates."""
        # The TF frames do not age out: process_once defaults to False and seed to
        # 0, so the next cloud -- within a second, offline -- refills the ranking
        # with bit-identical grasps, and publish_static_gripper_tfs re-stamps the
        # noN/gripper_flange leg unconditionally either way. What actually retires a
        # candidate set is the batch sequence on /grasp_batch: a consumer records
        # the sequence it acted on and ignores that batch thereafter.
        with self._lock:
            self.ranked_grasps = []
        response.success = True
        response.message = "cleared"
        return response

    def setup_publishers(self):
        """Create the TF broadcaster and optional per-grasp PoseStamped topics."""
        self.tf_broadcaster = TransformBroadcaster(self)

        # Global rather than node-relative: the consumer (pose_commander) runs
        # under an arm namespace, the same reason clear_grasps and object_pcd are
        # addressed globally. Latched so a consumer starting late still learns
        # which batch the frames it can already see belong to.
        self.batch_publisher = self.create_publisher(
            UInt32,
            "/grasp_batch",
            QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1),
        )

        self.debug_axes_pub = self.create_publisher(MarkerArray, "/grasp_debug_axes", 1)  # DEBUG-VIZ

        self.pose_publishers = []
        if self.publish_pose_topics:
            for rank in range(self.top_k):
                topic = f"/grasp_candidate_{rank:02d}/pose"
                publisher = self.create_publisher(PoseStamped, topic, 10)
                self.pose_publishers.append(publisher)

        self.publish_timer = self.create_timer(
            self.publish_period,
            self.publish_grasps,
        )

        if self.publish_gripper_tfs:
            self.publisher_gripper_timer = self.create_timer(
                self.publish_period,
                self.publish_static_gripper_tfs
            )

            self.pub_move = self.create_publisher(PoseStamped, "nero_right/control/move_p", 1)
        
        self.pub_grasp_success = self.create_publisher(Bool, '/grasp_success', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        # self.create_publisher()

    def setup_subscribers(self):
        """Subscribe to the sensor point-cloud stream."""
        self.pointcloud_subscriber = self.create_subscription(
            PointCloud2,
            self.pcd_topic,
            self.pointcloud_callback,
            1#qos_profile_sensor_data,
        )

    def pointcloud_callback(self, msg: PointCloud2):
        """Process the first usable point cloud, or every cloud if requested."""
        self.get_logger().info("Received point cloud.")
        with self._lock:
            if self._processing_cloud:
                return

            if self.process_once and self._processed_once:
                return

            self._processing_cloud = True

        success = False

        try:
            points_B, normals_B, pcd = self.load_point_cloud(msg)

            with self._lock:
                self.points_B = points_B
                self.normals_B = normals_B
                self.pcd = pcd

            self.sample_grasps()

            success = True
            self.grasp_published = False

        except TransformException as exc:
            # This often occurs briefly at startup, before TF is populated.
            self.get_logger().error(
                f"Cannot transform cloud from '{msg.header.frame_id}' "
                f"to '{self.base_frame}' yet: {exc}"
            )

        except Exception as exc:
            self.get_logger().error(f"Could not process point cloud: {exc}")
            self.pub_grasp_success.publish(Bool(data=False))

        finally:
            with self._lock:
                self._processing_cloud = False
                if success:
                    self._processed_once = True

    def load_point_cloud(
        self,
        msg: PointCloud2,
    ) -> tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
        """
        Read PointCloud2, transform its points into base_link, downsample,
        and estimate normals.

        Returned pose convention:
            points_B: points expressed in base_frame.
            Therefore each grasp X_G returned by sample_grasps is T_base_gripper.
        """
        cloud_frame = msg.header.frame_id.lstrip("/")

        if not cloud_frame:
            raise ValueError("Incoming PointCloud2 has an empty header.frame_id.")

        points_C = pointcloud2_to_xyz(msg)
        if points_C.shape[0] == 0:
            raise ValueError("Incoming PointCloud2 has no finite XYZ points.")

        # lookup_transform(target, source, time) returns T_target_source.
        if cloud_frame == self.base_frame:
            X_BC = np.eye(4)
        else:
            cloud_stamp = Time.from_msg(msg.header.stamp)

            transform_BC = self.tf_buffer.lookup_transform(
                self.base_frame,
                cloud_frame,
                cloud_stamp,
            )
            X_BC = transform_msg_to_matrix(transform_BC)

        # Transform raw cloud points before normal estimation, so all subsequent
        # grasp computation is consistently expressed in base_link.
        points_B_raw = transform_points(X_BC, points_C)

        points_B, normals_B, pcd = preprocess_xyz_cloud(
            points_B_raw,
            voxel_size=self.voxel_size,
        )

        self.get_logger().info(
            f"Loaded cloud: {points_C.shape[0]} raw points, "
            f"{points_B.shape[0]} points after downsampling."
        )

        return points_B, normals_B, pcd

    def sample_grasps(self):
        """Run the original antipodal sampler on the most recent base-frame cloud."""
        with self._lock:
            if self.points_B is None or self.normals_B is None:
                raise RuntimeError("Cannot sample grasps before loading a cloud.")

            points_B = self.points_B.copy()
            normals_B = self.normals_B.copy()

        ranked = sample_grasps(
            points_B,
            normals_B,
            num_candidates=self.candidates,
            seed=self.seed,
        )

        ranked = ranked[: self.top_k]

        with self._lock:
            self.ranked_grasps = ranked
            self._debug_seeds = DEBUG_SEEDS[: self.top_k]  # DEBUG-VIZ
            self._batch_seq += 1

        if ranked:
            self.get_logger().info(
                f"Found {len(ranked)} feasible grasp candidates."
            )

            for rank, (cost, X_BG) in enumerate(ranked):
                p = X_BG[:3, 3]
                self.get_logger().info(
                    f"  #{rank:02d}: cost={cost:.4f}, "
                    f"position=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"
                )
        else:
            self.get_logger().warn(
                "No feasible grasps found. Try more candidates or a smaller voxel size."
            )

    def grasp_frame_name(self, rank: int) -> str:
        """Return a valid child frame name for one ranked candidate."""
        # No leading slash: ROS 2 tf2 rejects frame ids that start with one, so
        # a "/grasp_candidate_NN" parent leaves this whole subtree -- and the
        # noN/gripper_flange frames hanging off it -- orphaned from the planning
        # frame, with nothing but a "not part of the same tree" warning to say so.
        return f"grasp_candidate_{rank:02d}"

    def pose_from_matrix(
        self,
        X_BG: np.ndarray,
        stamp,
    ) -> PoseStamped:
        """Build a PoseStamped equivalent of T_base_gripper."""
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.base_frame

        qx, qy, qz, qw = rotation_matrix_to_quaternion(X_BG[:3, :3])

        pose.pose.position.x = float(X_BG[0, 3])
        pose.pose.position.y = float(X_BG[1, 3])
        pose.pose.position.z = float(X_BG[2, 3])

        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)

        return pose

    def publish_grasps(self):
        """
        Publish the current top-k grasps as TF frames:

          base_link
            ├── camera_r/grasp_candidate_00
            ├── camera_r/grasp_candidate_01
            └── ...

        All TF transforms are sent through the normal /tf transport.
        """
        self.publish_debug_axes(self.get_clock().now().to_msg())  # DEBUG-VIZ

        with self._lock:
            # Pulled back here, at the one place grasps leave the node, so the TF
            # frames, the pose topics and the move_p command below all agree --
            # RViz then shows where the hand actually goes. pull_back() returns a
            # new matrix, so it also serves the .copy() this used to do.
            ranked = [
                (cost, pull_back(X_BG))
                for cost, X_BG in self.ranked_grasps
            ]
            seq = self._batch_seq

        if not ranked:
            return

        stamp = self.get_clock().now().to_msg()
        transforms = []

        for rank, (_, X_BG) in enumerate(ranked):
            transform = TransformStamped()

            transform.header.stamp = stamp
            transform.header.frame_id = self.base_frame
            transform.child_frame_id = self.grasp_frame_name(rank)

            qx, qy, qz, qw = rotation_matrix_to_quaternion(X_BG[:3, :3])

            transform.transform.translation.x = float(X_BG[0, 3])
            transform.transform.translation.y = float(X_BG[1, 3])
            transform.transform.translation.z = float(X_BG[2, 3])

            transform.transform.rotation.x = float(qx)
            transform.transform.rotation.y = float(qy)
            transform.transform.rotation.z = float(qz)
            transform.transform.rotation.w = float(qw)

            transforms.append(transform)

            # Optional ordinary ROS topics for consumers that do not use TF.
            if self.publish_pose_topics and rank < len(self.pose_publishers):
                self.pose_publishers[rank].publish(
                    self.pose_from_matrix(X_BG, stamp)
                )

        self.tf_broadcaster.sendTransform(transforms)

        # Strictly after sendTransform, and only on change. A consumer that sees
        # sequence N must already be able to resolve batch N's frames; announcing
        # it from process_cloud instead would publish the number up to one
        # publish_period before the transforms it refers to, and a consumer
        # polling in that window would plan the *previous* batch's grasp under the
        # new number -- silently, since the number is what it trusts.
        if seq != self._published_seq:
            self.batch_publisher.publish(UInt32(data=seq))
            self._published_seq = seq

        # optionally execute the pose!
        publish_number = self.get_parameter('execute_number').get_parameter_value().integer_value
        # Bounds-checked because ranked can be shorter than execute_number (an
        # IndexError here kills the timer, and with it all TF publishing), and
        # cost-checked because grasp_step puts infeasible poses in ranked to
        # visualize them -- those must never reach the arm.
        if (0 <= publish_number < len(ranked)
                and np.isfinite(ranked[publish_number][0])) and not self.grasp_published:
            # there's a rotation offset too
            X_OG = np.eye(4)
            # map x->y, y->z, z->x
            X_OG[:3,:3] = np.array([[0., 0., 1.],
                                    [1., 0., 0.],
                                    [0., 1., 0.]]).T

            # lookup transform from palm to link7
            transform_GL7 = self.tf_buffer.lookup_transform(
                f"{self.namespace}/gripper_palm",
                f"{self.namespace}/link7",
                self.get_clock().now().to_msg(),
                timeout=Duration(seconds=10.0)
            )
            X_GL7 = transform_msg_to_matrix(transform_GL7)

            X_BO = ranked[publish_number][1]
            X_BL7 = X_BO @ X_OG @ X_GL7
            self.pub_move.publish(self.pose_from_matrix(X_BO @ X_OG, stamp))
            self.grasp_published = True
        else:
            # self.pub_grasp_success.publish(Bool(data=False))
            pass

    def publish_static_gripper_tfs(self):
        stamp = self.get_clock().now().to_msg()
        transforms = []
        X_GP = np.eye(4)
        # map x->y, y->z, z->x
        X_GP[:3,:3] = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]]).T
        
        # lookup transform from palm to flange
        transform_BC = self.tf_buffer.lookup_transform(
            f"{self.namespace}/gripper_palm",
            f"{self.namespace}/gripper_flange",
            self.get_clock().now().to_msg(),
            timeout=Duration(seconds=10.0)
        )
        X_PF = transform_msg_to_matrix(transform_BC)
        X_GF = X_GP @ X_PF

        for k in range(self.top_k):
            # robot to gripper
            transform = TransformStamped()

            transform.header.stamp = stamp
            transform.header.frame_id = self.grasp_frame_name(k)
            transform.child_frame_id = f"no{k}/gripper_flange"

            transform.transform.translation.x = float(X_GF[0, 3])
            transform.transform.translation.y = float(X_GF[1, 3])
            transform.transform.translation.z = float(X_GF[2, 3])

            qx, qy, qz, qw = rotation_matrix_to_quaternion(X_GF[:3, :3])
            transform.transform.rotation.x = float(qx)
            transform.transform.rotation.y = float(qy)
            transform.transform.rotation.z = float(qz)
            transform.transform.rotation.w = float(qw)

            transforms.append(transform)

        self.tf_broadcaster.sendTransform(transforms)

    # ---- DEBUG-VIZ: delete this whole block when done ----------------------
    def publish_debug_axes(self, stamp):
        """Green = preferred approach at the object centroid, red = grasp0's actual
        g_y, cyan = cloud normals (yellow where inside grasp0's closing region)."""
        with self._lock:
            if self.points_B is None or self.normals_B is None:
                return
            points, normals = self.points_B, self.normals_B
            X_BG0 = self.ranked_grasps[0][1] if self.ranked_grasps else None
            seeds = getattr(self, "_debug_seeds", [])
            seed0 = seeds[0] if (seeds and X_BG0 is not None) else None

        centroid = points.mean(axis=0)

        # Same blend as grasp_candidate_cost, evaluated at the centroid.
        pref, _ = preferred_approach(centroid)

        def base(ns, idx, mtype, action=Marker.ADD):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.base_frame
            m.ns, m.id, m.type, m.action = ns, idx, mtype, action
            m.pose.orientation.w = 1.0  # points[] are already in the marker frame
            return m

        def pt(p):
            return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

        def arrow(idx, start, direction, rgb, length=0.15):
            m = base("axes", idx, Marker.ARROW)
            # shaft diameter, head diameter, head length
            m.scale.x, m.scale.y, m.scale.z = 0.008, 0.016, 0.02
            m.color.r, m.color.g, m.color.b = rgb
            m.color.a = 1.0
            m.points = [pt(start), pt(start + length * direction)]
            return m

        # Which points end up between the fingers of grasp0 -- same crop test as
        # grasp_candidate_cost, reusing invert_transform / transform_points.
        inside = np.zeros(len(points), dtype=bool)
        if X_BG0 is not None:
            p_GC = transform_points(invert_transform(X_BG0), points)
            inside = np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)

        stride = max(1, len(points) // 800)  # keeps big live clouds in check
        n_marker = base("normals", 0, Marker.LINE_LIST)
        n_marker.scale.x = 0.0015  # line width
        n_marker.color.a = 1.0
        for p, n, hit in zip(points[::stride], normals[::stride], inside[::stride]):
            n_marker.points += [pt(p), pt(p + 0.02 * n)]
            c = ColorRGBA(r=1.0, g=0.95, b=0.1, a=1.0) if hit else ColorRGBA(r=0.0, g=0.45, b=0.55, a=1.0)
            n_marker.colors += [c, c]

        markers = [n_marker, arrow(0, centroid, pref, (0.0, 1.0, 0.0))]
        if X_BG0 is not None:
            markers.append(arrow(1, X_BG0[:3, 3], X_BG0[:3, 1], (1.0, 0.0, 0.0)))
        else:
            # otherwise the last red arrow lingers in RViz after grasps are cleared
            markers.append(base("axes", 1, Marker.ARROW, action=Marker.DELETE))

        # The sampled point whose normal seeded grasp0: its normal became the
        # finger-closing axis, so X_BG0[:3, 0] == +/- normals[seed0].
        if seed0 is not None and seed0 < len(points):
            m = arrow(0, points[seed0], normals[seed0], (1.0, 0.0, 1.0), length=0.05)
            m.ns = "seed"
            m.scale.x, m.scale.y, m.scale.z = 0.004, 0.012, 0.015
            markers.append(m)
        else:
            markers.append(base("seed", 0, Marker.ARROW, action=Marker.DELETE))

        self.debug_axes_pub.publish(MarkerArray(markers=markers))
    # ---- end DEBUG-VIZ ----------------------------------------------------


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = GraspSelection()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        rclpy.logging.get_logger("grasp_selection").error(str(exc))

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

## TODO:
# - better grasps (trick: make the fingers shorter?)
# - no collision with ground plane