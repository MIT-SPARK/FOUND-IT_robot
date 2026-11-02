"""Interactive, one-normal-at-a-time grasp debugger.

`grasp_selection` samples ~200 random cloud points, keeps the candidates with a
finite cost and silently drops the rest, so there is no way to see *why* a point
failed. This node walks the cloud instead: one point/normal pair per Enter,
ordered by how perpendicular the normal is to the preferred approach, showing
the resulting gripper pose in RViz **even when it is rejected**.

It is a subclass rather than a mode, so the production node keeps its behavior.
Everything visual is inherited: writing a single (cost, X_G) into
`self.ranked_grasps` is enough, because `publish_grasps` only checks that the
list is non-empty -- never that the cost is finite -- so it broadcasts
grasp_candidate_00 for a rejected pose, `publish_static_gripper_tfs` hangs
no0/gripper_flange off it, and RViz's Grasp0 gripper mesh jumps there.

On top of that this node draws what the URDF cannot: wireframes of the analytic
CROP_MIN/CROP_MAX closing box and the three GRIPPER_COLLISION_BOXES, colored by
which one actually fired. Those boxes exist nowhere in the gripper mesh, and
they are what the accept/reject decision is really made on.

Run it with `ros2 run` in its own terminal -- `ros2 launch` does not give a node
a usable stdin.

See HELP below for the keys.
"""

from __future__ import annotations

import sys
import threading
import traceback

import numpy as np

import rclpy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .grasp_selection import (
    CROP_MAX,
    CROP_MIN,
    GRIPPER_COLLISION_BOXES,
    REJECT_GROUND,
    REJECT_NORMAL_PARALLEL,
    ROLL_ANGLES,
    GraspSelection,
    generate_antipodal_grasp_candidate,
    invert_transform,
    preferred_approach,
    transform_points,
)

# The 12 edges of an axis-aligned box, as index pairs into the 8 corners
# enumerated x-major (corner index bits are x*4 + y*2 + z).
_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),  # along z
    (0, 2), (1, 3), (4, 6), (5, 7),  # along y
    (0, 4), (1, 5), (2, 6), (3, 7),  # along x
)

HELP = """
    Enter  next point                  b      previous point
    r      next roll (0-6)             R      previous roll
    <int>  jump to that queue position a      next accepted point
    ?      this help                   q      stop stepping
"""

_GREY = ColorRGBA(r=0.55, g=0.55, b=0.6, a=0.9)
_RED = ColorRGBA(r=1.0, g=0.15, b=0.1, a=1.0)
_GREEN = ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0)


def box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box, ordered to match _BOX_EDGES."""
    return np.array(
        [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    )


class GraspStepDebug(GraspSelection):
    def __init__(self):
        super().__init__("grasp_step")

        # Never announce a batch. /grasp_batch is what tells pose_commander a new
        # candidate set is worth planning; _batch_seq stays 0 here (this class
        # never calls the inherited sample_grasps), and pre-setting _published_seq
        # to match stops publish_grasps' `seq != _published_seq` from firing even
        # once. Publishing seq 0 a single time would be enough for pose_commander
        # to plan a rejected debug pose.
        self._published_seq = 0

        # Freeze the cloud: pcd_publisher republishes every 10 s, and each cloud
        # would otherwise rebuild the queue and throw away your position in it.
        # _cloud_key below is the belt-and-braces version, for -p process_once:=False.
        self.process_once = True

        self.declare_parameter("step_min_dist", 0.02)
        self.declare_parameter("step_max_points", 300)
        self.declare_parameter("step_verbose_cost", False)

        self._debug_seeds: list[int] = []  # the parent only getattr-guards this
        self._step_order: list[int] = []   # cloud indices, most-perpendicular first
        self._step_perp: np.ndarray | None = None
        self._step_i = 0
        self._step_rolls: list[tuple] = []
        self._step_roll_i: int | None = None  # None = auto-pick the shown roll
        self._step_view: dict | None = None   # everything the markers need
        self._cloud_key: tuple | None = None

        self.step_info_pub = self.create_publisher(MarkerArray, "/grasp_step_info", 1)

    # ------------------------------------------------------------------ setup
    def setup_publishers(self):
        # Must run before the parent sizes pose_publishers and the timers off
        # top_k. Only Grasp0 exists as far as this node is concerned, so a
        # larger top_k would just make publish_static_gripper_tfs broadcast
        # no1..noN/gripper_flange off grasp_candidate frames nothing publishes.
        self.top_k = 1
        super().setup_publishers()

    # ------------------------------------------------------------------- queue
    def sample_grasps(self):
        """Build the step queue instead of sampling. Called from the cloud callback."""
        with self._lock:
            points = self.points_B.copy()
            normals = self.normals_B.copy()

        key = (points.shape[0], float(points[:, 0].sum()))
        if key == self._cloud_key:
            return  # same cloud again: keep the queue and the current position
        self._cloud_key = key

        min_dist = float(self.get_parameter("step_min_dist").value)
        max_points = int(self.get_parameter("step_max_points").value)

        pref, _ = preferred_approach(points)
        perp = np.abs(np.einsum("ij,ij->i", normals, pref))
        finite = np.isfinite(perp)
        order = np.argsort(np.where(finite, perp, np.inf))

        # Ascending |n . pref| is ascending comfort cost: with Gx = n and
        # Gy = normalize(pref - (pref.n)n), Gy.pref = sqrt(1 - (n.pref)^2), and
        # the roll about Gx leaves Gz.pref = 0, so the comfort term
        # 18*(1 - cos(theta)*sqrt(1 - (n.pref)^2)) grows with |n . pref|.
        #
        # Sorting alone puts hundreds of near-coincident surface points at the
        # front -- they share a normal, so they share a perp -- and you would
        # press Enter forty times to see the same grasp. Keep the best-perp
        # point in each neighborhood instead. Note this thins the *seed queue*
        # only: the cost function still scores against the full cloud, unlike
        # -p voxel_size:=..., which would change the very collision and normal
        # tests being debugged.
        kept: list[int] = []
        for i in order:
            if not finite[i]:
                break
            if kept and np.min(np.linalg.norm(points[kept] - points[i], axis=1)) < min_dist:
                continue
            kept.append(int(i))
            if len(kept) >= max_points:
                break

        self._step_order = kept
        self._step_perp = perp
        self._step_i = 0
        self._step_roll_i = None

        self.get_logger().info(
            f"Step queue: {len(kept)} seeds from {len(points)} points "
            f"(>= {min_dist * 1000:.0f} mm apart), most perpendicular first."
        )
        print(self.show(0), flush=True)

    # ------------------------------------------------------------------- steps
    def show(self, i: int) -> str:
        """Evaluate queue position i, push it to RViz, and return a summary line."""
        if not self._step_order:
            return "no point cloud yet"

        self._step_i = i = int(np.clip(i, 0, len(self._step_order) - 1))
        idx = self._step_order[i]

        with self._lock:
            points = self.points_B
            normals = self.normals_B

        self._step_rolls = []
        generate_antipodal_grasp_candidate(
            points,
            normals,
            idx,
            roll_sink=self._step_rolls,
            stop_at_first=False,
            verbose=bool(self.get_parameter("step_verbose_cost").value),
        )

        if not self._step_rolls:
            # antipodal_grasp_frame bailed: no pose exists to show.
            theta, cost, X_G, info = float("nan"), np.inf, None, REJECT_NORMAL_PARALLEL
        elif self._step_roll_i is None:
            # The roll the sampler would have taken, else the first one tried.
            feasible = [r for r in self._step_rolls if np.isfinite(r[1])]
            theta, cost, X_G, info = feasible[0] if feasible else self._step_rolls[0]
        else:
            theta, cost, X_G, info = self._step_rolls[self._step_roll_i % len(self._step_rolls)]

        view = self._analyze(idx, X_G, theta, cost, info, points, normals)

        with self._lock:
            self.ranked_grasps = [(cost, X_G)] if X_G is not None else []
            self._debug_seeds = [idx]
            self._step_view = view
            # _batch_seq deliberately not bumped -- see __init__.

        return self._summary(view)

    def _analyze(self, idx, X_G, theta, cost, info, points, normals) -> dict:
        """Recompute which analytic box actually fired, for the wireframes."""
        pref, _ = preferred_approach(points[idx])
        perp = abs(float(normals[idx] @ pref))

        view = {
            "i": self._step_i,
            "idx": idx,
            "seed": points[idx],
            "normal": normals[idx],
            "pref": pref,
            "perp": perp,
            "theta": theta,
            "cost": cost,
            "info": info,
            "X_G": X_G,
            "n_inside": 0,
            "box_hits": [False] * len(GRIPPER_COLLISION_BOXES),
            "ground_hits": [False] * len(GRIPPER_COLLISION_BOXES),
            "ground_z": float(points[:, 2].min()) - 0.01,
            "rolls": list(self._step_rolls),
        }
        if X_G is None:
            return view

        p_GC = transform_points(invert_transform(X_G), points)
        view["n_inside"] = int(np.sum(np.all((p_GC >= CROP_MIN) & (p_GC <= CROP_MAX), axis=1)))

        # grasp_candidate_cost returns on the first offending box; report them
        # all, and reproduce its lo/hi-corner-only ground test exactly so the
        # colors match the decision that was actually made.
        for b, (lo, hi) in enumerate(GRIPPER_COLLISION_BOXES):
            view["box_hits"][b] = bool(np.any(np.all((p_GC >= lo) & (p_GC <= hi), axis=1)))
            lo_W = transform_points(X_G, lo)
            hi_W = transform_points(X_G, hi)
            view["ground_hits"][b] = bool(
                lo_W[2] < view["ground_z"] or hi_W[2] < view["ground_z"]
            )
        return view

    def _summary(self, view) -> str:
        off = np.degrees(np.arcsin(min(view["perp"], 1.0)))
        head = (
            f"[{view['i'] + 1:4d}/{len(self._step_order):4d}] idx={view['idx']:6d}  "
            f"perp={view['perp']:.4f} (off perpendicular by {off:5.1f} deg)  "
            f"grasped={view['n_inside']:4d}"
        )
        if np.isfinite(view["cost"]):
            b = view["info"] if isinstance(view["info"], dict) else {}
            # The breakdown also carries a "debug" list of per-box ground-check
            # dicts, which has no scalar format; only the cost terms belong here.
            terms = "  ".join(
                f"{k}={v:+.2f}"
                for k, v in b.items()
                if isinstance(v, (int, float, np.integer, np.floating))
            )
            body = f"ACCEPT  cost={view['cost']:+8.3f}   {terms}"
        else:
            body = f"REJECT  {view['info']}"

        lines = [f"{head}\n  {body}"]
        if self._step_roll_i is not None:
            lines.append(f"  showing roll {self._step_roll_i % max(len(view['rolls']), 1)}")
        for r, (th, c, _, ri) in enumerate(view["rolls"]):
            mark = ">" if th == view["theta"] else " "
            outcome = f"cost={c:+8.3f}" if np.isfinite(c) else f"{ri}"
            lines.append(f"  {mark} roll {r} {np.degrees(th):+6.1f} deg  {outcome}")
        return "\n".join(lines)

    # ----------------------------------------------------------------- markers
    def publish_debug_axes(self, stamp):
        # The inherited markers (cloud normals, yellow inside grasp0's closing
        # box, the centroid pref arrow, grasp0's approach axis) on
        # /grasp_debug_axes, then this node's own on /grasp_step_info.
        super().publish_debug_axes(stamp)

        view = self._step_view
        if view is None:
            return

        def base(ns, mtype, action=Marker.ADD, idx=0):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.base_frame
            m.ns, m.id, m.type, m.action = ns, idx, mtype, action
            m.pose.orientation.w = 1.0
            # A LINE_LIST left at the default color.a = 0 renders fully
            # transparent even when per-point colors[] are populated -- the same
            # reason the parent sets color.a on its normals marker. Folded into
            # base() so no marker here can forget it.
            m.color.a = 1.0
            return m

        def pt(p):
            return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

        def arrow(ns, start, direction, color, length):
            m = base(ns, Marker.ARROW)
            m.scale.x, m.scale.y, m.scale.z = 0.004, 0.012, 0.015
            m.color = color
            m.points = [pt(start), pt(np.asarray(start) + length * np.asarray(direction))]
            return m

        rejected = not np.isfinite(view["cost"])
        markers = [
            # The angle between these two arrows *is* perp -- the sort key.
            arrow("step_seed", view["seed"], view["normal"],
                  ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0), 0.06),
            arrow("step_pref", view["seed"], view["pref"], _GREEN, 0.12),
        ]

        m = base("step_seed_dot", Marker.SPHERE)
        m.scale.x = m.scale.y = m.scale.z = 0.006
        m.color = _RED if rejected else _GREEN
        m.pose.position = pt(view["seed"])
        markers.append(m)

        label = base("step_text", Marker.TEXT_VIEW_FACING)
        label.scale.z = 0.018
        label.color = _RED if rejected else _GREEN
        anchor = view["seed"] if view["X_G"] is None else view["X_G"][:3, 3]
        label.pose.position = pt(np.asarray(anchor) + np.array([0.0, 0.0, 0.14]))
        cost = "inf" if rejected else f"{view['cost']:+.2f}"
        label.text = (
            f"#{view['i'] + 1}/{len(self._step_order)}  perp={view['perp']:.3f}  "
            f"roll={np.degrees(view['theta']):+.0f}deg  cost={cost}\n"
            f"{view['info'] if rejected else 'ACCEPT'}"
        )
        markers.append(label)

        if view["X_G"] is None:
            # Nothing to draw the boxes on; clear last step's, or they linger.
            for ns in ("step_boxes", "step_crop", "step_ground"):
                markers.append(base(ns, Marker.LINE_LIST, action=Marker.DELETE))
            self.step_info_pub.publish(MarkerArray(markers=markers))
            return

        # The analytic geometry the accept/reject decision is actually made on,
        # and which the gripper mesh does not show. Red = this box is why.
        boxes = base("step_boxes", Marker.LINE_LIST)
        boxes.scale.x = 0.0015
        for b, (lo, hi) in enumerate(GRIPPER_COLLISION_BOXES):
            corners = transform_points(view["X_G"], box_corners(lo, hi))
            hit = view["box_hits"][b] or (view["info"] == REJECT_GROUND and view["ground_hits"][b])
            for a, z in _BOX_EDGES:
                boxes.points += [pt(corners[a]), pt(corners[z])]
                boxes.colors += [_RED, _RED] if hit else [_GREY, _GREY]
        markers.append(boxes)

        crop = base("step_crop", Marker.LINE_LIST)
        crop.scale.x = 0.002
        crop_color = _GREEN if view["n_inside"] else _RED
        corners = transform_points(view["X_G"], box_corners(CROP_MIN, CROP_MAX))
        for a, z in _BOX_EDGES:
            crop.points += [pt(corners[a]), pt(corners[z])]
            crop.colors += [crop_color, crop_color]
        markers.append(crop)

        ground = base("step_ground", Marker.LINE_LIST)
        if view["info"] == REJECT_GROUND:
            ground.scale.x = 0.002
            c = np.asarray(view["X_G"][:3, 3])
            z = view["ground_z"]
            s = 0.15
            sq = [
                np.array([c[0] - s, c[1] - s, z]), np.array([c[0] + s, c[1] - s, z]),
                np.array([c[0] + s, c[1] + s, z]), np.array([c[0] - s, c[1] + s, z]),
            ]
            for k in range(4):
                ground.points += [pt(sq[k]), pt(sq[(k + 1) % 4])]
                ground.colors += [_RED, _RED]
        else:
            ground.action = Marker.DELETE
        markers.append(ground)

        self.step_info_pub.publish(MarkerArray(markers=markers))

    # ------------------------------------------------------------ input thread
    def input_loop(self):
        """Own the prompt on a worker thread.

        Never blocks a callback: the pipeline runs inline in the cloud
        subscription on a single-threaded executor, so reading stdin there would
        stop both publish timers, let grasp_candidate_00 age out of TF, and make
        RViz drop the very gripper mesh being inspected. rclpy.spin keeps the
        main thread, and this thread only touches ranked_grasps/_debug_seeds/
        _step_view under the existing self._lock (never held across a show()
        call -- threading.Lock is not reentrant and publish_debug_axes
        reacquires it).
        """
        print(HELP, flush=True)
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    return  # stdin closed
                cmd = line.strip()

                if not self._step_order:
                    print("waiting for a point cloud...", flush=True)
                    continue

                if cmd == "":
                    self._step_roll_i = None
                    out = self.show(self._step_i + 1)
                elif cmd == "b":
                    self._step_roll_i = None
                    out = self.show(self._step_i - 1)
                elif cmd == "r":
                    self._step_roll_i = 0 if self._step_roll_i is None else self._step_roll_i + 1
                    out = self.show(self._step_i)
                elif cmd == "R":
                    self._step_roll_i = -1 if self._step_roll_i is None else self._step_roll_i - 1
                    out = self.show(self._step_i)
                elif cmd == "a":
                    out = self._next_accepted()
                elif cmd.lstrip("+-").isdigit():
                    self._step_roll_i = None
                    out = self.show(int(cmd) - 1)
                elif cmd in ("q", "quit"):
                    print("stopped stepping; still publishing the last pose", flush=True)
                    return
                else:
                    out = HELP
                print(out, flush=True)
            except Exception as exc:  # a debugger that dies on a typo is useless
                # With the traceback: without it a failure names the exception
                # but not the line, which is the only thing worth knowing here.
                self.get_logger().error(
                    f"step failed: {exc!r}\n{traceback.format_exc()}"
                )

    def _next_accepted(self) -> str:
        """Skip ahead to the next queue position with a feasible grasp."""
        self._step_roll_i = None
        for i in range(self._step_i + 1, len(self._step_order)):
            out = self.show(i)
            if self._step_view is not None and np.isfinite(self._step_view["cost"]):
                return out
        return f"no accepted grasp after #{self._step_i + 1}"


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = GraspStepDebug()
        threading.Thread(
            target=node.input_loop,
            # daemon: Ctrl+C lands on the main thread, and a non-daemon thread
            # parked in readline() would hold the process open until Enter.
            daemon=True,
        ).start()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        rclpy.logging.get_logger("grasp_step").error(str(exc))

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
