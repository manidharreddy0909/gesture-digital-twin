"""
Heuristic gesture detection module operating on MediaPipe hand landmarks.

Current phase:
- Detects simple per-hand gestures: fist, open hand, N fingers, pinch.
- Adds lightweight swipe detection using recent fingertip motion history.

Future phases can swap the heuristics with ML models (e.g., small CNN/RNN
or transformer) without changing the external API of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from motion_analyzer import HandMotionInfo, FingerTrajectory


@dataclass
class GestureResult:
    """
    Result of gesture detection for a single hand.

    Attributes:
        hand_id: Index of the hand in the frame (0, 1, ...).
        handedness: "Left", "Right", or "Unknown".
        gesture: High-level label (e.g., "fist", "open", "3_fingers", "pinch").
        num_fingers: Count of extended fingers (rough estimate).
        swipe: Optional swipe classification ("swipe_left", "swipe_right", ...).
    """

    hand_id: int
    handedness: str
    gesture: str
    num_fingers: int
    swipe: Optional[str]
    # Simple confidence score in [0, 1] for the selected gesture label.
    confidence: float = 0.0


@dataclass
class TwoHandGestureResult:
    """
    High-level description of a two-hand gesture, if any.

    Examples: "zoom_in", "zoom_out", "rotate_cw", "rotate_ccw", "separate".
    """

    gesture: str
    extra: str = ""


class GestureDetector:
    """
    Simple rule-based gesture recognizer.

    The detector uses:
        - Per-finger extension estimation from landmark geometry.
        - Thumb–index distance for pinch detection.
        - Short history of index-tip positions for coarse swipe detection.

    It does not depend on MediaPipe directly; it consumes the list of
    normalized landmarks produced by `HandTracker`.
    """

    def __init__(
        self,
        pinch_distance_threshold: float = 0.05,
        swipe_distance_threshold: float = 0.20,
        history_size: int = 5,
        # Angle threshold for finger straightness. Higher = stricter.
        extended_angle_threshold_deg: float = 160.0,
        # Extra margin for wrist-distance sanity checks (normalized units).
        wrist_distance_margin: float = 0.015,
        circle_min_path: float = 0.6,
        circle_close_distance: float = 0.05,
    ):
        self.pinch_distance_threshold = pinch_distance_threshold
        self.swipe_distance_threshold = swipe_distance_threshold
        self.history_size = history_size
        self.extended_angle_threshold_deg = extended_angle_threshold_deg
        self.wrist_distance_margin = wrist_distance_margin
        # Circle detection thresholds in normalized units.
        self.circle_min_path = circle_min_path
        self.circle_close_distance = circle_close_distance

        # Optional ML models (to be plugged in by the user).
        # static_model: Callable[[List], Tuple[str, float]]
        # dynamic_model: Callable[[HandMotionInfo], Tuple[str, float]]
        # They should return (label, confidence) and can be used to override
        # or refine the heuristic detections below.
        self.static_model = None
        self.dynamic_model = None

    # ---- Public API ----

    def detect(
        self,
        hand_id: int,
        handedness: str,
        landmarks: List,
        timestamp: float,  # kept for future temporal models
        motion: Optional[HandMotionInfo] = None,
    ) -> GestureResult:
        """
        Compute a gesture label for one hand.

        Args:
            hand_id: Index of the hand within the frame.
            handedness: "Left" / "Right" / "Unknown".
            landmarks: Sequence of normalized landmarks (0..20).
            timestamp: Current frame time in seconds (not heavily used yet).
            motion: Optional motion information for this hand, as computed
                by MotionAnalyzer. If provided, it is used for dynamic
                gesture classification (e.g., swipes).
        """
        num_fingers, is_pinch = self._classify_static_posture(landmarks, handedness)

        # --- Heuristic static gesture and confidence -------------------------
        if is_pinch:
            base_gesture = "pinch"
            static_conf = 1.0
        elif num_fingers == 0:
            base_gesture = "fist"
            static_conf = 0.9
        elif num_fingers >= 4:
            base_gesture = "open"
            static_conf = 0.8
        else:
            base_gesture = f"{num_fingers}_fingers"
            static_conf = 0.6

        # Optional ML-based static gesture override.
        if self.static_model is not None:
            ml_label, ml_conf = self.static_model(landmarks)
            if ml_conf > static_conf:
                base_gesture = ml_label
                static_conf = ml_conf

        # --- Dynamic gestures from motion history ----------------------------
        swipe: Optional[str] = None
        circle: Optional[str] = None
        dyn_conf = 0.0

        if motion is not None:
            # Heuristic dynamic detectors.
            swipe = self._classify_swipe_from_motion(motion)
            circle = self._classify_circle_from_motion(motion)

            if circle is not None:
                dyn_label = circle
                dyn_conf = 0.8
            elif swipe is not None:
                dyn_label = swipe
                dyn_conf = 0.7
            else:
                dyn_label = base_gesture

            # Optional ML-based dynamic gesture override.
            if self.dynamic_model is not None:
                ml_dyn_label, ml_dyn_conf = self.dynamic_model(motion)
                if ml_dyn_conf > dyn_conf:
                    dyn_label = ml_dyn_label
                    dyn_conf = ml_dyn_conf
        else:
            dyn_label = base_gesture

        # Final label: dynamic (if present) overrides static.
        if motion is not None and (circle is not None or swipe is not None or self.dynamic_model is not None):
            final_label = dyn_label
            final_conf = max(static_conf, dyn_conf)
        else:
            final_label = base_gesture
            final_conf = static_conf

        return GestureResult(
            hand_id=hand_id,
            handedness=handedness,
            gesture=final_label,
            num_fingers=num_fingers,
            swipe=swipe,
            confidence=final_conf,
        )

    # ---- Static posture heuristics ----

    def _classify_static_posture(
        self,
        landmarks: List,
        handedness: str,
    ) -> Tuple[int, bool]:
        """
        Estimate how many fingers are extended and whether a pinch occurs.

        This implementation is designed to be more robust than a simple
        "tip above PIP" rule:

        - **Finger extension**: computed from joint straightness (angles) plus
          a wrist-distance sanity check. This reduces false positives when the
          hand is rotated or when fingers are partially bent.
        - **Thumb extension**: handled via angles too (CMC→MCP→IP and MCP→IP→TIP),
          which avoids reliance on handedness (important when the input is mirrored).
        """
        # Landmark index groups:
        # Thumb: 1(CMC),2(MCP),3(IP),4(TIP)
        # Index: 5(MCP),6(PIP),7(DIP),8(TIP)
        # Middle: 9,10,11,12
        # Ring: 13,14,15,16
        # Pinky: 17,18,19,20
        wrist = landmarks[0]

        def p(idx: int) -> Tuple[float, float]:
            lm = landmarks[idx]
            return float(lm.x), float(lm.y)

        def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return math.sqrt(dx * dx + dy * dy)

        def angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
            """
            Angle at point b for triangle (a, b, c) in degrees.
            Straight finger joint -> ~180 deg.
            """
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            nba = math.sqrt(ba[0] * ba[0] + ba[1] * ba[1])
            nbc = math.sqrt(bc[0] * bc[0] + bc[1] * bc[1])
            if nba <= 1e-8 or nbc <= 1e-8:
                return 0.0
            cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (nba * nbc)
            cosv = max(-1.0, min(1.0, cosv))
            return math.degrees(math.acos(cosv))

        wrist_xy = (float(wrist.x), float(wrist.y))
        th = self.extended_angle_threshold_deg
        margin = self.wrist_distance_margin

        extended_flags: List[bool] = []

        # Thumb extension via angles + distance-from-wrist sanity check.
        thumb_cmc = p(1)
        thumb_mcp = p(2)
        thumb_ip = p(3)
        thumb_tip = p(4)
        thumb_angle1 = angle_deg(thumb_cmc, thumb_mcp, thumb_ip)
        thumb_angle2 = angle_deg(thumb_mcp, thumb_ip, thumb_tip)
        thumb_straight = (thumb_angle1 >= th) and (thumb_angle2 >= th)
        thumb_farther = dist(wrist_xy, thumb_tip) > dist(wrist_xy, thumb_ip) + margin
        extended_flags.append(thumb_straight and thumb_farther)

        # Other fingers: angles at PIP and DIP + wrist-distance check.
        fingers = [
            ("index", 5, 6, 7, 8),
            ("middle", 9, 10, 11, 12),
            ("ring", 13, 14, 15, 16),
            ("pinky", 17, 18, 19, 20),
        ]
        for _name, mcp_i, pip_i, dip_i, tip_i in fingers:
            mcp = p(mcp_i)
            pip = p(pip_i)
            dip = p(dip_i)
            tip = p(tip_i)

            a1 = angle_deg(mcp, pip, dip)
            a2 = angle_deg(pip, dip, tip)
            straight = (a1 >= th) and (a2 >= th)

            # Sanity: tip should be meaningfully farther from wrist than PIP.
            farther = dist(wrist_xy, tip) > dist(wrist_xy, pip) + margin

            extended_flags.append(straight and farther)

        num_fingers = int(sum(1 for f in extended_flags if f))

        # Pinch: thumb tip close to index tip in (x, y) space.
        index_tip = p(8)
        pinch_dist = dist(thumb_tip, index_tip)
        is_pinch = pinch_dist < self.pinch_distance_threshold

        return num_fingers, is_pinch

    # ---- Motion-based swipe detection (legacy, image-space only) ----

    def _classify_swipe(self, hand_id: int, landmarks: List) -> Optional[str]:
        """
        Fallback swipe detector that looks only at the current frame,
        used when MotionAnalyzer information is not available.
        """
        # Without history we cannot classify a real swipe; treat as none.
        return None

    # ---- Motion-based swipe / drag / circle detection via MotionAnalyzer ----

    def _classify_swipe_from_motion(
        self,
        motion: HandMotionInfo,
    ) -> Optional[str]:
        """
        Use index fingertip trajectory to estimate swipe direction.

        The same trajectory metrics can later be reused for richer classes
        (e.g., circular motions, drags), but here we focus on coarse swipes.
        """
        traj = motion.fingertip_trajectories.get(8)
        if traj is None or len(traj.points) < 2:
            return None

        x0, y0, _ = traj.points[0]
        x1, y1, _ = traj.points[-1]
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.swipe_distance_threshold:
            return None

        if abs(dx) > abs(dy):
            return "swipe_right" if dx > 0 else "swipe_left"
        return "swipe_down" if dy > 0 else "swipe_up"

    def _classify_circle_from_motion(
        self,
        motion: HandMotionInfo,
    ) -> Optional[str]:
        """
        Detect circular motion of the index fingertip based on its trajectory:
        - long path length,
        - small net displacement (start and end near each other).
        """
        traj = motion.fingertip_trajectories.get(8)
        if traj is None or len(traj.points) < 4:
            return None

        path_len = traj.path_length()
        net_disp = traj.net_displacement()

        if path_len < self.circle_min_path:
            return None
        if net_disp > self.circle_close_distance:
            return None

        return "circle"

    # ---- Two-hand gestures (zoom, rotate, mirror, push/pull) ----

    def detect_two_hand_gesture(
        self,
        hands: List,
        motions: Dict[int, HandMotionInfo],
    ) -> Optional[TwoHandGestureResult]:
        """
        Inspect two hands to detect advanced two-hand gestures:
        - zoom_in / zoom_out  (change in distance between wrists/index fingertips)
        - rotate_cw / rotate_ccw (relative angular motion)
        - mirror              (symmetric motion opposite directions)
        - push / pull         (hands moving toward/away from center)
        - cross               (hands crossing each other)

        This is heuristic-based but enhanced with multiple detection strategies.
        """
        if len(hands) < 2:
            return None

        h0, h1 = hands[0], hands[1]
        m0 = motions.get(h0.hand_id)
        m1 = motions.get(h1.hand_id)
        if m0 is None or m1 is None:
            return None

        t0 = m0.fingertip_trajectories.get(8)
        t1 = m1.fingertip_trajectories.get(8)
        if t0 is None or t1 is None or len(t0.points) < 2 or len(t1.points) < 2:
            return None

        def dist_pair(traj0: FingerTrajectory, traj1: FingerTrajectory, idx: int) -> float:
            x0, y0, _ = traj0.points[idx]
            x1, y1, _ = traj1.points[idx]
            dx = x1 - x0
            dy = y1 - y0
            return math.sqrt(dx * dx + dy * dy)

        def angle_pair(traj0: FingerTrajectory, traj1: FingerTrajectory, idx: int) -> float:
            """Angle from hand0 to hand1 at index idx."""
            x0, y0, _ = traj0.points[idx]
            x1, y1, _ = traj1.points[idx]
            return math.atan2(y1 - y0, x1 - x0)

        # --- Zoom detection (distance change) ---
        d0 = dist_pair(t0, t1, 0)
        d1 = dist_pair(t0, t1, -1)
        zoom_delta = d1 - d0
        zoom_threshold = 0.08  # normalized distance

        if zoom_delta > zoom_threshold:
            return TwoHandGestureResult(gesture="zoom_out", extra=f"Δd={zoom_delta:.3f}")
        if zoom_delta < -zoom_threshold:
            return TwoHandGestureResult(gesture="zoom_in", extra=f"Δd={zoom_delta:.3f}")

        # --- Rotation detection (angular change) ---
        if len(t0.points) >= 3 and len(t1.points) >= 3:
            ang_start = angle_pair(t0, t1, 0)
            ang_end = angle_pair(t0, t1, -1)
            ang_delta = ang_end - ang_start

            # Normalize angle delta to [-pi, pi]
            while ang_delta > math.pi:
                ang_delta -= 2 * math.pi
            while ang_delta < -math.pi:
                ang_delta += 2 * math.pi

            rotation_threshold = 0.5  # radians
            if abs(ang_delta) > rotation_threshold:
                if ang_delta > 0:
                    return TwoHandGestureResult(gesture="rotate_ccw", extra=f"Δθ={math.degrees(ang_delta):.1f}°")
                else:
                    return TwoHandGestureResult(gesture="rotate_cw", extra=f"Δθ={math.degrees(ang_delta):.1f}°")

        # --- Mirror detection (symmetric opposite motion) ---
        if len(t0.velocities) > 0 and len(t1.velocities) > 0:
            vx0, vy0 = t0.velocities[-1]
            vx1, vy1 = t1.velocities[-1]

            # Check if velocities are roughly opposite (mirror motion)
            dot_product = vx0 * vx1 + vy0 * vy1
            mag0 = math.sqrt(vx0 * vx0 + vy0 * vy0)
            mag1 = math.sqrt(vx1 * vx1 + vy1 * vy1)

            if mag0 > 0.1 and mag1 > 0.1:
                normalized_dot = dot_product / (mag0 * mag1)
                if normalized_dot < -0.7:  # Nearly opposite directions
                    return TwoHandGestureResult(gesture="mirror", extra=f"sim={normalized_dot:.2f}")

        # --- Push/Pull detection (hands moving toward/away from midpoint) ---
        if len(t0.points) >= 2 and len(t1.points) >= 2:
            # Get midpoint
            x0_start, y0_start, _ = t0.points[-2]
            x1_start, y1_start, _ = t1.points[-2]
            mid_x_start = (x0_start + x1_start) / 2.0
            mid_y_start = (y0_start + y1_start) / 2.0

            x0_end, y0_end, _ = t0.points[-1]
            x1_end, y1_end, _ = t1.points[-1]
            mid_x_end = (x0_end + x1_end) / 2.0
            mid_y_end = (y0_end + y1_end) / 2.0

            # Distances from hands to midpoint
            d0_to_mid_start = math.sqrt((x0_start - mid_x_start) ** 2 + (y0_start - mid_y_start) ** 2)
            d1_to_mid_start = math.sqrt((x1_start - mid_x_start) ** 2 + (y1_start - mid_y_start) ** 2)
            d0_to_mid_end = math.sqrt((x0_end - mid_x_end) ** 2 + (y0_end - mid_y_end) ** 2)
            d1_to_mid_end = math.sqrt((x1_end - mid_x_end) ** 2 + (y1_end - mid_y_end) ** 2)

            avg_dist_start = (d0_to_mid_start + d1_to_mid_start) / 2.0
            avg_dist_end = (d0_to_mid_end + d1_to_mid_end) / 2.0
            push_pull_delta = avg_dist_end - avg_dist_start
            push_pull_threshold = 0.05

            if push_pull_delta > push_pull_threshold:
                return TwoHandGestureResult(gesture="push", extra=f"Δmid={push_pull_delta:.3f}")
            if push_pull_delta < -push_pull_threshold:
                return TwoHandGestureResult(gesture="pull", extra=f"Δmid={push_pull_delta:.3f}")

        # --- Cross detection (hands crossing paths) ---
        if len(t0.points) >= 2 and len(t1.points) >= 2:
            x0_prev, y0_prev, _ = t0.points[-2]
            x0_curr, y0_curr, _ = t0.points[-1]
            x1_prev, y1_prev, _ = t1.points[-2]
            x1_curr, y1_curr, _ = t1.points[-1]

            # Check if hands crossed x-coordinate (simple crossing detection)
            if (x0_prev < x1_prev and x0_curr > x1_curr) or (x0_prev > x1_prev and x0_curr < x1_curr):
                return TwoHandGestureResult(gesture="cross", extra="hands_crossed")

        sep_threshold = 0.05
        if abs(zoom_delta) > sep_threshold:
            return TwoHandGestureResult(gesture="separate", extra=f"Δd={zoom_delta:.3f}")

        return None

