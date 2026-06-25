"""MediaPipe Hands wrapper that yields stable, structured HandData.

Uses the modern MediaPipe **Tasks** API (`mediapipe.tasks.python.vision`)
which is the only one available on Python 3.13. Requires the
`hand_landmarker.task` model file at ``models/hand_landmarker.task``
(download URL recorded in README).

Why this exists:
- Raw MediaPipe output is noisy frame-to-frame (sub-pixel jitter).
- When we mirror the camera for a selfie view, MediaPipe's "Left"/"Right"
  labels invert because they are derived from the original (un-mirrored)
  image perspective. We correct this here so downstream code can trust
  the labels.
- Per-landmark One Euro filtering removes shake without adding lag for
  fast motion.
- Per-hand state (previous palm, velocity, smoothing filters) is keyed
  by *corrected* label so it survives momentary detection drops as long
  as MediaPipe re-identifies the same hand.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from math import atan2, degrees

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from core.state import HandData
from gestures.smoothing import OneEuroFilter
from vision.landmarks import (
    FINGER_TIPS,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
)

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "hand_landmarker.task",
)


@dataclass
class _HandState:
    """Per-hand persistent smoothing + velocity state."""
    landmark_filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            mincutoff=config.ONE_EURO_MIN_CUTOFF,
            beta=config.ONE_EURO_BETA,
            dcutoff=config.ONE_EURO_DCUTOFF,
        )
    )
    velocity_filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            mincutoff=config.HAND_VELOCITY_MIN_CUTOFF,
            beta=config.HAND_VELOCITY_BETA,
            dcutoff=config.ONE_EURO_DCUTOFF,
        )
    )
    palm_size_velocity_filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            mincutoff=config.HAND_PALM_SIZE_VELOCITY_MIN_CUTOFF,
            beta=config.HAND_PALM_SIZE_VELOCITY_BETA,
            dcutoff=config.ONE_EURO_DCUTOFF,
        )
    )
    # Index-fingertip velocity uses the same responsiveness as the palm velocity.
    index_tip_velocity_filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(
            mincutoff=config.HAND_VELOCITY_MIN_CUTOFF,
            beta=config.HAND_VELOCITY_BETA,
            dcutoff=config.ONE_EURO_DCUTOFF,
        )
    )
    last_palm: np.ndarray | None = None
    last_t: float | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_index_tip: np.ndarray | None = None
    index_tip_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_palm_size: float | None = None
    palm_size_velocity: float = 0.0
    last_seen_t: float | None = None
    # Flick history: deque of (timestamp, velocity_2d) tuples kept for a short
    # rolling window.  Used to find the peak movement near the moment of release.
    flick_history: deque = field(default_factory=deque)
    # Most recently captured strong flick: unit direction + speed + capture time.
    flick_dir: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    flick_speed: float = 0.0
    flick_captured_t: float | None = None

    def reset_tracking(self) -> None:
        self.landmark_filter.reset()
        self.velocity_filter.reset()
        self.palm_size_velocity_filter.reset()
        self.index_tip_velocity_filter.reset()
        self.last_palm = None
        self.last_t = None
        self.velocity = np.zeros(2)
        self.last_index_tip = None
        self.index_tip_velocity = np.zeros(2)
        self.last_palm_size = None
        self.palm_size_velocity = 0.0
        self.last_seen_t = None
        self.flick_history.clear()
        self.flick_dir = np.zeros(2, dtype=np.float32)
        self.flick_speed = 0.0
        self.flick_captured_t = None


class HandTracker:
    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Download with:\n  curl -L -o models/hand_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=config.MP_MAX_HANDS,
            min_hand_detection_confidence=config.MP_MIN_DETECT_CONF,
            min_hand_presence_confidence=config.MP_MIN_TRACK_CONF,
            min_tracking_confidence=config.MP_MIN_TRACK_CONF,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        self._states: dict[str, _HandState] = {
            "Left": _HandState(),
            "Right": _HandState(),
        }
        # Tasks VIDEO mode wants strictly monotonically increasing
        # millisecond timestamps. We track our own counter to guarantee that
        # even if wall-clock occasionally goes backwards (rare).
        self._last_ts_ms: int = 0

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            log.warning("HandTracker.close() failed", exc_info=True)

    @staticmethod
    def _correct_label(raw_label: str) -> str:
        """Mirror-fix: when we flip the input image for a selfie view,
        MediaPipe's handedness is inverted. Swap it back."""
        if not config.INVERT_HANDEDNESS_AFTER_MIRROR:
            return raw_label
        return "Left" if raw_label == "Right" else "Right"

    @staticmethod
    def _resolve_duplicate_labels(detections: list[dict]) -> None:
        """MediaPipe occasionally emits two hands with the same handedness.

        When that happens, downstream state keyed by "Left"/"Right" can flicker
        or overwrite itself. Screen x-position is a reliable fallback in selfie
        view: left side of the image is the user's left hand when mirrored.
        """
        if len(detections) != 2:
            return
        labels = {str(d["label"]) for d in detections}
        if labels == {"Left", "Right"}:
            return

        screen_left, screen_right = (
            ("Left", "Right") if config.MIRROR_INPUT else ("Right", "Left")
        )
        ordered = sorted(detections, key=lambda d: float(d["raw_palm"][0]))
        ordered[0]["label"] = screen_left
        ordered[1]["label"] = screen_right

    def process(self, frame_bgr: np.ndarray, t: float) -> list[HandData]:
        h, w = frame_bgr.shape[:2]
        # Tasks API wants an mp.Image in SRGB, not raw numpy.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Downscale the inference image when the capture is large. MediaPipe
        # returns NORMALISED landmarks, so downstream geometry (palm_px uses the
        # original h, w) is unaffected — the detector just preprocesses far fewer
        # pixels. Cheapest reliable FPS win at 720p+. INTER_AREA preserves edges.
        if w > config.MP_INPUT_MAX_WIDTH:
            scale = config.MP_INPUT_MAX_WIDTH / float(w)
            rgb = cv2.resize(
                rgb,
                (config.MP_INPUT_MAX_WIDTH, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = max(self._last_ts_ms + 1, int(t * 1000.0))
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        out: list[HandData] = []
        if not result.hand_landmarks:
            for state in self._states.values():
                state.velocity *= 0.5
                state.palm_size_velocity *= 0.5
                if (
                    state.last_seen_t is not None
                    and t - state.last_seen_t > config.HAND_STALE_RESET_SECONDS
                ):
                    state.reset_tracking()
            return out

        detections: list[dict] = []
        for lm_set, handed in zip(result.hand_landmarks, result.handedness, strict=False):
            category = handed[0] if handed else None
            raw_label = category.category_name if category is not None else "Right"
            score = (
                float(getattr(category, "score", 1.0))
                if category is not None
                else 1.0
            )
            label = self._correct_label(raw_label)
            lm = np.array(
                [[p.x, p.y, p.z] for p in lm_set],
                dtype=np.float32,
            )
            raw_palm = np.mean(
                lm[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2],
                axis=0,
            )
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "landmarks": lm,
                    "raw_palm": raw_palm,
                }
            )

        self._resolve_duplicate_labels(detections)

        seen: set[str] = set()
        for detection in detections:
            label = str(detection["label"])
            if label not in self._states:
                continue
            seen.add(label)
            state = self._states[label]
            lm = detection["landmarks"]
            score = float(detection["score"])

            prev_t = state.last_t
            if (
                prev_t is not None
                and t - prev_t > config.HAND_STALE_RESET_SECONDS
            ):
                state.reset_tracking()
                prev_t = None

            lm_smoothed = state.landmark_filter(lm, t)

            # Use centroid of WRIST + finger MCPs as a stable palm centre.
            palm = np.mean(
                lm_smoothed[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2],
                axis=0,
            )

            prev_palm = state.last_palm
            if prev_palm is not None and prev_t is not None:
                dt = max(1e-3, t - prev_t)
                raw_velocity = (palm - prev_palm) / dt
                state.velocity = np.asarray(
                    state.velocity_filter(raw_velocity, t),
                    dtype=np.float32,
                )
            else:
                state.velocity = np.zeros(2, dtype=np.float32)
                state.velocity_filter.reset()

            # Index-fingertip velocity: how fast the tip itself is flicking,
            # independent of the palm. This is what fires the fireball.
            index_tip = lm_smoothed[INDEX_TIP, :2]
            prev_index_tip = state.last_index_tip
            if prev_index_tip is not None and prev_t is not None:
                tip_dt = max(1e-3, t - prev_t)
                raw_tip_velocity = (index_tip - prev_index_tip) / tip_dt
                state.index_tip_velocity = np.asarray(
                    state.index_tip_velocity_filter(raw_tip_velocity, t),
                    dtype=np.float32,
                )
            else:
                state.index_tip_velocity = np.zeros(2, dtype=np.float32)
                state.index_tip_velocity_filter.reset()
            state.last_index_tip = index_tip

            fingers_open = _finger_openness(lm_smoothed)
            spread = _finger_spread(lm_smoothed)
            pinch = float(
                np.linalg.norm(
                    lm_smoothed[THUMB_TIP, :2] - lm_smoothed[INDEX_TIP, :2]
                )
            )
            palm_normal = _palm_normal(lm_smoothed, label)

            # Palm size: distance from wrist to middle MCP. Grows when the
            # hand approaches the camera, which is the signal we use for
            # forward-thrust detection.
            palm_size = float(
                np.linalg.norm(
                    lm_smoothed[MIDDLE_MCP, :2] - lm_smoothed[WRIST, :2]
                )
            )
            prev_palm_size = state.last_palm_size
            if prev_palm_size is not None and prev_t is not None:
                if palm_size < 1e-4:
                    # Degenerate palm: landmarks collapsed; skip to avoid spurious spikes.
                    state.palm_size_velocity = 0.0
                else:
                    ddt = max(1e-3, t - prev_t)
                    # Fractional rate so it's scale-invariant (per-second growth).
                    raw_size_velocity = (palm_size - prev_palm_size) / max(
                        1e-4, prev_palm_size
                    ) / ddt
                    state.palm_size_velocity = float(
                        state.palm_size_velocity_filter(raw_size_velocity, t)
                    )
            else:
                state.palm_size_velocity = 0.0
                state.palm_size_velocity_filter.reset()

            state.last_palm = palm
            state.last_palm_size = palm_size
            state.last_t = t
            state.last_seen_t = t

            # --- New derived signals ---
            _update_flick(state, t)
            orientation = _hand_orientation(palm_normal)
            wrist_angle = _wrist_angle_deg(lm_smoothed)

            palm_px = (int(palm[0] * w), int(palm[1] * h))

            out.append(
                HandData(
                    label=label,
                    palm=palm,
                    palm_px=palm_px,
                    velocity=state.velocity.copy(),
                    fingers_open=fingers_open,
                    openness=float(np.mean(fingers_open)),
                    spread=spread,
                    pinch=pinch,
                    landmarks=lm_smoothed,
                    palm_size=palm_size,
                    palm_size_velocity=float(state.palm_size_velocity),
                    tracking_confidence=score,
                    palm_normal=palm_normal,
                    orientation=orientation,
                    wrist_angle_deg=wrist_angle,
                    flick=state.flick_dir.copy(),
                    flick_speed=state.flick_speed,
                    index_tip_velocity=state.index_tip_velocity.copy(),
                )
            )

        for missing in set(self._states) - seen:
            state = self._states[missing]
            state.velocity *= 0.5
            state.palm_size_velocity *= 0.5
            if (
                state.last_seen_t is not None
                and t - state.last_seen_t > config.HAND_STALE_RESET_SECONDS
            ):
                state.reset_tracking()

        return out


def _vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two 3-D vectors (uses all three components)."""
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    cos_val = dot / (na * nb)
    # Clamp to [-1, 1] to guard against floating-point drift.
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def _finger_openness(lm: np.ndarray) -> np.ndarray:
    """Return 5-vector in 0..1 of how extended each finger is.

    For the four non-thumb fingers (index..pinky) we compute the *interior*
    joint angle at the PIP using the 3-D landmarks:
      - vector A = PIP -> MCP
      - vector B = PIP -> TIP
    (the angle whose vertex is the PIP).
    When the finger is straight, the angle at PIP approaches 180° → mapped
    to 1.0.  When fully curled the angle drops toward 90° (or less) → 0.0.
    The mapping is a linear interpolation between
    config.HAND_FINGER_FOLDED_ANGLE_DEG (→ 0.0) and
    config.HAND_FINGER_EXTENDED_ANGLE_DEG (→ 1.0).

    The thumb (index 0) is measured differently: it barely curls at its IP
    joint, so the interior-angle metric doesn't apply. We compare the thumb
    TIP's distance from the opposite palm corner (PINKY_MCP) to the thumb IP
    joint's distance from that same anchor. Extended => the tip reaches past the
    joint (ratio > 1); tucked across the palm => the tip swings back toward the
    pinky side, closer than the joint (ratio < 1). This is curl-direction aware
    and rotation-invariant — unlike the previous cross-product-vs-knuckle-axis
    heuristic, which read a closed fist's thumb as "extended" and a splayed
    thumb as "folded" (the inverted-thumb bug seen on the debug overlay).
    """
    out = np.zeros(5, dtype=np.float32)

    # --- Thumb (index 0) ---
    anchor = lm[PINKY_MCP, :2]
    d_tip = float(np.linalg.norm(lm[THUMB_TIP, :2] - anchor))
    d_ip = float(np.linalg.norm(lm[THUMB_IP, :2] - anchor))
    ratio = d_tip / (d_ip + 1e-9)
    lo_r = config.THUMB_OPEN_RATIO_FOLDED
    hi_r = config.THUMB_OPEN_RATIO_EXTENDED
    out[0] = float(np.clip((ratio - lo_r) / (hi_r - lo_r + 1e-9), 0.0, 1.0))

    # --- Index, Middle, Ring, Pinky (indices 1-4) ---
    # Landmark tuples: (MCP, PIP, TIP) for each finger.
    non_thumb = (
        (INDEX_MCP,  INDEX_PIP,  INDEX_TIP),
        (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
        (RING_MCP,   RING_PIP,   RING_TIP),
        (PINKY_MCP,  PINKY_PIP,  PINKY_TIP),
    )
    lo = config.HAND_FINGER_FOLDED_ANGLE_DEG
    hi = config.HAND_FINGER_EXTENDED_ANGLE_DEG
    span = hi - lo

    for i, (mcp_i, pip_i, tip_i) in enumerate(non_thumb, start=1):
        # Interior joint angle AT the PIP vertex: between PIP->MCP and PIP->TIP.
        # Straight finger ≈ 180°, curled finger drops toward 90° or less. Using
        # MCP->PIP for vec_a instead would measure the deflection (the supplement
        # of this angle), which inverts the mapping and makes a fist read as a
        # fully-open hand — the "fist shows green / matches open palm" bug.
        vec_a = lm[mcp_i] - lm[pip_i]    # PIP -> MCP (3-D)
        vec_b = lm[tip_i] - lm[pip_i]    # PIP -> TIP (3-D)
        angle = _vec_angle_deg(vec_a, vec_b)
        out[i] = float(np.clip((angle - lo) / (span + 1e-9), 0.0, 1.0))

    return out


def _palm_normal(lm: np.ndarray, label: str = "Right") -> np.ndarray:
    """Compute an approximate unit normal to the palm plane.

    Uses three palm-base landmarks: WRIST (0), INDEX_MCP (5), PINKY_MCP (17).
    The normal is computed as:

        n = (INDEX_MCP - WRIST) x (PINKY_MCP - WRIST)

    then normalised to unit length.

    HANDEDNESS: a left hand and a right hand are mirror images, so the
    cross-product winding flips between them — the SAME physical orientation
    yields opposite normals. That made Time Freeze (which needs palm-toward-
    camera) require the *back* of the right hand while the left hand worked
    normally. We negate the normal for one handedness so that "palm toward
    camera" (and "palm up", for Rasengan) map to the same sign for both hands.
    Empirically the Right-labelled hand was the inverted one; if a future build
    flips the other way, invert the single ``label`` condition below.

    Sign convention (MediaPipe normalised coordinate space):
        - x increases to the right of the *original* (un-mirrored) image.
        - y increases downward.
        - z is negative when a point is CLOSER to the camera, positive when
          further away.

    Therefore, for a right hand with palm facing the camera (selfie view),
    INDEX_MCP is to the left of WRIST and PINKY_MCP is to the right, making
    the cross product point in the +z direction (away from the camera).

    In practice:
        palm_normal[2] < 0  →  palm facing TOWARD the camera (the common pose)
        palm_normal[2] > 0  →  palm facing AWAY from the camera

    To check "palm faces camera", test: -palm_normal[2] > config.PALM_FACING_CAMERA_DOT

    Returns a (3,) float32 array. Returns a zero-z vector (0, 0, 0) only if
    landmarks are degenerate (area collapses to zero).
    """
    wrist = lm[WRIST]           # shape (3,)
    idx_mcp = lm[INDEX_MCP]
    pinky_mcp = lm[PINKY_MCP]

    v1 = idx_mcp - wrist        # WRIST -> INDEX_MCP
    v2 = pinky_mcp - wrist      # WRIST -> PINKY_MCP

    normal = np.cross(v1, v2).astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    normal /= norm
    if label == "Right":
        normal = -normal
    return normal


def _finger_spread(lm: np.ndarray) -> float:
    """Average gap between adjacent fingertips, normalised by palm width."""
    palm_width = np.linalg.norm(lm[INDEX_MCP, :2] - lm[PINKY_MCP, :2]) + 1e-6
    gaps = [
        np.linalg.norm(lm[FINGER_TIPS[i], :2] - lm[FINGER_TIPS[i + 1], :2])
        for i in range(1, 4)
    ]
    return float(np.clip(np.mean(gaps) / palm_width, 0, 2))


def _hand_orientation(palm_normal: np.ndarray) -> str:
    """Classify the hand as "palm", "back", or "edge" from *palm_normal*.

    MediaPipe's z is negative toward the camera, so:
      -palm_normal[2] > 0  →  palm faces the camera  ("palm")
      +palm_normal[2] > 0  →  back faces the camera  ("back")
      neither               →  edge-on view           ("edge")
    """
    neg_z = float(-palm_normal[2])
    pos_z = float(palm_normal[2])
    if neg_z >= config.HAND_ORIENT_FACING_MIN:
        return "palm"
    if pos_z >= config.HAND_ORIENT_BACK_MIN:
        return "back"
    return "edge"


def _wrist_angle_deg(lm: np.ndarray) -> float:
    """Return angle of the wrist→MIDDLE_MCP vector from straight up (screen −y).

    0 ≈ pointing up, 90 ≈ horizontal, 180 ≈ pointing down.
    Uses only the 2-D (x, y) components of landmarks.
    """
    vec = lm[MIDDLE_MCP, :2] - lm[WRIST, :2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return 0.0
    # atan2(x, -y): zero when pointing straight up (y decreasing), ±90 when horizontal.
    angle = degrees(atan2(float(vec[0]), float(-vec[1])))
    # Map to 0..180 (symmetric: we care about tilt magnitude, not left/right lean).
    return float(abs(angle))


def _update_flick(state: _HandState, t: float) -> None:
    """Update the per-hand flick history and capture/decay the current flick.

    Appends the current smoothed velocity to a rolling window keyed by *t*, trims
    entries older than ``config.HAND_FLICK_HISTORY_SECONDS``, then finds the
    sample with the maximum speed. If that speed exceeds
    ``config.HAND_FLICK_MIN_SPEED``, the flick direction and speed are captured
    (overwriting any older capture). A captured flick persists for
    ``config.HAND_FLICK_DECAY_SECONDS``; after that it resets to zero so stale
    aim data cannot confuse later throws.
    """
    vel = state.velocity  # smoothed, already set for this frame

    # Record the current velocity sample.
    state.flick_history.append((t, vel.copy()))

    # Trim the window.
    cutoff = t - config.HAND_FLICK_HISTORY_SECONDS
    while state.flick_history and state.flick_history[0][0] < cutoff:
        state.flick_history.popleft()

    # Find the peak speed in the window.
    peak_speed: float = 0.0
    peak_vel: np.ndarray = np.zeros(2, dtype=np.float32)
    for _ts, v in state.flick_history:
        spd = float(np.linalg.norm(v))
        if spd > peak_speed:
            peak_speed = spd
            peak_vel = v

    if peak_speed >= config.HAND_FLICK_MIN_SPEED:
        # Capture (or refresh) the flick.
        state.flick_dir = (peak_vel / peak_speed).astype(np.float32)
        state.flick_speed = peak_speed
        state.flick_captured_t = t

    # Decay: if the captured flick is older than the decay window, clear it.
    if (
        state.flick_captured_t is not None
        and t - state.flick_captured_t > config.HAND_FLICK_DECAY_SECONDS
    ):
        state.flick_dir = np.zeros(2, dtype=np.float32)
        state.flick_speed = 0.0
        state.flick_captured_t = None
