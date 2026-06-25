"""MediaPipe Face Landmarker wrapper that yields structured FaceData.

Uses the modern MediaPipe **Tasks** API (`mediapipe.tasks.python.vision`)
with `FaceLandmarker` in `RunningMode.VIDEO`.  Requires the
``face_landmarker.task`` model file at ``models/face_landmarker.task``
(download with ``python scripts/download_model.py``).

Design notes
============
- The model file is only required at construction time; importing this module
  is always safe.
- Timestamps must be strictly monotonically increasing (same contract as
  ``HandTracker``).  We manage our own millisecond counter.
- Eye-blink detection uses the ``eyeBlinkLeft`` / ``eyeBlinkRight`` face
  blendshapes that the ``FaceLandmarker`` optionally outputs.
- ``eyes_closed_duration`` accumulates while both eyes are continuously closed
  and is reset to 0.0 whenever either eye opens (or the face disappears).
- When no face is detected this frame, ``FaceData(present=False)`` is returned
  and the blink timer resets.
- All blendshape access is wrapped defensively; no exception propagates out of
  ``process()``.
"""

from __future__ import annotations

import logging
import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from core.state import FaceData

log = logging.getLogger(__name__)

# Default model path relative to the repo root.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)

# MediaPipe FaceLandmarker has 468 landmarks in its base mesh, plus 10 iris
# landmarks (indices 468-477) when iris is enabled.  We use only the mesh here.
#
# Stable "face centre" landmarks — roughly the nose tip and four cheek/chin
# anchor points.  These indices are consistent across the 468-point model.
_STABLE_LM_INDICES = (1, 152, 234, 454, 33, 263)  # nose, chin, cheeks, temples

# Approximate eye-region landmark indices used to compute pixel positions.
# Left eye centre ≈ mean of left iris ring; right likewise.
# Fallback: well-known "left eye" / "right eye" mesh landmarks.
_LEFT_EYE_INDICES = (468, 469, 470, 471, 472)   # left iris (if present)
_RIGHT_EYE_INDICES = (473, 474, 475, 476, 477)  # right iris (if present)
# Fallback eye landmarks present in the 468-point mesh.
_LEFT_EYE_FALLBACK = (33, 133, 160, 159, 158, 144, 145, 153)
_RIGHT_EYE_FALLBACK = (362, 263, 387, 386, 385, 373, 374, 380)

# Head-orientation landmarks used to derive looking direction from head pose.
# Landmark 1 = nose tip (MediaPipe canonical face mesh, consistent across 468-
# and 478-point models).  33 = left eye outer corner; 263 = right eye outer
# corner.  The vector from the inter-eye midpoint to the nose tip captures
# both left/right (head yaw) and up/down (head pitch) in 2-D screen space.
#
# Mirror / sign note:
#   The frame arriving here is ALREADY mirrored (selfie view, MIRROR_INPUT=True).
#   In a mirrored frame the x-axis still increases leftward on screen — it has
#   NOT been re-flipped; OpenCV flip() was applied to the pixel buffer so index 0
#   is still the left pixel column as drawn on screen.  Consequently MediaPipe
#   landmark x also increases to the RIGHT of the screen, which is the direction
#   the USER sees as "right."  No sign negation is needed: if the user turns to
#   THEIR right the nose tip is to the right of the eye-midpoint → dx > 0 →
#   beam travels right on screen, which is what the user sees as "right."
_NOSE_TIP_INDEX: int = 1
_LEFT_EYE_OUTER: int = 33    # user's left eye outer corner in MP face topology
_RIGHT_EYE_OUTER: int = 263  # user's right eye outer corner

# Reference cadence (≈ the face-detection rate) at which LASER_EYES_GAZE_SMOOTH is
# calibrated. The gaze EMA is scaled by dt against this so the smoothing FEELS the
# same whether the camera runs at 30 or 60 fps (frame-rate-correct smoothing).
_GAZE_SMOOTH_REF_DT: float = 1.0 / 30.0


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker for single-face tracking.

    Parameters
    ----------
    model_path:
        Path to ``face_landmarker.task``.  Raises ``FileNotFoundError`` at
        construction time if the file is missing.

    Usage::

        tracker = FaceTracker()
        face = tracker.process(frame_bgr, t)
        # face.present, face.both_eyes_closed, face.eyes_closed_duration, …
        tracker.close()
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face landmarker model not found at {model_path}.\n"
                "Download it with:\n"
                "  python scripts/download_model.py\n"
                "or manually:\n"
                "  curl -L -o models/face_landmarker.task \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            )

        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            num_faces=1,
            min_face_detection_confidence=config.FACE_MIN_DETECT_CONF,
            min_face_presence_confidence=config.FACE_MIN_PRESENCE_CONF,
            min_tracking_confidence=config.FACE_MIN_TRACK_CONF,
            output_face_blendshapes=True,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

        # Strictly monotonically increasing millisecond timestamp counter.
        self._last_ts_ms: int = 0
        # Running eye-closed duration accumulator (seconds).
        self._eyes_closed_duration: float = 0.0
        # Timestamp of the last processed frame (for dt calculation).
        self._last_t: float | None = None
        # EMA-smoothed gaze OFFSET (magnitude-carrying 2-vector, screen space
        # x-right y-down). (0, 0) = looking straight ahead. See _compute_gaze.
        self._prev_gaze: np.ndarray = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, t: float) -> FaceData:
        """Detect and track one face in *frame_bgr*, sampled at time *t*.

        Parameters
        ----------
        frame_bgr:
            Current (already-mirrored) webcam frame in BGR uint8.
        t:
            Monotonic timestamp in seconds (same clock used throughout Conjure).

        Returns
        -------
        FaceData
            Always succeeds; returns ``FaceData(present=False)`` when no face
            is detected or if an internal error occurs.
        """
        try:
            return self._process_impl(frame_bgr, t)
        except Exception:
            log.exception("FaceTracker.process() raised unexpectedly")
            return FaceData(present=False)

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker."""
        try:
            self._landmarker.close()
        except Exception:
            log.warning("FaceTracker.close() failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_impl(self, frame_bgr: np.ndarray, t: float) -> FaceData:
        h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = max(self._last_ts_ms + 1, int(t * 1000.0))
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        # Compute dt for duration accumulation.
        dt = (t - self._last_t) if self._last_t is not None else 0.0
        self._last_t = t

        if not result.face_landmarks:
            self._eyes_closed_duration = 0.0
            self._prev_gaze = np.zeros(2, dtype=np.float32)
            return FaceData(present=False)

        lm_list = result.face_landmarks[0]  # list of NormalizedLandmark
        num_lm = len(lm_list)

        # Build numpy array of normalised landmarks: (N, 3)
        lm = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)

        # --- Eye blink via blendshapes ---
        both_eyes_closed = False
        try:
            blendshapes = (
                result.face_blendshapes[0]
                if result.face_blendshapes
                else []
            )
            blink_left = _read_blendshape(blendshapes, "eyeBlinkLeft")
            blink_right = _read_blendshape(blendshapes, "eyeBlinkRight")
            thresh = config.FACE_EYE_CLOSED_BLENDSHAPE_THRESHOLD
            both_eyes_closed = (
                blink_left is not None
                and blink_right is not None
                and blink_left > thresh
                and blink_right > thresh
            )
        except Exception:
            log.debug("FaceTracker: blendshape read failed", exc_info=True)

        # Accumulate or reset the closed-eyes timer.
        if both_eyes_closed:
            self._eyes_closed_duration += max(0.0, dt)
        else:
            self._eyes_closed_duration = 0.0

        # --- Eye pixel positions ---
        left_eye_px = _eye_center_px(lm, num_lm, _LEFT_EYE_INDICES,
                                     _LEFT_EYE_FALLBACK, w, h)
        right_eye_px = _eye_center_px(lm, num_lm, _RIGHT_EYE_INDICES,
                                      _RIGHT_EYE_FALLBACK, w, h)

        # --- Face centre ---
        face_center = _face_center(lm, num_lm, _STABLE_LM_INDICES)

        # --- Gaze direction ---
        # Freeze the gaze while the eyes are closed (a blink or the charge hold):
        # the iris/eyelid landmarks are unreliable then, so a blink must NOT fling
        # the aim, and the value held through the close becomes the clean
        # per-activation baseline the instant the eyes reopen to fire.
        gaze = self._compute_gaze(lm, num_lm, dt, frozen=both_eyes_closed)

        return FaceData(
            present=True,
            both_eyes_closed=both_eyes_closed,
            eyes_closed_duration=self._eyes_closed_duration,
            left_eye_px=left_eye_px,
            right_eye_px=right_eye_px,
            face_center=face_center,
            gaze=gaze,
            landmarks=lm,
        )

    def _compute_gaze(
        self, lm: np.ndarray, num_lm: int, dt: float = 0.0, frozen: bool = False
    ) -> np.ndarray:
        """Estimate gaze as a smoothed, MAGNITUDE-CARRYING 2-D offset (screen space).

        Unlike a unit direction, the LENGTH of the returned vector encodes how
        far the user is looking from straight-ahead, so the laser can map it
        directly to a pixel offset and reach any point — including the area right
        around the face — with no unreachable ring. Two signals are blended, both
        divided by the inter-eye width so the result is invariant to how close
        the user sits to the camera:

        (a) Head pose — nose tip relative to the inter-eye midpoint. Captures
            head yaw/pitch; stable with a large range. Scaled by
            LASER_EYES_HEAD_GAIN. The frame is already mirrored so +x is screen-
            right as the user sees it; no sign flip is required.

        (b) Iris offset — iris centre minus eye-socket centre, averaged over both
            eyes. Captures fine eye movement within the head pose. Scaled by
            LASER_EYES_IRIS_GAIN.

        The summed offset is clamped to LASER_EYES_GAZE_MAX and EMA-smoothed
        (LASER_EYES_GAZE_SMOOTH). It is NOT re-centred here — the laser effect
        subtracts a per-activation baseline (the gaze at the instant the eyes
        open to fire), so whatever the user looks at when the beam starts becomes
        "centre". Returns the previous value when required landmarks are missing.

        When *frozen* (eyes detected closed — a blink or the charge hold), the
        update is skipped entirely and the last gaze is returned unchanged, so a
        blink can't jerk the aim and the baseline captured at fire-time stays put.
        """
        if frozen:
            return self._prev_gaze.copy()
        try:
            head_ok = (
                _NOSE_TIP_INDEX < num_lm
                and _LEFT_EYE_OUTER < num_lm
                and _RIGHT_EYE_OUTER < num_lm
            )
            if not head_ok:
                return self._prev_gaze.copy()

            # Inter-eye width = face scale → distance-invariant signal.
            eye_l = lm[_LEFT_EYE_OUTER, :2]
            eye_r = lm[_RIGHT_EYE_OUTER, :2]
            eye_mid = (eye_l + eye_r) * 0.5
            eye_w = float(np.linalg.norm(eye_r - eye_l)) + 1e-6

            # ---- (a) Head pose offset (eye-width units) ------------------------
            nose = lm[_NOSE_TIP_INDEX, :2]
            head = ((nose - eye_mid) / eye_w).astype(np.float32)

            # ---- (b) Iris offset (eye-width units) -----------------------------
            iris = np.zeros(2, dtype=np.float32)
            iris_present = num_lm > max(_LEFT_EYE_INDICES[-1], _RIGHT_EYE_INDICES[-1])
            if iris_present:
                left_iris = _mean_pts_2d(lm, _LEFT_EYE_INDICES, num_lm)
                right_iris = _mean_pts_2d(lm, _RIGHT_EYE_INDICES, num_lm)
                left_socket = _mean_pts_2d(lm, _LEFT_EYE_FALLBACK, num_lm)
                right_socket = _mean_pts_2d(lm, _RIGHT_EYE_FALLBACK, num_lm)
                iris = (
                    ((left_iris - left_socket) + (right_iris - right_socket))
                    * 0.5
                    / eye_w
                ).astype(np.float32)

            # ---- Combine: head AND eye both drive the aim ----------------------
            raw = (
                head * config.LASER_EYES_HEAD_GAIN
                + iris * config.LASER_EYES_IRIS_GAIN
            )

            # Clamp magnitude (one bad landmark frame can't fling the dot away).
            mag = float(np.linalg.norm(raw))
            if mag > config.LASER_EYES_GAZE_MAX:
                raw = raw / mag * config.LASER_EYES_GAZE_MAX

            # Frame-rate-correct EMA: LASER_EYES_GAZE_SMOOTH is the per-step factor
            # at the reference cadence; scale it by dt so the feel is stable across
            # frame rates (faster fps → smaller per-frame step, same overall lag).
            ref = config.LASER_EYES_GAZE_SMOOTH
            if dt > 0.0:
                alpha = 1.0 - (1.0 - ref) ** (dt / _GAZE_SMOOTH_REF_DT)
            else:
                alpha = ref
            alpha = float(np.clip(alpha, 0.0, 1.0))
            smoothed = self._prev_gaze * (1.0 - alpha) + raw * alpha
            self._prev_gaze = smoothed.astype(np.float32)
            return self._prev_gaze.copy()

        except Exception:
            log.debug("FaceTracker: gaze estimation failed", exc_info=True)
            return self._prev_gaze.copy()


# ------------------------------------------------------------------
# Module-level helpers (private)
# ------------------------------------------------------------------

def _mean_pts_2d(
    lm: np.ndarray,
    indices: tuple[int, ...],
    num_lm: int,
) -> np.ndarray:
    """Return mean (x, y) of the given landmark *indices* in normalised space.

    Only includes indices that are actually present in *lm*.  Returns (0, 0)
    if no valid indices exist.
    """
    valid = [i for i in indices if i < num_lm]
    if not valid:
        return np.zeros(2, dtype=np.float32)
    pts = lm[valid, :2]
    return np.mean(pts, axis=0).astype(np.float32)


def _read_blendshape(blendshapes: list, name: str) -> float | None:
    """Return the score for *name* from a blendshape category list, or None."""
    for cat in blendshapes:
        try:
            cat_name = cat.category_name if hasattr(cat, "category_name") else str(cat)
            if cat_name == name:
                return float(cat.score) if hasattr(cat, "score") else None
        except Exception:
            continue
    return None


def _eye_center_px(
    lm: np.ndarray,
    num_lm: int,
    iris_indices: tuple[int, ...],
    fallback_indices: tuple[int, ...],
    w: int,
    h: int,
) -> tuple[int, int]:
    """Return the pixel (x, y) of an eye centre.

    Tries iris landmarks first (only available when iris tracking is on),
    falls back to the mesh eye-region landmarks.
    """
    # Try iris landmarks (indices >= 468).
    valid_iris = [i for i in iris_indices if i < num_lm]
    if valid_iris:
        pts = lm[valid_iris, :2]
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        return int(cx * w), int(cy * h)

    # Fallback: standard mesh eye-ring landmarks.
    valid_fb = [i for i in fallback_indices if i < num_lm]
    if valid_fb:
        pts = lm[valid_fb, :2]
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        return int(cx * w), int(cy * h)

    return (0, 0)


def _face_center(
    lm: np.ndarray,
    num_lm: int,
    stable_indices: tuple[int, ...],
) -> np.ndarray:
    """Return mean (x, y) of *stable_indices* landmarks in normalised space."""
    valid = [i for i in stable_indices if i < num_lm]
    if valid:
        pts = lm[valid, :2]
        return np.mean(pts, axis=0).astype(np.float32)
    # Fallback: centroid of all landmarks.
    if num_lm > 0:
        return np.mean(lm[:, :2], axis=0).astype(np.float32)
    return np.array([0.5, 0.5], dtype=np.float32)
