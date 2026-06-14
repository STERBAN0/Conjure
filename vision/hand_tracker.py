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
import os
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from core.state import HandData
from gestures.smoothing import OneEuroFilter


# MediaPipe landmark indices we care about.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)

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
    last_palm: Optional[np.ndarray] = None
    last_t: Optional[float] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_palm_size: Optional[float] = None
    palm_size_velocity: float = 0.0
    last_seen_t: Optional[float] = None

    def reset_tracking(self) -> None:
        self.landmark_filter.reset()
        self.velocity_filter.reset()
        self.palm_size_velocity_filter.reset()
        self.last_palm = None
        self.last_t = None
        self.velocity = np.zeros(2)
        self.last_palm_size = None
        self.palm_size_velocity = 0.0
        self.last_seen_t = None


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
            pass

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
        for lm_set, handed in zip(result.hand_landmarks, result.handedness):
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

            fingers_open = _finger_openness(lm_smoothed)
            spread = _finger_spread(lm_smoothed)
            pinch = float(
                np.linalg.norm(
                    lm_smoothed[THUMB_TIP, :2] - lm_smoothed[INDEX_TIP, :2]
                )
            )

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


def _finger_openness(lm: np.ndarray) -> np.ndarray:
    """Return 5-vector in 0..1 of how extended each finger is.

    For the four non-thumb fingers we compare tip-to-MCP distance against
    a normalising factor (palm width). For the thumb we use the tip-to-IP
    angle relative to the index MCP, since the thumb articulates differently.
    """
    palm_width = np.linalg.norm(lm[INDEX_MCP, :2] - lm[PINKY_MCP, :2]) + 1e-6

    out = np.zeros(5, dtype=np.float32)
    palm_dir = lm[INDEX_MCP, :2] - lm[PINKY_MCP, :2]
    palm_dir /= np.linalg.norm(palm_dir) + 1e-6
    thumb_vec = lm[THUMB_TIP, :2] - lm[THUMB_MCP, :2]
    thumb_vec /= np.linalg.norm(thumb_vec) + 1e-6
    out[0] = float(np.clip(abs(palm_dir[0] * thumb_vec[1] - palm_dir[1] * thumb_vec[0]) * 1.6, 0, 1))

    for i, (tip, mcp) in enumerate(zip(FINGER_TIPS[1:], FINGER_MCPS[1:]), start=1):
        d = np.linalg.norm(lm[tip, :2] - lm[mcp, :2])
        out[i] = float(np.clip((d / palm_width - 0.5) / 1.1, 0, 1))
    return out


def _finger_spread(lm: np.ndarray) -> float:
    """Average gap between adjacent fingertips, normalised by palm width."""
    palm_width = np.linalg.norm(lm[INDEX_MCP, :2] - lm[PINKY_MCP, :2]) + 1e-6
    gaps = [
        np.linalg.norm(lm[FINGER_TIPS[i], :2] - lm[FINGER_TIPS[i + 1], :2])
        for i in range(1, 4)
    ]
    return float(np.clip(np.mean(gaps) / palm_width, 0, 2))
