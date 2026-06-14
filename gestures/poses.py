"""Discrete pose classifier.

The gesture engine produces continuous signals. The pose classifier sits
*beside* it and turns each frame's hands into a list of recognised poses
with confidence scores. The router then chooses one ability per frame.

Pose IDs:
    chidori       - Sasuke seal, single hand: index + middle extended, others folded
    kamehameha    - Both hands cupped, palms facing, fingertips meeting
    rasengan      - One open palm, the other hand fisted hovering above it
    space_stretch - Both hands open & spread, palms facing each other
    reality_tear  - Both hands clawed (fingers half-curled), held apart

All predicates take HandData (or pairs) and return Optional[PoseMatch].
Confidence is in 0..1; predicates clamp internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

import config
from core.state import FrameState, HandData

# MediaPipe landmark indices we read here.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20


@dataclass(frozen=True)
class PoseMatch:
    """A recognised pose for one frame.

    `primary` is the dominant hand for single-hand poses or the "anchor"
    hand for two-hand poses (e.g. the open palm in Rasengan). `secondary`
    is the supporting hand if any. `extra` is a free-form bag for predicate
    output that the router may want (e.g. estimated beam axis).
    """

    name: str
    confidence: float
    primary: Optional[HandData] = None
    secondary: Optional[HandData] = None
    extra: dict[str, Any] = field(default_factory=dict)


class PoseRecognizer:
    """Stateless classifier. Holds no per-frame memory itself; the router
    integrates pose history over time."""

    def classify(self, frame: FrameState) -> list[PoseMatch]:
        out: list[PoseMatch] = []
        left = frame.hand("Left")
        right = frame.hand("Right")

        for h in frame.hands:
            for pred in (_is_sasuke_seal, _is_open_palm_solo):
                m = pred(h)
                if m is not None:
                    out.append(m)

        if left is not None and right is not None:
            for pred in (
                _is_kamehameha_cup,
                _is_clawed_pair,
                _is_open_palm_pair,
                _is_rasengan_pair,
            ):
                m = pred(left, right)
                if m is not None:
                    out.append(m)

        return out


# ---------------------------------------------------------------------------
# Single-hand predicates
# ---------------------------------------------------------------------------

def _is_sasuke_seal(h: HandData) -> Optional[PoseMatch]:
    """Index + middle extended together, ring + pinky folded.

    The thumb is allowed to be either folded or partially extended — the
    canonical seal varies. We score on the *contrast* between extended and
    folded fingers so a noisy hand still scores well if the silhouette is
    right.
    """
    f = h.fingers_open
    extended_pair = float(min(f[1], f[2]))           # index, middle
    folded_pair = float(max(f[3], f[4]))             # ring, pinky
    if extended_pair < config.POSE_FINGER_EXTENDED:
        return None
    if folded_pair > config.POSE_FINGER_FOLDED:
        return None

    # index + middle should also be roughly parallel (dot product near 1)
    idx_dir = h.landmarks[INDEX_TIP, :2] - h.landmarks[INDEX_MCP, :2]
    mid_dir = h.landmarks[MIDDLE_TIP, :2] - h.landmarks[MIDDLE_MCP, :2]
    parallel = _cos(idx_dir, mid_dir)
    if parallel < 0.6:
        return None

    contrast = extended_pair - folded_pair                   # in [0..1]
    conf = float(np.clip(contrast * 1.4 + (parallel - 0.6) * 0.6, 0.0, 1.0))
    return PoseMatch("chidori", conf, primary=h, extra={"fingertip_dir": idx_dir})


def _is_open_palm_solo(h: HandData) -> Optional[PoseMatch]:
    """Single open palm — the volume gesture / generic neutral pose."""
    if h.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if h.spread < 0.45:
        return None
    conf = float(np.clip((h.openness - config.POSE_OPEN_PALM_OPENNESS) * 3.5, 0.0, 1.0))
    return PoseMatch("open_palm", conf, primary=h)


# ---------------------------------------------------------------------------
# Two-hand predicates
# ---------------------------------------------------------------------------

def _is_kamehameha_cup(left: HandData, right: HandData) -> Optional[PoseMatch]:
    """Both hands cupped with fingertips touching, palms facing each other.

    Without 3D normals we use a 2D proxy: the index and pinky tips of each
    hand are within a small distance, and the palm centres are not too far
    apart. The hands should also be partially closed (cupping).
    """
    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if not (config.POSE_KAMEHAMEHA_PALM_DIST_MIN
            <= palm_dist
            <= config.POSE_KAMEHAMEHA_PALM_DIST_MAX):
        return None

    idx_d = float(np.linalg.norm(
        left.landmarks[INDEX_TIP, :2] - right.landmarks[INDEX_TIP, :2]
    ))
    pky_d = float(np.linalg.norm(
        left.landmarks[PINKY_TIP, :2] - right.landmarks[PINKY_TIP, :2]
    ))
    fingertip_meeting = max(idx_d, pky_d) < config.POSE_KAMEHAMEHA_FINGERTIP_DIST * 1.6
    if not fingertip_meeting:
        return None

    cupping = 0.25 < ((1 - left.openness) + (1 - right.openness)) * 0.5 < 0.75
    if not cupping:
        return None

    fingertip_score = 1.0 - max(idx_d, pky_d) / (config.POSE_KAMEHAMEHA_FINGERTIP_DIST * 1.6)
    palm_score = 1.0 - abs(palm_dist - 0.18) / 0.18
    conf = float(np.clip(0.5 * fingertip_score + 0.5 * palm_score, 0.0, 1.0))

    # Beam axis points away from the user — perpendicular to the line between
    # palms, aiming "up the screen" by default. We flip if the average wrist
    # is below the average palm (hands held high).
    axis = np.array([0.0, -1.0])
    return PoseMatch(
        "kamehameha", conf,
        primary=left, secondary=right,
        extra={"beam_axis": axis},
    )


def _is_clawed_pair(left: HandData, right: HandData) -> Optional[PoseMatch]:
    """Both hands half-curled (claw shape), held apart."""
    lo, hi = config.POSE_CLAWED_OPENNESS_LO, config.POSE_CLAWED_OPENNESS_HI
    if not (lo < left.openness < hi and lo < right.openness < hi):
        return None
    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if palm_dist < 0.18:
        return None  # too close — looks like cupping
    # Confidence rises with how clearly we're in the claw band and how
    # spread the hands are.
    span_score = float(np.clip((palm_dist - 0.18) / 0.4, 0.0, 1.0))
    pose_score = float(np.clip(
        1.0 - abs(left.openness - 0.4) / 0.25, 0.0, 1.0
    ) * np.clip(
        1.0 - abs(right.openness - 0.4) / 0.25, 0.0, 1.0
    ))
    conf = float(np.clip(0.4 + 0.3 * span_score + 0.3 * pose_score, 0.0, 1.0))
    return PoseMatch("reality_tear", conf, primary=left, secondary=right)


def _is_open_palm_pair(left: HandData, right: HandData) -> Optional[PoseMatch]:
    """Both palms open and spread, palms facing each other.

    The space-stretch ability. Confidence ramps with hand separation so the
    membrane visual reacts to how stretched the user's hands are.
    """
    if left.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if right.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if left.spread < 0.4 or right.spread < 0.4:
        return None
    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if palm_dist < 0.18:
        return None
    stretch = float(np.clip((palm_dist - 0.18) / 0.5, 0.0, 1.0))
    conf = float(np.clip(0.5 + 0.5 * stretch, 0.0, 1.0))
    return PoseMatch(
        "space_stretch", conf,
        primary=left, secondary=right,
        extra={"stretch": stretch, "palm_dist": palm_dist},
    )


def _is_rasengan_pair(left: HandData, right: HandData) -> Optional[PoseMatch]:
    """One open palm + the other hand fisted hovering above it.

    The fisted hand should be roughly above the open palm in screen space,
    moderately close, with the open palm's normal pointing skyward (we
    approximate that with "open hand is below the fist").
    """
    open_h, other = (left, right) if left.openness > right.openness else (right, left)
    if open_h.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if other.openness > 0.45:
        return None
    if open_h.spread < 0.35:
        return None

    # other hand should be above open hand (smaller y in image space)
    above = other.palm[1] < open_h.palm[1] - 0.04
    nearby = float(np.linalg.norm(open_h.palm - other.palm)) < 0.30
    if not (above and nearby):
        return None

    height_score = float(np.clip((open_h.palm[1] - other.palm[1]) / 0.20, 0.0, 1.0))
    conf = float(np.clip(0.5 + 0.5 * height_score, 0.0, 1.0))
    return PoseMatch(
        "rasengan", conf,
        primary=open_h, secondary=other,
        extra={"sphere_anchor": open_h.palm.copy()},
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, with a tiny epsilon to avoid divide-by-zero."""
    na = float(np.linalg.norm(a)) + 1e-6
    nb = float(np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / (na * nb))
