"""Pytest configuration + shared fixtures.

Builds hand-crafted HandData fixtures for the canonical poses Conjure
recognises. Each fixture returns a ``HandData`` (or pair) whose landmark
geometry is intentionally clean so predicates score near 1.0.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.state import FaceData, HandData

# Camera-facing palm_normal: -z points toward camera so -palm_normal[2] > 0.5.
_PALM_TOWARD_CAMERA = np.array([0.0, 0.0, -0.8], dtype=np.float32)
_PALM_AWAY = np.array([0.0, 0.0, 0.8], dtype=np.float32)


def _make_hand(
    label: str,
    palm_xy: tuple[float, float],
    fingers_open: tuple[float, float, float, float, float],
    *,
    spread: float = 0.6,
    pinch: float = 0.5,
    palm_size: float = 0.15,
    palm_normal: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    landmarks: np.ndarray | None = None,
    orientation: str = "edge",
    wrist_angle_deg: float = 0.0,
    flick: np.ndarray | None = None,
    flick_speed: float = 0.0,
    index_tip_velocity: np.ndarray | None = None,
) -> HandData:
    """Build a HandData with synthetic landmark geometry.

    `fingers_open` is thumb..pinky openness in 0..1. We synthesise landmark
    positions in normalised image space such that:
      - WRIST at palm_xy + (0, +0.07)
      - finger MCPs evenly spread around the palm
      - finger tips extended outward from MCPs proportional to openness

    This is enough for predicate logic that reads landmark vectors
    (parallel-finger checks, fingertip distance) to behave correctly.
    """
    if landmarks is None:
        landmarks = _synth_landmarks(palm_xy, fingers_open)

    if palm_normal is None:
        palm_normal = _PALM_TOWARD_CAMERA.copy()

    if velocity is None:
        velocity = np.zeros(2, dtype=np.float32)

    if flick is None:
        flick = np.zeros(2, dtype=np.float32)

    if index_tip_velocity is None:
        index_tip_velocity = np.zeros(2, dtype=np.float32)

    palm = np.asarray(palm_xy, dtype=np.float32)
    return HandData(
        label=label,
        palm=palm,
        palm_px=(int(palm[0] * 1280), int(palm[1] * 720)),
        velocity=velocity,
        fingers_open=np.asarray(fingers_open, dtype=np.float32),
        openness=float(np.mean(fingers_open)),
        spread=spread,
        pinch=pinch,
        landmarks=landmarks,
        palm_size=palm_size,
        palm_size_velocity=0.0,
        palm_normal=palm_normal,
        orientation=orientation,
        wrist_angle_deg=wrist_angle_deg,
        flick=flick,
        flick_speed=flick_speed,
        index_tip_velocity=index_tip_velocity,
    )


def _synth_landmarks(
    palm_xy: tuple[float, float],
    fingers_open: tuple[float, float, float, float, float],
) -> np.ndarray:
    """21 landmark approximation. Sloppy on z but sufficient for 2D predicates."""
    px, py = palm_xy
    lm = np.zeros((21, 3), dtype=np.float32)

    # Wrist below palm centre
    lm[0] = (px, py + 0.07, 0.0)

    # MCPs (5,9,13,17). Fan from index to pinky horizontally near palm centre.
    mcp_indices = [5, 9, 13, 17]   # index, middle, ring, pinky MCPs
    mcp_offsets = [(-0.04, 0.0), (-0.013, -0.005), (0.013, 0.0), (0.04, 0.005)]
    for idx, off in zip(mcp_indices, mcp_offsets, strict=True):
        lm[idx] = (px + off[0], py + off[1], 0.0)

    # Thumb base sits to the side
    lm[1] = (px - 0.06, py + 0.04, 0.0)   # CMC
    lm[2] = (px - 0.07, py + 0.02, 0.0)   # MCP
    lm[3] = (px - 0.075, py, 0.0)         # IP
    # Thumb tip extended along that direction
    thumb_open = fingers_open[0]
    lm[4] = (px - 0.075 - 0.05 * thumb_open, py - 0.02 * thumb_open, 0.0)

    # Other finger tips at openness * length above the MCP
    finger_tips = [8, 12, 16, 20]
    for finger_index, (mcp_idx, tip_idx) in enumerate(zip(mcp_indices, finger_tips, strict=True), start=1):
        mcp = lm[mcp_idx]
        length = 0.04 + 0.07 * fingers_open[finger_index]
        lm[tip_idx] = (mcp[0], mcp[1] - length, 0.0)
        # PIP roughly halfway, slightly bent for low openness
        pip_idx = tip_idx - 2
        bend = (1.0 - fingers_open[finger_index]) * 0.03
        lm[pip_idx] = (mcp[0], mcp[1] - length * 0.5 + bend, 0.0)
        # DIP closer to tip
        dip_idx = tip_idx - 1
        lm[dip_idx] = (mcp[0], mcp[1] - length * 0.8, 0.0)

    return lm


# ----- Common fixtures -----------------------------------------------------

@pytest.fixture
def open_palm_right() -> HandData:
    return _make_hand(
        "Right", palm_xy=(0.7, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
        palm_normal=_PALM_AWAY.copy(),  # no palm-normal constraint on open_palm_solo
    )


@pytest.fixture
def open_palm_left() -> HandData:
    return _make_hand(
        "Left", palm_xy=(0.3, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
        palm_normal=_PALM_AWAY.copy(),
    )


@pytest.fixture
def closed_fist() -> HandData:
    """Closed fist facing the camera — the fist itself IS fireball."""
    return _make_hand(
        "Right", palm_xy=(0.5, 0.4),
        fingers_open=(0.1, 0.05, 0.05, 0.05, 0.05),
        spread=0.2,
        orientation="palm",
    )


@pytest.fixture
def sasuke_seal() -> HandData:
    """Index + middle extended, others folded. Single hand (right)."""
    return _make_hand(
        "Right", palm_xy=(0.6, 0.5),
        fingers_open=(0.2, 0.95, 0.95, 0.1, 0.1),
        spread=0.4,
    )


@pytest.fixture
def kamehameha_pair() -> tuple[HandData, HandData]:
    """Two hands raised together, palms to camera, fingertips meeting at the
    triangle apex between the palms (kamehameha)."""
    left = _make_hand(
        "Left", palm_xy=(0.45, 0.55),
        fingers_open=(0.4, 0.5, 0.5, 0.5, 0.5),
        spread=0.45,
        orientation="palm",
    )
    right = _make_hand(
        "Right", palm_xy=(0.55, 0.55),
        fingers_open=(0.4, 0.5, 0.5, 0.5, 0.5),
        spread=0.45,
        orientation="palm",
    )
    # Hand-craft fingertips so they meet between the palms.
    mid_x = 0.5
    for tip_idx in (4, 8, 12, 16, 20):
        left.landmarks[tip_idx] = (mid_x - 0.01, 0.55, 0.0)
        right.landmarks[tip_idx] = (mid_x + 0.01, 0.55, 0.0)
    return left, right


@pytest.fixture
def reality_tear_pair() -> tuple[HandData, HandData]:
    """Two closed fists held together (reality_tear charge pose)."""
    left = _make_hand(
        "Left", palm_xy=(0.45, 0.5),
        fingers_open=(0.1, 0.1, 0.1, 0.1, 0.1),
        spread=0.2,
    )
    right = _make_hand(
        "Right", palm_xy=(0.55, 0.5),
        fingers_open=(0.1, 0.1, 0.1, 0.1, 0.1),
        spread=0.2,
    )
    return left, right


@pytest.fixture
def open_palm_pair() -> tuple[HandData, HandData]:
    """Both open palms held apart — the restored space_stretch pose.

    Placed at x=0.15 and x=0.85 so palm_dist ≈ 0.70, well above
    SPACE_STRETCH_DIST_MIN (0.18). Open palms (openness ~0.9, spread 0.7) facing
    each other, no charge — the warp grows with the separation.
    """
    left = _make_hand(
        "Left", palm_xy=(0.15, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
        palm_normal=np.array([0.8, 0.0, 0.0], dtype=np.float32),
    )
    right = _make_hand(
        "Right", palm_xy=(0.85, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
        palm_normal=np.array([-0.8, 0.0, 0.0], dtype=np.float32),
    )
    return left, right


@pytest.fixture
def rasengan_pair() -> tuple[HandData, HandData]:
    """Lower cupped palm facing UP + the other hand stacked on top (rasengan).

    The lower (Right) hand is open with its palm to the ceiling (palm_normal
    points up, -y); the upper (Left) hand sits directly above it within a
    sphere's width. The upward normal is what separates rasengan from kamehameha.
    """
    lower = _make_hand(
        "Right", palm_xy=(0.5, 0.6),
        fingers_open=(0.8, 0.9, 0.9, 0.9, 0.8),
        spread=0.6,
        palm_normal=np.array([0.0, -0.85, -0.1], dtype=np.float32),
    )
    upper = _make_hand(
        "Left", palm_xy=(0.5, 0.45),
        fingers_open=(0.5, 0.5, 0.5, 0.5, 0.5),
        spread=0.4,
    )
    return lower, upper


# ----- New single-hand fixtures -------------------------------------------

@pytest.fixture
def rasengan_hand() -> HandData:
    """Single open palm facing UP — the sphere cradled in the hand (rasengan).

    All fingers extended; palm_normal points up (negative y). That upward normal
    is what separates rasengan from time_freeze (palm toward camera).
    """
    return _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.6,
        palm_normal=np.array([0.0, -0.85, -0.1], dtype=np.float32),
    )


@pytest.fixture
def fireball_hand() -> HandData:
    """Index finger pointing up, the other fingers folded (fireball)."""
    return _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.3, 0.95, 0.05, 0.05, 0.05),
        spread=0.3,
        palm_normal=_PALM_TOWARD_CAMERA.copy(),
        orientation="palm",
    )


@pytest.fixture
def time_freeze_hand() -> HandData:
    """Closed fist with the palm facing the camera (time_freeze gesture)."""
    return _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.1, 0.05, 0.05, 0.05, 0.05),
        spread=0.2,
        palm_normal=_PALM_TOWARD_CAMERA.copy(),
        velocity=np.zeros(2, dtype=np.float32),
        orientation="palm",
    )


# ----- New two-hand fixtures ----------------------------------------------

@pytest.fixture
def frost_nova_pair() -> tuple[HandData, HandData]:
    """Crossed wrists: wrists close together, left palm right of right palm."""
    mid_x, mid_y = 0.5, 0.5
    left = _make_hand(
        "Left", palm_xy=(0.58, 0.43),  # left-labelled hand is on the RIGHT side
        fingers_open=(0.5, 0.5, 0.5, 0.5, 0.5),
        spread=0.4,
    )
    right = _make_hand(
        "Right", palm_xy=(0.42, 0.43),  # right-labelled hand is on the LEFT side
        fingers_open=(0.5, 0.5, 0.5, 0.5, 0.5),
        spread=0.4,
    )
    # Force wrists (landmark 0) to overlap closely within FROST_NOVA_WRIST_DIST_MAX=0.18
    left.landmarks[0] = (mid_x + 0.02, mid_y + 0.06, 0.0)
    right.landmarks[0] = (mid_x - 0.02, mid_y + 0.06, 0.0)
    return left, right


# ----- Face fixtures -------------------------------------------------------

@pytest.fixture
def laser_eyes_face() -> FaceData:
    """Face with both eyes closed long enough to pass the blink-grace check.

    eyes_closed_duration must exceed LASER_EYES_BLINK_GRACE_SECONDS to
    distinguish intentional activation from a natural blink.
    """
    return FaceData(
        present=True,
        both_eyes_closed=True,
        eyes_closed_duration=1.0,
        left_eye_px=(400, 300),
        right_eye_px=(880, 300),
    )


@pytest.fixture
def eyes_open_face() -> FaceData:
    """Face present but eyes open — should NOT trigger laser_eyes."""
    return FaceData(
        present=True,
        both_eyes_closed=False,
        eyes_closed_duration=0.0,
        left_eye_px=(400, 300),
        right_eye_px=(880, 300),
    )
