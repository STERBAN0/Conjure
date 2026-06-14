"""Pytest configuration + shared fixtures.

Builds hand-crafted HandData fixtures for the canonical poses Aether
recognises. Each fixture returns a ``HandData`` (or pair) whose landmark
geometry is intentionally clean so predicates score near 1.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.state import HandData


def _make_hand(
    label: str,
    palm_xy: tuple[float, float],
    fingers_open: tuple[float, float, float, float, float],
    *,
    spread: float = 0.6,
    pinch: float = 0.5,
    palm_size: float = 0.15,
    landmarks: np.ndarray | None = None,
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

    palm = np.asarray(palm_xy, dtype=np.float32)
    return HandData(
        label=label,
        palm=palm,
        palm_px=(int(palm[0] * 1280), int(palm[1] * 720)),
        velocity=np.zeros(2, dtype=np.float32),
        fingers_open=np.asarray(fingers_open, dtype=np.float32),
        openness=float(np.mean(fingers_open)),
        spread=spread,
        pinch=pinch,
        landmarks=landmarks,
        palm_size=palm_size,
        palm_size_velocity=0.0,
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
    for idx, off in zip(mcp_indices, mcp_offsets):
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
    for finger_index, (mcp_idx, tip_idx) in enumerate(zip(mcp_indices, finger_tips), start=1):
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
    )


@pytest.fixture
def open_palm_left() -> HandData:
    return _make_hand(
        "Left", palm_xy=(0.3, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
    )


@pytest.fixture
def closed_fist() -> HandData:
    return _make_hand(
        "Right", palm_xy=(0.5, 0.4),
        fingers_open=(0.1, 0.05, 0.05, 0.05, 0.05),
        spread=0.2,
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
    """Two cupped hands with fingertips meeting at the centre.

    Built so the index and pinky tips of left and right are within the
    POSE_KAMEHAMEHA_FINGERTIP_DIST tolerance.
    """
    # Mirror the synthetic landmarks so the fingertips approximately meet.
    left = _make_hand(
        "Left", palm_xy=(0.45, 0.55),
        fingers_open=(0.4, 0.5, 0.5, 0.5, 0.5),  # cupped (~half open)
        spread=0.45,
    )
    right = _make_hand(
        "Right", palm_xy=(0.55, 0.55),
        fingers_open=(0.4, 0.5, 0.5, 0.5, 0.5),
        spread=0.45,
    )
    # Hand-craft fingertips so they meet between the palms.
    mid_x = 0.5
    for tip_idx in (4, 8, 12, 16, 20):
        left.landmarks[tip_idx] = (mid_x - 0.01, 0.55, 0.0)
        right.landmarks[tip_idx] = (mid_x + 0.01, 0.55, 0.0)
    return left, right


@pytest.fixture
def clawed_pair() -> tuple[HandData, HandData]:
    """Both hands half-curled, held apart."""
    left = _make_hand(
        "Left", palm_xy=(0.25, 0.5),
        fingers_open=(0.4, 0.4, 0.4, 0.4, 0.4),
        spread=0.4,
    )
    right = _make_hand(
        "Right", palm_xy=(0.75, 0.5),
        fingers_open=(0.4, 0.4, 0.4, 0.4, 0.4),
        spread=0.4,
    )
    return left, right


@pytest.fixture
def open_palm_pair() -> tuple[HandData, HandData]:
    left = _make_hand(
        "Left", palm_xy=(0.3, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
    )
    right = _make_hand(
        "Right", palm_xy=(0.7, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
    )
    return left, right


@pytest.fixture
def rasengan_pair() -> tuple[HandData, HandData]:
    """One open palm at the bottom, one fist hovering above it."""
    open_palm = _make_hand(
        "Right", palm_xy=(0.55, 0.65),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
    )
    fist = _make_hand(
        "Left", palm_xy=(0.55, 0.45),
        fingers_open=(0.1, 0.1, 0.1, 0.1, 0.1),
        spread=0.2,
    )
    return open_palm, fist
