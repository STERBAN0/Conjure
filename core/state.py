"""Shared, immutable-by-convention data types passed between modules.

All vectors are numpy arrays in normalised image coordinates (0..1) unless
explicitly named *_px (pixels). Keeping data normalised lets every layer
above the tracker stay resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class HandData:
    """One hand at one instant. All values are smoothed by the tracker."""

    label: str                       # "Left" or "Right" (post mirror-fix)
    palm: np.ndarray                 # (x, y) in 0..1
    palm_px: Tuple[int, int]         # pixel coordinates
    velocity: np.ndarray             # (dx, dy) per second, normalised
    fingers_open: np.ndarray         # 5-vector in 0..1, thumb..pinky
    openness: float                  # mean of fingers_open
    spread: float                    # average inter-fingertip distance
    pinch: float                     # thumb-tip to index-tip, 0..1
    landmarks: np.ndarray            # (21, 3) normalised landmarks
    palm_size: float = 0.0           # wrist -> middle MCP, normalised
    palm_size_velocity: float = 0.0  # d(palm_size)/dt; positive = approaching camera
    tracking_confidence: float = 1.0 # MediaPipe handedness score, when available


@dataclass
class FrameState:
    """Everything the rest of the system needs to know about one frame."""

    frame_bgr: np.ndarray            # raw (already-mirrored) webcam frame
    timestamp: float                 # monotonic seconds
    dt: float                        # seconds since previous frame
    hands: list[HandData] = field(default_factory=list)

    def hand(self, label: str) -> Optional[HandData]:
        for h in self.hands:
            if h.label == label:
                return h
        return None


@dataclass
class GestureSignals:
    """Continuous expressive signals. All in 0..1 unless stated otherwise."""

    span: float = 0.0                # current normalised inter-hand distance
    expansion: float = 0.0           # signed rate of change of span (~ -1..1)
    rotation: float = 0.0            # signed angular velocity of hand axis
    grip: float = 0.0                # average grip across visible hands
    motion_energy: float = 0.0       # smoothed kinetic energy of hands
    midpoint: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    midpoint_px: Tuple[int, int] = (0, 0)
    axis_angle: float = 0.0          # radians, hand-to-hand vector angle
    time_scale: float = 1.0          # global slow-mo factor (<=1)
    audio_level: float = 0.0         # 0..1 broadband audio energy
    audio_bands: np.ndarray = field(default_factory=lambda: np.zeros(8))


# -- Ability lifecycle -------------------------------------------------------

# Phases of an ability's life. Effects use these to decide what to draw.
PHASE_IDLE = "idle"
PHASE_CHARGING = "charging"      # pose held, charge ramping
PHASE_ACTIVE = "active"          # post-release "live" phase (e.g. Kamehameha beam)
PHASE_RELEASING = "releasing"    # one-shot release animation, decays out
PHASE_COOLDOWN = "cooldown"      # blocking new abilities briefly


@dataclass
class AbilityState:
    """Snapshot of the router's state for the current frame.

    Effects read this rather than the router directly, so they stay loosely
    coupled and can be unit-tested with a hand-crafted state.
    """

    name: str = ""                   # active ability id, "" when IDLE
    phase: str = PHASE_IDLE
    charge: float = 0.0              # 0..1
    age: float = 0.0                 # seconds since enter
    phase_age: float = 0.0           # seconds since last phase change
    intensity: float = 0.0           # release strength 0..1 (set on release)
    primary_hand: Optional[HandData] = None
    secondary_hand: Optional[HandData] = None

    @property
    def active(self) -> bool:
        return self.phase != PHASE_IDLE
