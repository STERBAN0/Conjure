"""Gesture engine: turn raw HandData into expressive, temporally-aware signals.

The engine reads a FrameState every tick and updates a single GestureSignals
object with continuous, smoothed measurements:

    span         - normalised inter-hand distance
    expansion    - signed time-derivative of span
    rotation     - signed angular velocity of the hand-to-hand vector
    grip         - average finger-folded-ness
    motion_energy- smoothed sum of palm speeds
    time_scale   - inverse of motion energy (slow hands -> slow time)

Discrete pose recognition and ability charge live in `gestures.poses` and
`gestures.router` respectively; this module is intentionally signal-only.

Smoothing happens in two layers: per-landmark in the tracker, then per-signal
here. Velocities use a slightly looser filter to keep responsiveness for
fast movements.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

import config
from core.state import FrameState, GestureSignals
from gestures.smoothing import EMA, OneEuroFilter


class GestureEngine:
    def __init__(self) -> None:
        self.signals = GestureSignals()

        self._span_f = OneEuroFilter(mincutoff=1.5, beta=0.02)
        self._expansion_f = OneEuroFilter(mincutoff=2.0, beta=0.1)
        self._rotation_f = OneEuroFilter(mincutoff=2.0, beta=0.1)
        self._grip_f = OneEuroFilter(mincutoff=2.0, beta=0.05)
        self._motion_ema = EMA(alpha=1.0 - config.MOTION_ENERGY_DECAY)

        self._prev_span: Optional[float] = None
        self._prev_angle: Optional[float] = None
        self._prev_t: Optional[float] = None

    def update(self, frame: FrameState) -> GestureSignals:
        s = self.signals
        t = frame.timestamp
        h, w = frame.frame_bgr.shape[:2]

        left = frame.hand("Left")
        right = frame.hand("Right")
        present = [hand for hand in (left, right) if hand is not None]

        # Motion energy across visible hands.
        if present:
            energy = float(sum(np.linalg.norm(hand.velocity) for hand in present))
        else:
            energy = 0.0
        s.motion_energy = float(np.clip(self._motion_ema(energy), 0.0, 5.0))

        # Grip = mean (1 - openness).
        if present:
            raw_grip = float(np.mean([1.0 - hand.openness for hand in present]))
        else:
            raw_grip = 0.0
        s.grip = float(np.clip(self._grip_f(raw_grip, t), 0.0, 1.0))

        # Two-handed signals.
        if left is not None and right is not None:
            mid = (left.palm + right.palm) * 0.5
            s.midpoint = mid
            s.midpoint_px = (int(mid[0] * w), int(mid[1] * h))

            diff = right.palm - left.palm
            raw_span = float(np.linalg.norm(diff))
            s.span = float(self._span_f(raw_span, t))

            angle = float(math.atan2(diff[1], diff[0]))
            s.axis_angle = angle

            if self._prev_span is not None and self._prev_t is not None:
                dt = max(1e-3, t - self._prev_t)
                raw_expansion = (
                    (s.span - self._prev_span) / dt * config.SPAN_EXPANSION_GAIN
                )
                s.expansion = float(np.clip(
                    self._expansion_f(raw_expansion, t), -3.0, 3.0
                ))
                if self._prev_angle is not None:
                    da = _angle_diff(angle, self._prev_angle) / dt
                    s.rotation = float(np.clip(
                        self._rotation_f(da, t), -math.pi, math.pi
                    ))
            self._prev_span = s.span
            self._prev_angle = angle
        else:
            # Reset two-handed history; decay derived signals fast so they
            # don't bleed into the next two-hand session.
            self._prev_span = None
            self._prev_angle = None
            s.expansion *= 0.6
            s.rotation *= 0.6
            mid = present[0].palm if present else np.array([0.5, 0.5])
            s.midpoint = mid
            s.midpoint_px = (int(mid[0] * w), int(mid[1] * h))

        # Default time scale remains explicit for effects that peek at it.
        s.time_scale = 1.0

        self._prev_t = t
        return s


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles in radians."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d
