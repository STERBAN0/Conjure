"""Tests for HandData.orientation, wrist_angle_deg, flick, and flick_speed.

These tests exercise the helper functions extracted from vision/hand_tracker.py
directly (no MediaPipe model required) as well as the full _HandState-driven
flick-capture/decay logic via _update_flick.
"""

from __future__ import annotations

import math

import numpy as np

import config
from core.state import HandData
from vision.hand_tracker import (
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
    _finger_openness,
    _hand_orientation,
    _HandState,
    _update_flick,
    _wrist_angle_deg,
)

# ---------------------------------------------------------------------------
# Helper: build a minimal 21×3 landmark array
# ---------------------------------------------------------------------------

def _landmarks_with_wrist_and_middle_mcp(
    wrist_xy: tuple[float, float],
    middle_mcp_xy: tuple[float, float],
) -> np.ndarray:
    """Build a 21×3 landmark array with the two landmarks we care about set."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[WRIST, :2] = wrist_xy
    lm[MIDDLE_MCP, :2] = middle_mcp_xy
    return lm


# ---------------------------------------------------------------------------
# orientation: palm / back / edge
# ---------------------------------------------------------------------------

class TestHandOrientation:
    def test_palm_facing_camera(self) -> None:
        """Large negative z → palm faces the camera."""
        normal = np.array([0.0, 0.0, -0.9], dtype=np.float32)
        assert _hand_orientation(normal) == "palm"

    def test_back_facing_camera(self) -> None:
        """Large positive z → back of hand faces the camera."""
        normal = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        assert _hand_orientation(normal) == "back"

    def test_edge(self) -> None:
        """Negligible z component → edge-on."""
        normal = np.array([0.95, 0.0, 0.05], dtype=np.float32)
        assert _hand_orientation(normal) == "edge"

    def test_boundary_facing_min(self) -> None:
        """Clearly above HAND_ORIENT_FACING_MIN threshold → palm."""
        # Use a value clearly above the threshold to avoid float32 precision issues.
        threshold = config.HAND_ORIENT_FACING_MIN + 0.01
        normal = np.array([0.0, 0.0, -threshold], dtype=np.float32)
        assert _hand_orientation(normal) == "palm"

    def test_boundary_back_min(self) -> None:
        """Clearly above HAND_ORIENT_BACK_MIN threshold → back."""
        # Use a value clearly above the threshold to avoid float32 precision issues.
        threshold = config.HAND_ORIENT_BACK_MIN + 0.01
        normal = np.array([0.0, 0.0, threshold], dtype=np.float32)
        assert _hand_orientation(normal) == "back"

    def test_below_both_thresholds_is_edge(self) -> None:
        """Values below both thresholds fall into edge."""
        small = config.HAND_ORIENT_FACING_MIN * 0.5
        normal = np.array([0.0, 0.0, -small], dtype=np.float32)
        assert _hand_orientation(normal) == "edge"

    def test_zero_normal_is_edge(self) -> None:
        """Zero normal (degenerate landmarks) must not crash and returns edge."""
        normal = np.zeros(3, dtype=np.float32)
        assert _hand_orientation(normal) == "edge"

    def test_orientation_field_on_handdata(self) -> None:
        """HandData exposes orientation as a plain str field."""
        h = HandData(
            label="Right",
            palm=np.zeros(2),
            palm_px=(0, 0),
            velocity=np.zeros(2),
            fingers_open=np.zeros(5),
            openness=0.0,
            spread=0.0,
            pinch=0.0,
            landmarks=np.zeros((21, 3)),
            orientation="palm",
        )
        assert h.orientation == "palm"


# ---------------------------------------------------------------------------
# wrist_angle_deg
# ---------------------------------------------------------------------------

class TestWristAngleDeg:
    def test_pointing_straight_up(self) -> None:
        """Wrist below middle MCP (y decreasing) → ~0 degrees."""
        lm = _landmarks_with_wrist_and_middle_mcp(
            wrist_xy=(0.5, 0.7),
            middle_mcp_xy=(0.5, 0.5),   # MCP directly above wrist
        )
        angle = _wrist_angle_deg(lm)
        assert abs(angle) < 5.0, f"Expected ~0°, got {angle}"

    def test_pointing_horizontally_right(self) -> None:
        """MCP to the right of wrist → ~90 degrees."""
        lm = _landmarks_with_wrist_and_middle_mcp(
            wrist_xy=(0.3, 0.5),
            middle_mcp_xy=(0.5, 0.5),   # MCP directly right of wrist
        )
        angle = _wrist_angle_deg(lm)
        assert abs(angle - 90.0) < 5.0, f"Expected ~90°, got {angle}"

    def test_pointing_horizontally_left(self) -> None:
        """MCP to the left of wrist → also ~90 degrees (absolute value)."""
        lm = _landmarks_with_wrist_and_middle_mcp(
            wrist_xy=(0.7, 0.5),
            middle_mcp_xy=(0.5, 0.5),   # MCP directly left of wrist
        )
        angle = _wrist_angle_deg(lm)
        assert abs(angle - 90.0) < 5.0, f"Expected ~90°, got {angle}"

    def test_pointing_straight_down(self) -> None:
        """MCP below wrist → ~180 degrees."""
        lm = _landmarks_with_wrist_and_middle_mcp(
            wrist_xy=(0.5, 0.3),
            middle_mcp_xy=(0.5, 0.5),   # MCP below wrist
        )
        angle = _wrist_angle_deg(lm)
        assert abs(angle - 180.0) < 5.0, f"Expected ~180°, got {angle}"

    def test_degenerate_zero_vector_returns_zero(self) -> None:
        """WRIST == MIDDLE_MCP → zero vector → 0.0 without crash."""
        lm = _landmarks_with_wrist_and_middle_mcp(
            wrist_xy=(0.5, 0.5),
            middle_mcp_xy=(0.5, 0.5),
        )
        angle = _wrist_angle_deg(lm)
        assert angle == 0.0

    def test_range_0_to_180(self) -> None:
        """Angle is always in [0, 180]."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            pts = rng.uniform(0.1, 0.9, size=(2, 2)).astype(np.float32)
            lm = _landmarks_with_wrist_and_middle_mcp(
                wrist_xy=(float(pts[0, 0]), float(pts[0, 1])),
                middle_mcp_xy=(float(pts[1, 0]), float(pts[1, 1])),
            )
            angle = _wrist_angle_deg(lm)
            assert 0.0 <= angle <= 180.0, f"Out-of-range angle: {angle}"


# ---------------------------------------------------------------------------
# flick capture / decay via _update_flick
# ---------------------------------------------------------------------------

class TestFlickCapture:
    def _make_state(self) -> _HandState:
        return _HandState()

    def _push_velocity(
        self,
        state: _HandState,
        t: float,
        vel: tuple[float, float],
    ) -> None:
        """Inject a velocity sample and call _update_flick."""
        state.velocity = np.array(vel, dtype=np.float32)
        _update_flick(state, t)

    # -- Basic capture --

    def test_fast_rightward_flick_captured(self) -> None:
        """A velocity clearly above HAND_FLICK_MIN_SPEED is captured."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 2.0
        self._push_velocity(state, t=1.0, vel=(speed, 0.0))

        assert state.flick_speed >= config.HAND_FLICK_MIN_SPEED
        assert abs(state.flick_dir[0] - 1.0) < 1e-4, "Direction should be rightward"
        assert abs(state.flick_dir[1]) < 1e-4

    def test_slow_motion_does_not_capture(self) -> None:
        """Velocity below HAND_FLICK_MIN_SPEED must not set a flick."""
        state = self._make_state()
        slow_speed = config.HAND_FLICK_MIN_SPEED * 0.5
        self._push_velocity(state, t=1.0, vel=(slow_speed, 0.0))

        assert state.flick_speed == 0.0
        assert np.allclose(state.flick_dir, 0.0)

    def test_downward_flick_direction(self) -> None:
        """Downward flick (positive y in image space) captured correctly."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 3.0
        self._push_velocity(state, t=1.0, vel=(0.0, speed))

        assert state.flick_speed >= config.HAND_FLICK_MIN_SPEED
        assert abs(state.flick_dir[1] - 1.0) < 1e-4, "Direction should be downward"

    def test_flick_unit_length(self) -> None:
        """Captured flick direction must be a unit vector."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 2.5
        self._push_velocity(state, t=1.0, vel=(speed * 0.6, speed * 0.8))

        mag = float(np.linalg.norm(state.flick_dir))
        assert abs(mag - 1.0) < 1e-5, f"Expected unit vector, magnitude={mag}"

    # -- Persistence: flick stays valid while within decay window --

    def test_flick_persists_after_hand_stills(self) -> None:
        """After the hand stops, the flick stays valid until HAND_FLICK_DECAY_SECONDS."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 3.0
        t0 = 0.0
        self._push_velocity(state, t=t0, vel=(speed, 0.0))

        # Feed still frames within the decay window.
        half_decay = config.HAND_FLICK_DECAY_SECONDS * 0.5
        for i in range(5):
            t_now = t0 + half_decay * (i + 1) / 5
            self._push_velocity(state, t=t_now, vel=(0.0, 0.0))

        # Flick should still be alive (capture was recent enough).
        assert state.flick_speed > 0.0, "Flick should persist within decay window"

    def test_flick_decays_to_zero_after_decay_seconds(self) -> None:
        """After HAND_FLICK_DECAY_SECONDS of stillness, flick resets to zero."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 3.0
        t0 = 0.0
        self._push_velocity(state, t=t0, vel=(speed, 0.0))

        # Advance well past the decay window.
        t_after = t0 + config.HAND_FLICK_DECAY_SECONDS + 0.1
        self._push_velocity(state, t=t_after, vel=(0.0, 0.0))

        assert state.flick_speed == 0.0, "Flick should have decayed to 0"
        assert np.allclose(state.flick_dir, 0.0), "Flick direction should be zero"

    # -- Peak selection: the strongest movement in the window wins --

    def test_peak_within_window_wins(self) -> None:
        """If a stronger flick happened earlier in the window, it is chosen."""
        state = self._make_state()
        fast = config.HAND_FLICK_MIN_SPEED * 4.0
        slow = config.HAND_FLICK_MIN_SPEED * 1.2
        # Fast upward at t=0.
        self._push_velocity(state, t=0.0, vel=(0.0, -fast))  # upward (negative y)
        # Slow rightward just after.
        self._push_velocity(state, t=0.05, vel=(slow, 0.0))

        # The captured flick should point upward (the faster sample).
        assert state.flick_dir[1] < 0.0, "Expected upward direction (negative y)"

    def test_old_samples_trimmed_from_history(self) -> None:
        """Samples older than HAND_FLICK_HISTORY_SECONDS are pruned."""
        state = self._make_state()
        speed = config.HAND_FLICK_MIN_SPEED * 3.0
        # Inject a rightward flick at t=0.
        self._push_velocity(state, t=0.0, vel=(speed, 0.0))

        # Advance well past the history window with no new fast movement.
        t_late = config.HAND_FLICK_HISTORY_SECONDS + 0.05
        for i in range(3):
            self._push_velocity(state, t=t_late + i * 0.01, vel=(0.0, 0.0))

        # The old fast sample is now outside the history window.
        # If flick_captured_t is also past the decay window, flick should be 0.
        # (In practice the decay window >= history window, so we just confirm
        # the history is pruned — the captured snapshot may still be alive.)
        old_samples_in_window = [
            ts for ts, _ in state.flick_history if ts < 0.01
        ]
        assert len(old_samples_in_window) == 0, "Old sample should be pruned from history"

    # -- HandData fields populated --

    def test_handdata_flick_field_default(self) -> None:
        """HandData.flick defaults to a zero 2-vector."""
        h = HandData(
            label="Left",
            palm=np.zeros(2),
            palm_px=(0, 0),
            velocity=np.zeros(2),
            fingers_open=np.zeros(5),
            openness=0.0,
            spread=0.0,
            pinch=0.0,
            landmarks=np.zeros((21, 3)),
        )
        assert h.flick.shape == (2,)
        assert np.allclose(h.flick, 0.0)
        assert h.flick_speed == 0.0

    def test_handdata_flick_field_custom(self) -> None:
        """HandData.flick can be set explicitly."""
        flick_dir = np.array([1.0, 0.0], dtype=np.float32)
        h = HandData(
            label="Right",
            palm=np.zeros(2),
            palm_px=(0, 0),
            velocity=np.zeros(2),
            fingers_open=np.zeros(5),
            openness=0.0,
            spread=0.0,
            pinch=0.0,
            landmarks=np.zeros((21, 3)),
            flick=flick_dir,
            flick_speed=1.5,
        )
        assert np.allclose(h.flick, [1.0, 0.0])
        assert math.isclose(h.flick_speed, 1.5)


# ---------------------------------------------------------------------------
# _finger_openness — sign of the PIP joint-angle mapping
# ---------------------------------------------------------------------------

_FINGER_JOINTS = (
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
)
_FINGER_X = (0.30, 0.45, 0.55, 0.70)


def _hand_landmarks(*, curled: bool) -> np.ndarray:
    """21×3 landmarks with all four non-thumb fingers straight or curled.

    Each finger sits in its own x column. MCP is below the PIP; the TIP either
    continues straight up (extended → interior PIP angle ≈ 180°) or folds back
    down toward the MCP (curled → interior angle ≈ 0°).
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    for x, (mcp, pip, tip) in zip(_FINGER_X, _FINGER_JOINTS, strict=True):
        lm[mcp, :2] = (x, 0.60)
        lm[pip, :2] = (x, 0.50)
        lm[tip, :2] = (x, 0.55 if curled else 0.40)
    return lm


class TestFingerOpenness:
    def test_extended_fingers_read_high(self) -> None:
        """Straight fingers map to ~1.0 (open)."""
        out = _finger_openness(_hand_landmarks(curled=False))
        assert all(out[i] > 0.8 for i in range(1, 5)), out

    def test_curled_fingers_read_low(self) -> None:
        """A fist (curled fingers) maps to ~0.0 (folded), not ~1.0.

        Regression: an earlier version measured the deflection angle
        (MCP->PIP vs PIP->TIP) against an interior-angle mapping, which inverted
        the result — a closed fist read as a fully-open palm.
        """
        out = _finger_openness(_hand_landmarks(curled=True))
        assert all(out[i] < 0.2 for i in range(1, 5)), out

    def test_open_hand_clearly_more_open_than_fist(self) -> None:
        """Mean openness of an open hand far exceeds that of a fist."""
        open_mean = float(np.mean(_finger_openness(_hand_landmarks(curled=False))[1:]))
        fist_mean = float(np.mean(_finger_openness(_hand_landmarks(curled=True))[1:]))
        assert open_mean > fist_mean + 0.5


def _thumb_landmarks(*, extended: bool) -> np.ndarray:
    """21×3 landmarks exercising only the thumb metric.

    The pinky MCP anchor sits on the right of the palm; the thumb is on the left.
    Extended: the tip reaches further left, well past the IP joint (ratio > 1).
    Tucked: the tip swings back toward the pinky side, closer than the IP (ratio < 1).
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[PINKY_MCP, :2] = (0.70, 0.60)   # anchor (far palm corner)
    lm[THUMB_IP, :2] = (0.40, 0.55)
    lm[THUMB_TIP, :2] = (0.30, 0.50) if extended else (0.55, 0.55)
    return lm


class TestThumbOpenness:
    """Regression for the inverted-thumb bug: a fist's thumb read 'extended'
    (green) and a splayed thumb read 'folded' (red). The distance-ratio metric
    fixes the direction."""

    def test_extended_thumb_reads_open(self) -> None:
        out = _finger_openness(_thumb_landmarks(extended=True))
        assert out[0] > 0.8, out[0]

    def test_tucked_thumb_reads_folded(self) -> None:
        out = _finger_openness(_thumb_landmarks(extended=False))
        assert out[0] < 0.2, out[0]

    def test_extended_thumb_more_open_than_tucked(self) -> None:
        ext = float(_finger_openness(_thumb_landmarks(extended=True))[0])
        tuck = float(_finger_openness(_thumb_landmarks(extended=False))[0])
        assert ext > tuck + 0.5
