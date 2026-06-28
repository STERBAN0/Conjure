"""Tests for GestureEngine — the continuous-signal stage of the pipeline.

GestureEngine.update(FrameState) -> GestureSignals

Covered signals:
  - span        : inter-hand distance (normalised)
  - expansion   : signed rate-of-change of span
  - rotation    : signed angular velocity of hand-to-hand vector
  - grip        : average finger-folded-ness
  - motion_energy: smoothed palm speed

Module-level helper also tested:
  - _angle_diff : shortest signed angular difference
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.state import FrameState, HandData
from gestures.engine import GestureEngine, _angle_diff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_frame(t: float = 0.0, dt: float = 1 / 60.0) -> FrameState:
    """Minimal FrameState with no hands — used for testing decay."""
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=t, dt=dt, hands=[])


def _frame_with_hands(
    *hands: HandData,
    t: float = 0.0,
    dt: float = 1 / 60.0,
) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=t, dt=dt, hands=list(hands))


def _make_hand(
    label: str,
    palm_xy: tuple[float, float],
    *,
    openness: float = 0.5,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> HandData:
    """Minimal HandData sufficient for engine signals (no landmark geometry needed)."""
    palm = np.asarray(palm_xy, dtype=np.float32)
    fingers_open = np.full(5, openness, dtype=np.float32)
    return HandData(
        label=label,
        palm=palm,
        palm_px=(int(palm_xy[0] * 1280), int(palm_xy[1] * 720)),
        velocity=np.asarray(velocity, dtype=np.float32),
        fingers_open=fingers_open,
        openness=openness,
        spread=0.4,
        pinch=0.3,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        palm_size=0.15,
    )


# ---------------------------------------------------------------------------
# _angle_diff helper
# ---------------------------------------------------------------------------

class TestAngleDiff:
    """The helper must return the *shortest* signed path between two angles."""

    def test_same_angle_is_zero(self):
        assert _angle_diff(1.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_positive_small_difference(self):
        # b is 0.1 rad behind a — diff should be +0.1
        assert _angle_diff(0.5, 0.4) == pytest.approx(0.1, abs=1e-6)

    def test_negative_small_difference(self):
        assert _angle_diff(0.4, 0.5) == pytest.approx(-0.1, abs=1e-6)

    def test_wrap_near_plus_pi(self):
        # +179° vs −179° is only 2° apart in the positive direction
        a = math.radians(179)
        b = math.radians(-179)
        diff = _angle_diff(a, b)
        assert abs(diff) == pytest.approx(math.radians(2), abs=1e-5)

    def test_wrap_near_minus_pi(self):
        # −179° vs +179° is 2° in the negative direction
        a = math.radians(-179)
        b = math.radians(179)
        diff = _angle_diff(a, b)
        assert abs(diff) == pytest.approx(math.radians(2), abs=1e-5)

    def test_result_always_within_pi(self):
        for deg in range(-360, 361, 45):
            for deg2 in range(-360, 361, 45):
                d = _angle_diff(math.radians(deg), math.radians(deg2))
                assert -math.pi <= d <= math.pi


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------

class TestSpan:
    def test_span_reflects_inter_hand_distance(self):
        """Span is the Euclidean distance between the two palms (normalised)."""
        engine = GestureEngine()
        left = _make_hand("Left", (0.2, 0.5))
        right = _make_hand("Right", (0.8, 0.5))
        sigs = engine.update(_frame_with_hands(left, right, t=0.0))

        # Exact raw distance is 0.6; after one OneEuro step the first sample
        # passes straight through (no previous state), so span == raw distance.
        assert sigs.span == pytest.approx(0.6, abs=0.02)

    def test_span_larger_when_hands_farther_apart(self):
        """Wider separation produces a higher span than closer placement."""
        engine_close = GestureEngine()
        engine_far = GestureEngine()

        left_close = _make_hand("Left", (0.4, 0.5))
        right_close = _make_hand("Right", (0.6, 0.5))
        left_far = _make_hand("Left", (0.1, 0.5))
        right_far = _make_hand("Right", (0.9, 0.5))

        sig_close = engine_close.update(_frame_with_hands(left_close, right_close, t=0.0))
        sig_far = engine_far.update(_frame_with_hands(left_far, right_far, t=0.0))

        assert sig_far.span > sig_close.span

    def test_span_zero_with_no_hands(self):
        """span stays at its initial value (0.0) when no hands are present."""
        engine = GestureEngine()
        sigs = engine.update(_blank_frame(t=0.0))
        assert sigs.span == pytest.approx(0.0, abs=1e-6)

    def test_span_zero_with_single_hand(self):
        """span is not computed for a single hand — must not be set."""
        engine = GestureEngine()
        right = _make_hand("Right", (0.6, 0.5))
        sigs = engine.update(_frame_with_hands(right, t=0.0))
        assert sigs.span == pytest.approx(0.0, abs=1e-6)

    def test_midpoint_set_with_two_hands(self):
        """midpoint should be the average of the two palm positions."""
        engine = GestureEngine()
        left = _make_hand("Left", (0.2, 0.4))
        right = _make_hand("Right", (0.8, 0.6))
        sigs = engine.update(_frame_with_hands(left, right, t=0.0))

        expected_mid = np.array([0.5, 0.5], dtype=np.float32)
        np.testing.assert_allclose(sigs.midpoint, expected_mid, atol=0.01)


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

class TestExpansion:
    def _warm_up(self, engine: GestureEngine, left_x: float, right_x: float) -> None:
        """Feed one priming frame so the engine has a previous-span to diff."""
        left = _make_hand("Left", (left_x, 0.5))
        right = _make_hand("Right", (right_x, 0.5))
        engine.update(_frame_with_hands(left, right, t=0.0))

    def test_expansion_positive_when_hands_move_apart(self):
        """Moving hands apart → expansion > 0."""
        engine = GestureEngine()
        # Frame 1: hands close
        self._warm_up(engine, left_x=0.4, right_x=0.6)
        # Frame 2: hands farther apart
        left2 = _make_hand("Left", (0.2, 0.5))
        right2 = _make_hand("Right", (0.8, 0.5))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))

        assert sigs.expansion > 0.0

    def test_expansion_negative_when_hands_move_together(self):
        """Moving hands together → expansion < 0 (or at most 0)."""
        engine = GestureEngine()
        # Frame 1: hands far apart
        self._warm_up(engine, left_x=0.1, right_x=0.9)
        # Frame 2: hands much closer
        left2 = _make_hand("Left", (0.4, 0.5))
        right2 = _make_hand("Right", (0.6, 0.5))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))

        assert sigs.expansion < 0.0

    def test_expansion_decays_when_hands_disappear(self):
        """After hands vanish expansion multiplies by 0.6 each frame (fast decay)."""
        engine = GestureEngine()
        # Build up a positive expansion over two frames
        self._warm_up(engine, left_x=0.4, right_x=0.6)
        left2 = _make_hand("Left", (0.2, 0.5))
        right2 = _make_hand("Right", (0.8, 0.5))
        sigs_before = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        expansion_before = abs(sigs_before.expansion)

        # Now drop both hands for several frames
        for i in range(5):
            sigs_after = engine.update(_blank_frame(t=(2 + i) / 60.0))

        assert abs(sigs_after.expansion) < expansion_before

    def test_expansion_zero_on_first_two_hand_frame(self):
        """expansion is not defined on the first two-hand frame — remains 0.0."""
        engine = GestureEngine()
        left = _make_hand("Left", (0.3, 0.5))
        right = _make_hand("Right", (0.7, 0.5))
        sigs = engine.update(_frame_with_hands(left, right, t=0.0))
        # No prev_span yet, so the expansion branch is skipped.
        assert sigs.expansion == pytest.approx(0.0, abs=1e-6)

    def test_expansion_clipped_to_minus_three_plus_three(self):
        """expansion is clipped to [-3, 3] even for extreme span jumps."""
        engine = GestureEngine()
        self._warm_up(engine, left_x=0.49, right_x=0.51)
        # Huge jump: from 0.02 span to 0.98 span in one frame at 60 fps
        left2 = _make_hand("Left", (0.01, 0.5))
        right2 = _make_hand("Right", (0.99, 0.5))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        assert -3.0 <= sigs.expansion <= 3.0


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestRotation:
    def _two_frames(
        self,
        engine: GestureEngine,
        left_x1: float, right_x1: float,
        left_y1: float, right_y1: float,
        left_x2: float, right_x2: float,
        left_y2: float, right_y2: float,
    ) -> float:
        """Drive two frames and return the rotation signal from the second."""
        left1 = _make_hand("Left", (left_x1, left_y1))
        right1 = _make_hand("Right", (right_x1, right_y1))
        engine.update(_frame_with_hands(left1, right1, t=0.0))

        left2 = _make_hand("Left", (left_x2, left_y2))
        right2 = _make_hand("Right", (right_x2, right_y2))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        return sigs.rotation

    def test_rotation_zero_when_axis_does_not_rotate(self):
        """Two frames with the same horizontal hand axis → rotation ≈ 0."""
        engine = GestureEngine()
        rot = self._two_frames(
            engine,
            0.3, 0.7, 0.5, 0.5,  # frame 1: left=(0.3,0.5), right=(0.7,0.5)
            0.3, 0.7, 0.5, 0.5,  # frame 2: identical
        )
        assert rot == pytest.approx(0.0, abs=0.05)

    def test_rotation_nonzero_when_axis_tilts(self):
        """Tilting the axis from horizontal to 45° produces a measurable rotation."""
        engine = GestureEngine()
        # Frame 1: horizontal axis
        left1 = _make_hand("Left", (0.3, 0.5))
        right1 = _make_hand("Right", (0.7, 0.5))
        engine.update(_frame_with_hands(left1, right1, t=0.0))

        # Frame 2: rotate ~45° counter-clockwise (right hand moves up)
        left2 = _make_hand("Left", (0.36, 0.64))
        right2 = _make_hand("Right", (0.64, 0.36))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        assert abs(sigs.rotation) > 0.01

    def test_rotation_clipped_to_pi(self):
        """rotation must not exceed ±π radians (allow float32 rounding at boundary)."""
        engine = GestureEngine()
        # Frame 1: axis pointing strongly right
        left1 = _make_hand("Left", (0.01, 0.5))
        right1 = _make_hand("Right", (0.99, 0.5))
        engine.update(_frame_with_hands(left1, right1, t=0.0))

        # Frame 2: axis pointing strongly up (≈90° turn in 1/60s)
        left2 = _make_hand("Left", (0.5, 0.99))
        right2 = _make_hand("Right", (0.5, 0.01))
        sigs = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        # Allow 1e-5 tolerance for float32 → float64 rounding at the ±π boundary.
        assert abs(sigs.rotation) <= math.pi + 1e-5

    def test_rotation_decays_when_hands_disappear(self):
        """Rotation decays by factor 0.6 each frame once hands are gone."""
        engine = GestureEngine()
        # Build a tilted axis
        left1 = _make_hand("Left", (0.3, 0.5))
        right1 = _make_hand("Right", (0.7, 0.5))
        engine.update(_frame_with_hands(left1, right1, t=0.0))

        left2 = _make_hand("Left", (0.36, 0.64))
        right2 = _make_hand("Right", (0.64, 0.36))
        sigs_before = engine.update(_frame_with_hands(left2, right2, t=1 / 60.0))
        rotation_before = abs(sigs_before.rotation)

        if rotation_before < 1e-6:
            pytest.skip("No rotation was produced to decay")

        for i in range(5):
            sigs_after = engine.update(_blank_frame(t=(2 + i) / 60.0))

        assert abs(sigs_after.rotation) < rotation_before


# ---------------------------------------------------------------------------
# Grip
# ---------------------------------------------------------------------------

class TestGrip:
    def test_grip_high_for_closed_fist(self):
        """All fingers closed → grip near 1.0."""
        engine = GestureEngine()
        # Warm up OneEuro with a second frame to get past the first-sample pass-through.
        for i in range(3):
            hand = _make_hand("Right", (0.5, 0.5), openness=0.05)
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert sigs.grip > 0.7

    def test_grip_low_for_open_palm(self):
        """All fingers open → grip near 0.0."""
        engine = GestureEngine()
        for i in range(3):
            hand = _make_hand("Right", (0.5, 0.5), openness=0.95)
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert sigs.grip < 0.3

    def test_grip_midrange_for_half_open(self):
        """Half-open fingers → grip roughly in 0.2–0.8 range."""
        engine = GestureEngine()
        for i in range(3):
            hand = _make_hand("Right", (0.5, 0.5), openness=0.5)
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert 0.2 <= sigs.grip <= 0.8

    def test_grip_decays_toward_zero_when_hands_absent(self):
        """After all hands disappear grip falls (raw_grip → 0.0)."""
        engine = GestureEngine()
        # Prime with a closed fist
        for i in range(3):
            hand = _make_hand("Right", (0.5, 0.5), openness=0.05)
            engine.update(_frame_with_hands(hand, t=i / 60.0))

        grip_primed = engine.signals.grip

        # Drop the hand; raw_grip becomes 0.0 — OneEuro converges toward 0
        for i in range(10):
            sigs = engine.update(_blank_frame(t=(3 + i) / 60.0))

        assert sigs.grip < grip_primed

    def test_grip_averaged_across_both_hands(self):
        """Two hands with different openness → grip is the mean (1-openness)."""
        engine = GestureEngine()
        # Hand with openness=0.0 → contributes 1.0 to raw_grip
        # Hand with openness=1.0 → contributes 0.0 to raw_grip
        # Mean raw_grip = 0.5; after OneEuro warm-up it converges to ≈0.5
        for i in range(5):
            left = _make_hand("Left", (0.3, 0.5), openness=0.0)
            right = _make_hand("Right", (0.7, 0.5), openness=1.0)
            sigs = engine.update(_frame_with_hands(left, right, t=i / 60.0))
        assert 0.3 <= sigs.grip <= 0.7


# ---------------------------------------------------------------------------
# Motion energy
# ---------------------------------------------------------------------------

class TestMotionEnergy:
    def test_motion_energy_zero_with_no_hands(self):
        """No hands → energy EMA input is 0.0 → motion_energy decays to 0."""
        engine = GestureEngine()
        sigs = engine.update(_blank_frame(t=0.0))
        assert sigs.motion_energy == pytest.approx(0.0, abs=1e-6)

    def test_motion_energy_positive_with_fast_hands(self):
        """Fast-moving hands produce motion_energy > 0."""
        engine = GestureEngine()
        for i in range(5):
            hand = _make_hand("Right", (0.5, 0.5), velocity=(3.0, 0.0))
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert sigs.motion_energy > 0.0

    def test_motion_energy_higher_for_faster_hands(self):
        """Faster velocity → higher motion_energy after warm-up."""
        engine_slow = GestureEngine()
        engine_fast = GestureEngine()

        for i in range(5):
            slow_hand = _make_hand("Right", (0.5, 0.5), velocity=(0.1, 0.0))
            fast_hand = _make_hand("Right", (0.5, 0.5), velocity=(2.0, 0.0))
            engine_slow.update(_frame_with_hands(slow_hand, t=i / 60.0))
            engine_fast.update(_frame_with_hands(fast_hand, t=i / 60.0))

        assert engine_fast.signals.motion_energy > engine_slow.signals.motion_energy

    def test_motion_energy_clipped_to_five(self):
        """motion_energy must not exceed the clip ceiling of 5.0."""
        engine = GestureEngine()
        for i in range(5):
            hand = _make_hand("Right", (0.5, 0.5), velocity=(100.0, 100.0))
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert sigs.motion_energy <= 5.0

    def test_motion_energy_decreases_after_hands_stop(self):
        """EMA means energy decays once hands slow to zero velocity."""
        engine = GestureEngine()
        # Build up energy with fast motion
        for i in range(10):
            hand = _make_hand("Right", (0.5, 0.5), velocity=(3.0, 0.0))
            engine.update(_frame_with_hands(hand, t=i / 60.0))

        peak_energy = engine.signals.motion_energy

        # Now freeze the hand (velocity 0)
        for i in range(20):
            hand = _make_hand("Right", (0.5, 0.5), velocity=(0.0, 0.0))
            sigs = engine.update(_frame_with_hands(hand, t=(10 + i) / 60.0))

        assert sigs.motion_energy < peak_energy


# ---------------------------------------------------------------------------
# time_scale
# ---------------------------------------------------------------------------

class TestTimeScale:
    def test_time_scale_always_one(self):
        """time_scale is hardcoded to 1.0 in the current engine implementation."""
        engine = GestureEngine()
        for i in range(3):
            hand = _make_hand("Right", (0.5, 0.5))
            sigs = engine.update(_frame_with_hands(hand, t=i / 60.0))
        assert sigs.time_scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# axis_angle
# ---------------------------------------------------------------------------

class TestAxisAngle:
    def test_axis_angle_horizontal_right(self):
        """Left at x=0.2, right at x=0.8 → diff points right → angle ≈ 0."""
        engine = GestureEngine()
        left = _make_hand("Left", (0.2, 0.5))
        right = _make_hand("Right", (0.8, 0.5))
        sigs = engine.update(_frame_with_hands(left, right, t=0.0))
        assert sigs.axis_angle == pytest.approx(0.0, abs=0.05)

    def test_axis_angle_vertical_up(self):
        """Left at y=0.8 (below), right at y=0.2 (above) → diff points up → angle ≈ -π/2."""
        engine = GestureEngine()
        # diff = right.palm - left.palm = (0,0.2-0.8) = (0,-0.6) → atan2(-0.6,0) = -π/2
        left = _make_hand("Left", (0.5, 0.8))
        right = _make_hand("Right", (0.5, 0.2))
        sigs = engine.update(_frame_with_hands(left, right, t=0.0))
        assert sigs.axis_angle == pytest.approx(-math.pi / 2, abs=0.05)
