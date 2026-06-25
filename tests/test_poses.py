"""Unit tests for the discrete pose classifier.

Each test feeds the recognizer a hand-crafted FrameState (built from the
fixtures in conftest.py) and asserts the expected pose is identified.

Design notes:
  - Positive geometry tests use ``_raw_matches`` (stateless) so they do not
    depend on the 4-frame hysteresis window.
  - Negative tests (pose should NOT match) use ``classify``: a single frame
    returning empty is sufficient evidence.
  - One dedicated test exercises the full hysteresis path through ``classify``.

Gesture definitions (orientation-sensitive poses read HandData.orientation /
palm_normal; the rest use 2-D finger geometry only):
  fireball   - single hand: index finger pointing up, others folded
  chidori    - single hand: index + middle extended, ring + pinky folded
  time_freeze - single hand: closed fist, palm facing the camera
  rasengan   - two hands: lower cupped palm-up + the other hand stacked on top
  kamehameha - two hands: raised together, fingertips touching at the triangle apex
  space_stretch - two hands: both open palms, pulled apart (no charge)
  reality_tear - two hands: both closed fists bumped together, then pulled apart
  frost_nova - two hands: crossed wrists
  (force_push was removed — it collided with space_stretch.)
  laser_eyes - face: both eyes closed >= LASER_EYES_BLINK_GRACE_SECONDS
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.state import FaceData, FrameState
from gestures.poses import PoseRecognizer


def _frame(hands, *, face: FaceData | None = None) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=0.0, dt=1 / 60.0, hands=list(hands), face=face)


def _pump_classify(recognizer: PoseRecognizer, frame: FrameState, n: int) -> list:
    """Feed the same frame ``n`` times through classify, return last result."""
    result: list = []
    for _ in range(n):
        result = recognizer.classify(frame)
    return result


# ----- Sasuke seal / Chidori ----------------------------------------------

@pytest.mark.unit
def test_sasuke_seal_detected_above_threshold(sasuke_seal):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([sasuke_seal])

    # Act — use stateless raw layer; no hysteresis needed for geometry assertion
    matches = recognizer._raw_matches(frame)

    # Assert
    chidori = [m for m in matches if m.name == "chidori"]
    assert len(chidori) == 1
    assert chidori[0].confidence >= config.POSE_MATCH_THRESHOLD


@pytest.mark.unit
def test_sasuke_seal_not_detected_when_all_fingers_extended(open_palm_right):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([open_palm_right])

    # Act
    matches = recognizer.classify(frame)

    # Assert — open palm should not register as the seal
    assert not any(m.name == "chidori" for m in matches)


# ----- Closed fist = Fireball (no palm_normal requirement) ----------------

@pytest.mark.unit
def test_closed_fist_no_longer_triggers_fireball(closed_fist):
    """Fireball moved to an index-point gesture; a closed fist now fires nothing."""
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([closed_fist]))
    assert not any(m.name == "fireball" for m in matches)


@pytest.mark.unit
def test_index_point_up_triggers_fireball(fireball_hand):
    """Index finger pointing up (others folded) is recognised as fireball."""
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([fireball_hand]))
    assert any(m.name == "fireball" for m in matches)


@pytest.mark.unit
def test_open_palm_only_matches_neutral_open_palm(open_palm_right):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([open_palm_right])

    # Act — raw matches: single open palm with no special geometry
    matches = recognizer._raw_matches(frame)

    # Assert — should appear as open_palm neutral, not any combat ability
    assert any(m.name == "open_palm" for m in matches)
    assert not any(m.name in {"chidori", "rasengan", "fireball"} for m in matches)


# ----- Kamehameha cup -----------------------------------------------------

@pytest.mark.unit
def test_kamehameha_cup_detected(kamehameha_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = kamehameha_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    cups = [m for m in matches if m.name == "kamehameha"]
    assert len(cups) == 1
    assert cups[0].confidence >= config.POSE_MATCH_THRESHOLD


@pytest.mark.unit
def test_kamehameha_not_detected_when_hands_too_far_apart(open_palm_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = open_palm_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer.classify(frame)

    # Assert — open palms held apart shouldn't trigger Kamehameha cup
    assert not any(m.name == "kamehameha" for m in matches)


# ----- Clawed pair / Reality tear -----------------------------------------

@pytest.mark.unit
def test_two_fists_together_detected_as_reality_tear(reality_tear_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = reality_tear_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    tears = [m for m in matches if m.name == "reality_tear"]
    assert len(tears) == 1
    assert tears[0].confidence >= config.POSE_MATCH_THRESHOLD


# ----- Open-palm pair / Space stretch -------------------------------------

@pytest.mark.unit
def test_open_palm_pair_detects_space_stretch(open_palm_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = open_palm_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert any(m.name == "space_stretch" for m in matches)


# ----- Rasengan (two-hand stacked: lower palm-up + hand on top) ------------

@pytest.mark.unit
def test_rasengan_pair_matches_rasengan(rasengan_pair):
    """Lower cupped palm-up + a hand stacked on top is recognised as rasengan."""
    recognizer = PoseRecognizer()
    lower, upper = rasengan_pair
    matches = recognizer._raw_matches(_frame([lower, upper]))
    rasengans = [m for m in matches if m.name == "rasengan"]
    assert len(rasengans) == 1
    assert rasengans[0].confidence >= config.POSE_MATCH_THRESHOLD


# ----- Rasengan is a TWO-hand pose: a single hand never fires it -----------

@pytest.mark.unit
def test_single_open_palm_does_not_match_rasengan(rasengan_hand):
    """Rasengan moved to two hands; a single open palm-up must not fire it."""
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([rasengan_hand]))
    assert not any(m.name == "rasengan" for m in matches)


# ----- Fireball -----------------------------------------------------------

@pytest.mark.unit
def test_fireball_hand_raw_matches(fireball_hand):
    """Closed fist → fireball regardless of palm direction."""
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([fireball_hand])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert any(m.name == "fireball" for m in matches)


@pytest.mark.unit
def test_fireball_hand_does_not_match_rasengan(fireball_hand):
    """Closed fist (all fingers folded) must not score as rasengan (index up)."""
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([fireball_hand])

    # Act
    matches = recognizer._raw_matches(frame)
    names = {m.name for m in matches}

    # fireball should score but not rasengan (index finger folded in fist)
    assert "fireball" in names
    assert "rasengan" not in names


# ----- Time freeze --------------------------------------------------------

@pytest.mark.unit
def test_time_freeze_raw_matches(time_freeze_hand):
    """Closed fist with the palm facing the camera → time_freeze."""
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([time_freeze_hand])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert any(m.name == "time_freeze" for m in matches)


@pytest.mark.unit
def test_open_palm_is_not_time_freeze():
    """An open palm facing the camera must NOT trigger time_freeze any more —
    the gesture is now a closed fist, so simply lifting an open hand is inert."""
    from tests.conftest import _make_hand
    open_hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.8, 0.95, 0.95, 0.95, 0.9),
        spread=0.7,
        palm_normal=np.array([0.0, 0.0, -0.8], dtype=np.float32),
        orientation="palm",
    )
    recognizer = PoseRecognizer()
    frame = _frame([open_hand])
    matches = recognizer._raw_matches(frame)
    assert not any(m.name == "time_freeze" for m in matches)


# ----- Frost nova ---------------------------------------------------------

@pytest.mark.unit
def test_frost_nova_pair_raw_matches(frost_nova_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = frost_nova_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert — crossed wrists → frost_nova
    assert any(m.name == "frost_nova" for m in matches)


# ----- Laser eyes (face-driven) -------------------------------------------

@pytest.mark.unit
def test_laser_eyes_detected_when_both_eyes_closed(laser_eyes_face):
    """Eyes closed long enough (>= grace period) should trigger laser_eyes."""
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([], face=laser_eyes_face)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert any(m.name == "laser_eyes" for m in matches)
    laser = next(m for m in matches if m.name == "laser_eyes")
    assert laser.confidence == pytest.approx(1.0)


@pytest.mark.unit
def test_laser_eyes_not_detected_when_eyes_open(eyes_open_face):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([], face=eyes_open_face)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert not any(m.name == "laser_eyes" for m in matches)


@pytest.mark.unit
def test_laser_eyes_not_detected_when_no_face():
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([])   # face=None by default

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    assert not any(m.name == "laser_eyes" for m in matches)


@pytest.mark.unit
def test_laser_eyes_not_detected_below_grace_period():
    """Eyes closed for only 0.1 s should NOT trigger laser_eyes (below blink grace)."""
    brief_blink = FaceData(
        present=True,
        both_eyes_closed=True,
        eyes_closed_duration=0.1,
        left_eye_px=(400, 300),
        right_eye_px=(880, 300),
    )
    recognizer = PoseRecognizer()
    frame = _frame([], face=brief_blink)
    matches = recognizer._raw_matches(frame)
    assert not any(m.name == "laser_eyes" for m in matches)


# ----- Hysteresis / classify path -----------------------------------------

@pytest.mark.unit
def test_classify_activates_pose_after_enough_frames(sasuke_seal):
    """classify() must NOT activate after a single frame; it needs POSE_ENTER_FRAMES."""
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([sasuke_seal])

    # Act — single frame should not activate
    one_frame = recognizer.classify(frame)
    assert not any(m.name == "chidori" for m in one_frame), (
        "hysteresis should block activation on the first frame"
    )

    # Act — pump enough frames to cross the threshold
    n = config.POSE_ENTER_FRAMES + 2
    result = _pump_classify(recognizer, frame, n)

    # Assert
    assert any(m.name == "chidori" for m in result), (
        f"chidori should be active after {n} frames"
    )


@pytest.mark.unit
def test_classify_returns_empty_when_no_hands_visible():
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([])

    # Act
    matches = recognizer.classify(frame)

    # Assert
    assert matches == []


# ----- Strict single-hand classifier regressions --------------------------
# These lock in the fixes for the dominant real-world bug: relaxed/half-curled
# and spread-out hands used to misfire abilities (fireball fired 217x over a
# 2-minute clip). The classifier now rejects ambiguous hands and gates the fist
# on a compact fingertip spread.


@pytest.mark.unit
def test_spread_out_half_fist_does_not_fire_fireball():
    """A spread-out, half-curled hand must NOT read as a fist.

    Real fists are compact (spread ~0.33); the false positives were spread-out
    transitional hands (spread 1.1-2.0). The spread gate rejects them.
    """
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.3, 0.3, 0.3, 0.3, 0.3),
        spread=1.5,
    )
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([hand]))
    assert not any(m.name == "fireball" for m in matches)


@pytest.mark.unit
def test_ambiguous_transitional_hand_fires_nothing():
    """A hand whose fingers sit in the dead-zone fires no combat pose."""
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.45, 0.45, 0.45, 0.45, 0.45),
        spread=0.5,
    )
    recognizer = PoseRecognizer()
    names = {m.name for m in recognizer._raw_matches(_frame([hand]))}
    assert names.isdisjoint({"fireball", "rasengan", "chidori", "time_freeze"})


@pytest.mark.unit
def test_single_hand_pose_is_mutually_exclusive():
    """At most one combat pose per frame: an index-point is fireball, nothing else."""
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.3, 0.95, 0.05, 0.05, 0.05),
        spread=0.3, orientation="palm",
    )
    recognizer = PoseRecognizer()
    combat = {
        m.name for m in recognizer._raw_matches(_frame([hand]))
    } & {"fireball", "rasengan", "chidori", "time_freeze"}
    assert combat == {"fireball"}


@pytest.mark.unit
def test_rasengan_requires_index_middle_contrast():
    """Index and middle both half-up (no contrast) must not read as rasengan."""
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.2, 0.6, 0.55, 0.1, 0.1),  # index barely beats middle
        spread=0.4,
    )
    recognizer = PoseRecognizer()
    assert not any(
        m.name == "rasengan" for m in recognizer._raw_matches(_frame([hand]))
    )


# ----- Hand-count debounce (one-vs-two-hand flicker) ----------------------
# Real footage flipped the hand count on ~4.4% of frames (phantom 2nd hand /
# momentary drop), whipsawing the classifier between branches. classify()
# debounces the count; _raw_matches stays strict for geometry tests.

_TWO_HAND_POSES = {
    "kamehameha", "space_stretch", "reality_tear", "frost_nova",
}


@pytest.mark.unit
def test_brief_phantom_second_hand_does_not_flip_to_dual(closed_fist, open_palm_left):
    """A single phantom-2nd-hand frame must not switch the branch to dual."""
    recognizer = PoseRecognizer()
    # Establish single mode with a held fist.
    for _ in range(config.POSE_ENTER_FRAMES + 2):
        recognizer.classify(_frame([closed_fist]))
    assert recognizer._hand_mode == "single"

    # One frame with a phantom second hand.
    out = recognizer.classify(_frame([closed_fist, open_palm_left]))
    assert recognizer._hand_mode == "single"
    assert not any(m.name in _TWO_HAND_POSES for m in out)


@pytest.mark.unit
def test_sustained_two_hands_switches_to_dual(kamehameha_pair):
    """Two hands held for the debounce window switches the branch to dual."""
    left, right = kamehameha_pair
    recognizer = PoseRecognizer()
    frame = _frame([left, right])
    for _ in range(config.HAND_COUNT_DEBOUNCE_FRAMES):
        recognizer.classify(frame)
    assert recognizer._hand_mode == "dual"


# ----- Gesture-redesign regressions (live-test reports) --------------------

@pytest.mark.unit
def test_index_middle_v_triggers_chidori(sasuke_seal):
    """A clean V (index+middle up) reads as chidori."""
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([sasuke_seal]))
    assert any(m.name == "chidori" for m in matches)


@pytest.mark.unit
def test_ring_pinky_up_does_not_trigger_chidori():
    """Raising ring+pinky (index+middle folded) must NOT read as chidori.

    Directly encodes the user's report that chidori was firing on the wrong
    fingers; the predicate requires index+middle extended, ring+pinky folded.
    """
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.2, 0.1, 0.1, 0.95, 0.95),  # ring+pinky up, index+middle down
        spread=0.4,
    )
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([hand]))
    assert not any(m.name == "chidori" for m in matches)


@pytest.mark.unit
def test_palm_up_stack_is_rasengan_not_kamehameha(rasengan_pair):
    """The palm-up stack reads as rasengan, not kamehameha (palms-to-camera)."""
    recognizer = PoseRecognizer()
    lower, upper = rasengan_pair
    names = {m.name for m in recognizer._raw_matches(_frame([lower, upper]))}
    assert "rasengan" in names
    assert "kamehameha" not in names


@pytest.mark.unit
def test_palm_to_camera_is_time_freeze_not_rasengan(time_freeze_hand):
    recognizer = PoseRecognizer()
    names = {m.name for m in recognizer._raw_matches(_frame([time_freeze_hand]))}
    assert "time_freeze" in names
    assert "rasengan" not in names


@pytest.mark.unit
def test_edge_on_fist_is_not_fireball():
    """A fist seen edge-on (a resting/idle hand) must not fire fireball."""
    from tests.conftest import _make_hand
    hand = _make_hand(
        "Right", palm_xy=(0.5, 0.5),
        fingers_open=(0.05, 0.05, 0.05, 0.05, 0.05),
        spread=0.15,
        orientation="edge",
    )
    recognizer = PoseRecognizer()
    assert not any(
        m.name == "fireball" for m in recognizer._raw_matches(_frame([hand]))
    )


@pytest.mark.unit
def test_two_fists_apart_still_recognised_as_reality_tear():
    """The pose keeps matching while fists separate, so the router can detect
    the pull-apart (it would lose the hands otherwise)."""
    from tests.conftest import _make_hand
    left = _make_hand("Left", palm_xy=(0.25, 0.5),
                      fingers_open=(0.1, 0.1, 0.1, 0.1, 0.1), spread=0.2)
    right = _make_hand("Right", palm_xy=(0.78, 0.5),
                       fingers_open=(0.1, 0.1, 0.1, 0.1, 0.1), spread=0.2)
    recognizer = PoseRecognizer()
    matches = recognizer._raw_matches(_frame([left, right]))
    tears = [m for m in matches if m.name == "reality_tear"]
    assert len(tears) == 1
    assert tears[0].extra["palm_dist"] >= config.REALITY_TEAR_PULL_APART_DIST


# ----- Disabled abilities (config.DISABLED_ABILITIES) ----------------------
# reality_tear was re-enabled now that force_push is gone and space_stretch went
# back to open palms, so the set is empty and the pose activates through the live
# classify() path again.


@pytest.mark.unit
def test_no_abilities_disabled():
    """The disabled set is empty — every pose is live again (reality_tear back)."""
    assert config.DISABLED_ABILITIES == frozenset()


@pytest.mark.unit
def test_reality_tear_activates_live(reality_tear_pair):
    """Two bumped fists classify as reality_tear and activate through classify()."""
    left, right = reality_tear_pair
    recognizer = PoseRecognizer()
    frame = _frame([left, right])

    raw = {m.name for m in recognizer._raw_matches(frame)}
    assert "reality_tear" in raw

    out = _pump_classify(recognizer, frame, config.POSE_ENTER_FRAMES + 4)
    assert any(m.name == "reality_tear" for m in out)
