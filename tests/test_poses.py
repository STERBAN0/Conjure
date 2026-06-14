"""Unit tests for the discrete pose classifier.

Each test feeds the recognizer a hand-crafted FrameState (built from the
fixtures in conftest.py) and asserts the expected pose is identified
with confidence above the configured match threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.state import FrameState
from gestures.poses import PoseRecognizer


def _frame(hands) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=0.0, dt=1 / 60.0, hands=list(hands))


# ----- Sasuke seal / Chidori ----------------------------------------------

@pytest.mark.unit
def test_sasuke_seal_detected_above_threshold(sasuke_seal):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([sasuke_seal])

    # Act
    matches = recognizer.classify(frame)

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


# ----- Closed fist neutrality ---------------------------------------------

@pytest.mark.unit
def test_closed_fist_does_not_trigger_an_ability(closed_fist):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([closed_fist])

    # Act
    matches = recognizer.classify(frame)

    # Assert
    ability_names = {
        "chidori",
        "kamehameha",
        "rasengan",
        "space_stretch",
        "reality_tear",
    }
    assert not any(m.name in ability_names for m in matches)


@pytest.mark.unit
def test_open_palm_only_matches_neutral_open_palm(open_palm_right):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([open_palm_right])

    # Act
    matches = recognizer.classify(frame)

    # Assert
    assert {m.name for m in matches} == {"open_palm"}


# ----- Kamehameha cup -----------------------------------------------------

@pytest.mark.unit
def test_kamehameha_cup_detected(kamehameha_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = kamehameha_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer.classify(frame)

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
def test_clawed_pair_detected(clawed_pair):
    # Arrange
    recognizer = PoseRecognizer()
    left, right = clawed_pair
    frame = _frame([left, right])

    # Act
    matches = recognizer.classify(frame)

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
    matches = recognizer.classify(frame)

    # Assert
    assert any(m.name == "space_stretch" for m in matches)


# ----- Rasengan -----------------------------------------------------------

@pytest.mark.unit
def test_rasengan_detected_when_fist_above_open_palm(rasengan_pair):
    # Arrange
    recognizer = PoseRecognizer()
    open_palm, fist = rasengan_pair
    frame = _frame([open_palm, fist])

    # Act
    matches = recognizer.classify(frame)

    # Assert
    assert any(m.name == "rasengan" for m in matches)


@pytest.mark.unit
def test_classifier_returns_empty_when_no_hands_visible():
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame([])

    # Act
    matches = recognizer.classify(frame)

    # Assert
    assert matches == []
