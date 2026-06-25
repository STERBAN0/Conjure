"""Tests for FaceData construction and laser_eyes pose detection.

Covers:
  - FaceData default and explicit construction
  - _raw_matches laser_eyes detection when both eyes are closed
  - _raw_matches returns no laser_eyes when eyes are open
  - _raw_matches returns no laser_eyes when face is absent
  - End-to-end: laser_eyes PoseMatch flows through PoseRecognizer.classify
    after the required hysteresis frames
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.state import FaceData, FrameState
from gestures.poses import PoseMatch, PoseRecognizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_with_face(face: FaceData | None) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=0.0, dt=1 / 60.0, hands=[], face=face)


def _both_eyes_closed() -> FaceData:
    return FaceData(
        present=True,
        both_eyes_closed=True,
        eyes_closed_duration=2.5,
        left_eye_px=(500, 360),
        right_eye_px=(780, 360),
        face_center=np.array([640.0, 360.0], dtype=np.float32),
    )


def _eyes_open() -> FaceData:
    return FaceData(
        present=True,
        both_eyes_closed=False,
        eyes_closed_duration=0.0,
        left_eye_px=(500, 360),
        right_eye_px=(780, 360),
        face_center=np.array([640.0, 360.0], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# FaceData construction tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_face_data_defaults():
    # Arrange / Act
    face = FaceData()

    # Assert
    assert face.present is False
    assert face.both_eyes_closed is False
    assert face.eyes_closed_duration == 0.0
    assert face.left_eye_px == (0, 0)
    assert face.right_eye_px == (0, 0)


@pytest.mark.unit
def test_face_data_explicit_construction():
    # Arrange / Act
    face = _both_eyes_closed()

    # Assert
    assert face.present is True
    assert face.both_eyes_closed is True
    assert face.eyes_closed_duration == pytest.approx(2.5)
    assert face.left_eye_px == (500, 360)
    assert face.right_eye_px == (780, 360)


# ---------------------------------------------------------------------------
# Laser_eyes raw geometry tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_raw_matches_laser_eyes_when_both_eyes_closed(laser_eyes_face: FaceData):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame_with_face(laser_eyes_face)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    laser_matches = [m for m in matches if m.name == "laser_eyes"]
    assert len(laser_matches) == 1
    assert laser_matches[0].confidence >= config.POSE_MATCH_THRESHOLD


@pytest.mark.unit
def test_raw_matches_no_laser_eyes_when_eyes_open(eyes_open_face: FaceData):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame_with_face(eyes_open_face)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    laser_matches = [m for m in matches if m.name == "laser_eyes"]
    assert len(laser_matches) == 0


@pytest.mark.unit
def test_raw_matches_no_laser_eyes_when_no_face():
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame_with_face(None)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert
    laser_matches = [m for m in matches if m.name == "laser_eyes"]
    assert len(laser_matches) == 0


@pytest.mark.unit
def test_raw_matches_no_laser_eyes_when_face_not_present():
    # Arrange — face object exists but present=False
    recognizer = PoseRecognizer()
    absent_face = FaceData(present=False, both_eyes_closed=True)
    frame = _frame_with_face(absent_face)

    # Act
    matches = recognizer._raw_matches(frame)

    # Assert — no laser_eyes without a tracked face
    laser_matches = [m for m in matches if m.name == "laser_eyes"]
    assert len(laser_matches) == 0


# ---------------------------------------------------------------------------
# Hysteresis integration: classify activates laser_eyes after POSE_ENTER_FRAMES
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_activates_laser_eyes_after_hysteresis(laser_eyes_face: FaceData):
    # Arrange
    recognizer = PoseRecognizer()
    frame = _frame_with_face(laser_eyes_face)

    # Act — pump POSE_ENTER_FRAMES frames at raw-conf level (+ a couple extra)
    results: list[list[PoseMatch]] = []
    for _ in range(config.POSE_ENTER_FRAMES + 2):
        results.append(recognizer.classify(frame))

    # Assert — laser_eyes active after the hysteresis window
    final = results[-1]
    active_names = [m.name for m in final]
    assert "laser_eyes" in active_names


@pytest.mark.unit
def test_classify_does_not_activate_laser_eyes_before_hysteresis(laser_eyes_face: FaceData):
    # Arrange — pump fewer frames than POSE_ENTER_FRAMES
    recognizer = PoseRecognizer()
    frame = _frame_with_face(laser_eyes_face)

    # Act — one frame fewer than required
    results: list[list[PoseMatch]] = []
    for _ in range(config.POSE_ENTER_FRAMES - 1):
        results.append(recognizer.classify(frame))

    # Assert — laser_eyes not yet active
    final = results[-1] if results else []
    active_names = [m.name for m in final]
    assert "laser_eyes" not in active_names
