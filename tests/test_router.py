"""Tests for the AbilityRouter state machine.

We drive the router with synthetic FrameState/PoseMatch sequences and
assert it transitions through CHARGING -> ACTIVE -> COOLDOWN -> IDLE
correctly. Effects subscribe via hooks; we use a recording HookBus
listener to verify the right events fire in the right order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import config
from core.hooks import HookBus
from core.state import (
    FrameState,
    GestureSignals,
    HandData,
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_IDLE,
    PHASE_RELEASING,
)
from gestures.poses import PoseMatch
from gestures.router import AbilityRouter


@dataclass
class _Recorder:
    events: list[tuple[str, tuple, dict]] = field(default_factory=list)

    def __call__(self, event: str):
        def fn(*args, **kwargs):
            self.events.append((event, args, kwargs))
        return fn


def _attach(hooks: HookBus) -> _Recorder:
    rec = _Recorder()
    for evt in (
        "ability_enter",
        "ability_charge",
        "ability_release",
        "ability_active",
        "ability_exit",
    ):
        hooks.on(evt, rec(evt))
    return rec


def _frame(dt: float = 1 / 60.0, t: float = 0.0) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=t, dt=dt, hands=[])


def _hand_with_velocity(palm_size_velocity: float) -> HandData:
    return HandData(
        label="Right",
        palm=np.array([0.5, 0.5], dtype=np.float32),
        palm_px=(640, 360),
        velocity=np.zeros(2, dtype=np.float32),
        fingers_open=np.array([0.2, 0.95, 0.95, 0.1, 0.1], dtype=np.float32),
        openness=0.46,
        spread=0.4,
        pinch=0.5,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        palm_size=0.15,
        palm_size_velocity=palm_size_velocity,
    )


# --- Idle -----------------------------------------------------------------

@pytest.mark.unit
def test_router_starts_idle():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)

    # Assert
    assert router.state.phase == PHASE_IDLE
    assert router.state.name == ""


@pytest.mark.unit
def test_router_stays_idle_with_no_matches():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)

    # Act
    router.update(_frame(), GestureSignals(), [])

    # Assert
    assert router.state.phase == PHASE_IDLE
    assert rec.events == []


# --- Continuous abilities (no release motion) -----------------------------

@pytest.mark.unit
def test_continuous_ability_enters_charging_then_active():
    # Arrange — space_stretch has no release motion + tiny charge_time
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)
    match = PoseMatch("space_stretch", confidence=0.9)

    # Act — feed the same match for several frames at 60fps
    for _ in range(20):
        router.update(_frame(), GestureSignals(), [match])

    # Assert — should have entered, charged to 1.0, and gone active
    assert router.state.phase == PHASE_ACTIVE
    assert router.state.name == "space_stretch"
    enter_events = [e for e in rec.events if e[0] == "ability_enter"]
    assert len(enter_events) == 1


@pytest.mark.unit
def test_continuous_ability_exits_when_pose_lost():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)
    match = PoseMatch("space_stretch", confidence=0.9)

    # Act — establish active, then drop the pose long enough to exit
    for _ in range(20):
        router.update(_frame(), GestureSignals(), [match])
    assert router.state.phase == PHASE_ACTIVE

    # Drop pose long enough to traverse RELEASING (0.25s) + COOLDOWN
    drop_frames = (
        config.POSE_LOST_GRACE_FRAMES
        + int(0.25 * 60)
        + int(config.ABILITY_COOLDOWN["space_stretch"] * 60)
        + 5
    )
    for _ in range(drop_frames):
        router.update(_frame(), GestureSignals(), [])

    # Assert — went through releasing + cooldown + back to idle
    assert router.state.phase == PHASE_IDLE
    assert any(e[0] == "ability_exit" for e in rec.events)


# --- Discrete abilities (require release motion) --------------------------

@pytest.mark.unit
def test_chidori_charges_but_waits_for_thrust():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    hand = _hand_with_velocity(palm_size_velocity=0.0)  # no thrust yet
    match = PoseMatch("chidori", confidence=0.9, primary=hand)

    # Act — hold for longer than charge_time without thrusting
    n_frames = int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 20
    for _ in range(n_frames):
        router.update(_frame(), GestureSignals(), [match])

    # Assert — fully charged but still in CHARGING (waiting on thrust)
    assert router.state.phase == PHASE_CHARGING
    assert router.state.charge >= 0.99


@pytest.mark.unit
def test_chidori_releases_when_thrust_detected():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)

    no_thrust = _hand_with_velocity(palm_size_velocity=0.0)
    thrust = _hand_with_velocity(palm_size_velocity=config.THRUST_RELEASE_RATE * 1.2)

    # Act — charge fully, then thrust on the next frame
    for _ in range(int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 5):
        match = PoseMatch("chidori", confidence=0.9, primary=no_thrust)
        router.update(_frame(), GestureSignals(), [match])
    thrust_match = PoseMatch("chidori", confidence=0.9, primary=thrust)
    router.update(_frame(), GestureSignals(), [thrust_match])

    # Assert — should have transitioned to ACTIVE and emitted release
    assert router.state.phase == PHASE_ACTIVE
    assert any(e[0] == "ability_release" for e in rec.events)


# --- Cooldown -------------------------------------------------------------

@pytest.mark.unit
def test_router_blocks_new_ability_during_cooldown():
    # Arrange — drive through chidori charge, release, sustain, cooldown
    hooks = HookBus()
    router = AbilityRouter(hooks)
    no_thrust = _hand_with_velocity(palm_size_velocity=0.0)
    thrust = _hand_with_velocity(palm_size_velocity=config.THRUST_RELEASE_RATE * 1.2)

    # Charge to full
    for _ in range(int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 5):
        m = PoseMatch("chidori", confidence=0.9, primary=no_thrust)
        router.update(_frame(), GestureSignals(), [m])
    # Trigger release
    router.update(_frame(),
                  GestureSignals(),
                  [PoseMatch("chidori", confidence=0.9, primary=thrust)])
    # Run through ACTIVE phase
    for _ in range(int(config.ABILITY_ACTIVE_DURATION["chidori"] * 60) + 5):
        router.update(_frame(), GestureSignals(), [])
    # Now in releasing or cooldown
    assert router.state.phase in (PHASE_RELEASING, PHASE_COOLDOWN, PHASE_IDLE)

    # Try to start a new ability immediately — should NOT enter while in cooldown
    if router.state.phase == PHASE_COOLDOWN:
        new_match = PoseMatch("space_stretch", confidence=0.9)
        router.update(_frame(), GestureSignals(), [new_match])
        # Phase should still be cooldown, not charging a new ability
        assert router.state.phase == PHASE_COOLDOWN


# --- Single-active-slot guarantee -----------------------------------------

@pytest.mark.unit
def test_only_one_ability_active_when_multiple_poses_match():
    # Arrange — feed two simultaneous matches
    hooks = HookBus()
    router = AbilityRouter(hooks)

    # Act
    matches = [
        PoseMatch("space_stretch", confidence=0.9),
        PoseMatch("reality_tear", confidence=0.95),
    ]
    for _ in range(15):
        router.update(_frame(), GestureSignals(), matches)

    # Assert — exactly one ability owns the slot, and it's the higher-conf one
    assert router.state.phase != PHASE_IDLE
    assert router.state.name == "reality_tear"
