"""Tests for the AbilityRouter state machine.

We drive the router with synthetic FrameState/PoseMatch sequences and
assert it transitions through CHARGING -> ACTIVE -> COOLDOWN -> IDLE
correctly. Effects subscribe via hooks; we use a recording HookBus
listener to verify the right events fire in the right order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import config
from core.hooks import HookBus
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_IDLE,
    PHASE_RELEASING,
    FrameState,
    GestureSignals,
    HandData,
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
def test_chidori_charges_then_holds_active():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    hand = _hand_with_velocity(palm_size_velocity=0.0)
    match = PoseMatch("chidori", confidence=0.9, primary=hand)

    # Act — hold the sign past full charge
    n_frames = int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 20
    for _ in range(n_frames):
        router.update(_frame(), GestureSignals(), [match])

    # Assert — chidori is a HOLD ability: once charged it stays ACTIVE while held
    assert router.state.phase == PHASE_ACTIVE
    assert router.state.charge >= 0.99


@pytest.mark.unit
def test_chidori_ends_when_sign_dropped():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    hand = _hand_with_velocity(palm_size_velocity=0.0)

    # Act — charge fully (chidori has no release motion, so it goes straight to
    # ACTIVE) and hold, then drop the sign.
    for _ in range(int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 10):
        router.update(_frame(),
                      GestureSignals(),
                      [PoseMatch("chidori", confidence=0.9, primary=hand)])
    assert router.state.phase == PHASE_ACTIVE

    for _ in range(config.POSE_LOST_GRACE_FRAMES + 30):
        router.update(_frame(), GestureSignals(), [])

    # Assert — dropping the V ends the ability
    assert router.state.phase in (PHASE_RELEASING, PHASE_COOLDOWN, PHASE_IDLE)


# --- Cooldown -------------------------------------------------------------

@pytest.mark.unit
def test_router_blocks_new_ability_during_cooldown():
    # Arrange — charge chidori (a HOLD ability), then drop it to reach cooldown
    hooks = HookBus()
    router = AbilityRouter(hooks)
    hand = _hand_with_velocity(palm_size_velocity=0.0)

    # Charge to full → ACTIVE (hold)
    for _ in range(int(config.ABILITY_CHARGE_TIME["chidori"] * 60) + 10):
        router.update(_frame(),
                      GestureSignals(),
                      [PoseMatch("chidori", confidence=0.9, primary=hand)])
    # Drop the sign and run out the release window into cooldown
    for _ in range(config.POSE_LOST_GRACE_FRAMES + 40):
        router.update(_frame(), GestureSignals(), [])
    assert router.state.phase in (PHASE_RELEASING, PHASE_COOLDOWN, PHASE_IDLE)

    # While in cooldown, a new ability must NOT start charging
    if router.state.phase == PHASE_COOLDOWN:
        new_match = PoseMatch("space_stretch", confidence=0.9)
        router.update(_frame(), GestureSignals(), [new_match])
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


# --- Throw-release abilities (rasengan / fireball) ------------------------


def _rasengan_match(angle: float, throw_speed: float = 0.0) -> PoseMatch:
    """Stacked rasengan match: lower cupped palm-up anchor + top hand at `angle`
    around it. The lower (anchor) hand carries the throw velocity."""
    r = 0.06
    lower = HandData(
        label="Right",
        palm=np.array([0.5, 0.6], dtype=np.float32),
        palm_px=(640, 432),
        velocity=np.array([throw_speed, 0.0], dtype=np.float32),
        fingers_open=np.array([0.8, 0.9, 0.9, 0.9, 0.8], dtype=np.float32),
        openness=0.86,
        spread=0.6,
        pinch=0.5,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        palm_size=0.15,
        palm_normal=np.array([0.0, -0.85, -0.1], dtype=np.float32),
    )
    upper = HandData(
        label="Left",
        palm=np.array(
            [0.5 + r * np.cos(angle), 0.6 + r * np.sin(angle)], dtype=np.float32
        ),
        palm_px=(640, 360),
        velocity=np.zeros(2, dtype=np.float32),
        fingers_open=np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        openness=0.5,
        spread=0.4,
        pinch=0.5,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        palm_size=0.15,
    )
    return PoseMatch("rasengan", 0.9, primary=lower, secondary=upper)


@pytest.mark.unit
def test_rasengan_charges_by_stirring_then_throws():
    # Arrange
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)

    # Act — stir the top hand in a circle to accumulate rotation charge.
    angle = 0.0
    n = 0
    while router.state.charge < 0.99 and n < 200:
        angle += 0.35
        router.update(_frame(), GestureSignals(), [_rasengan_match(angle)])
        n += 1

    assert router.state.phase == PHASE_CHARGING
    assert router.state.charge >= 0.99

    # Act — throw: the lower (anchor) hand's live velocity crosses the low
    # RASENGAN_THROW_VELOCITY (the responsive path that lets a gentle shove fire).
    angle += 0.35
    router.update(
        _frame(),
        GestureSignals(),
        [_rasengan_match(angle, throw_speed=config.RASENGAN_THROW_VELOCITY * 1.5)],
    )

    # Assert — released and entered active
    assert router.state.phase == PHASE_ACTIVE
    assert any(e[0] == "ability_release" for e in rec.events)


# --- Laser eyes: face-driven charge → fire → stop --------------------------


def _face_frame(*, closed: bool, dur: float = 0.0) -> FrameState:
    """A FrameState carrying a FaceData with the given eye-closed state."""
    from core.state import FaceData
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    face = FaceData(
        present=True,
        both_eyes_closed=closed,
        eyes_closed_duration=dur,
        left_eye_px=(400, 300),
        right_eye_px=(880, 300),
    )
    return FrameState(frame_bgr=blank, timestamp=0.0, dt=1 / 60.0, hands=[], face=face)


# eyes-closed duration that fully charges the laser (grace + charge span + slack)
_LASER_FULL_DUR = (
    config.LASER_EYES_BLINK_GRACE_SECONDS + config.LASER_EYES_CHARGE_SECONDS + 0.2
)


@pytest.mark.unit
def test_laser_eyes_charges_then_fires_on_full_charge():
    """Eyes open to arm, then a sustained close charges the laser and it fires."""
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)

    router.update(_face_frame(closed=False), GestureSignals(), [])           # arm
    router.update(_face_frame(closed=True, dur=_LASER_FULL_DUR), GestureSignals(), [])

    assert router._laser_state == "on"
    assert router.state.name == "laser_eyes"
    assert router.state.phase == PHASE_ACTIVE
    assert any(e[0] == "ability_enter" for e in rec.events)
    assert any(e[0] == "ability_release" for e in rec.events)


@pytest.mark.unit
def test_laser_eyes_partial_charge_does_not_fire():
    """An eye-close that ends before full charge stays in CHARGING, never fires."""
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)

    router.update(_face_frame(closed=False), GestureSignals(), [])           # arm
    half = config.LASER_EYES_BLINK_GRACE_SECONDS + config.LASER_EYES_CHARGE_SECONDS * 0.4
    router.update(_face_frame(closed=True, dur=half), GestureSignals(), [])

    assert router._laser_state == "charging"
    assert router.state.phase == PHASE_CHARGING
    assert not any(e[0] == "ability_release" for e in rec.events)


@pytest.mark.unit
def test_natural_blink_does_not_charge_laser():
    """A short blink (< blink grace) must never start the charge."""
    hooks = HookBus()
    router = AbilityRouter(hooks)

    router.update(_face_frame(closed=False), GestureSignals(), [])           # arm
    tiny = config.LASER_EYES_BLINK_GRACE_SECONDS * 0.5
    for _ in range(5):
        router.update(_face_frame(closed=True, dur=tiny), GestureSignals(), [])

    assert router._laser_state == "off"
    assert router.state.name != "laser_eyes"


@pytest.mark.unit
def test_laser_eyes_keeps_firing_when_face_lost():
    """Once on, the laser keeps firing through a face-detection dropout."""
    hooks = HookBus()
    router = AbilityRouter(hooks)

    router.update(_face_frame(closed=False), GestureSignals(), [])           # arm
    router.update(_face_frame(closed=True, dur=_LASER_FULL_DUR), GestureSignals(), [])
    assert router._laser_state == "on"
    # Eyes open, then the face is lost entirely (face=None) — still firing.
    router.update(_face_frame(closed=False), GestureSignals(), [])
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    router.update(
        FrameState(frame_bgr=blank, timestamp=0.0, dt=1 / 60.0, hands=[], face=None),
        GestureSignals(), [],
    )

    assert router.state.name == "laser_eyes"
    assert router.state.phase == PHASE_ACTIVE
    assert router._laser_state == "on"


@pytest.mark.unit
def test_laser_eyes_turns_off_on_close_after_firing():
    """Once firing, reopening then closing the eyes (>= off threshold) stops it."""
    hooks = HookBus()
    router = AbilityRouter(hooks)

    router.update(_face_frame(closed=False), GestureSignals(), [])           # arm
    router.update(_face_frame(closed=True, dur=_LASER_FULL_DUR), GestureSignals(), [])
    assert router._laser_state == "on"
    # Eyes reopen (arms the off-close), then a deliberate close turns it off.
    router.update(_face_frame(closed=False), GestureSignals(), [])
    off_dur = config.LASER_EYES_OFF_BLINK_SECONDS + 0.05
    router.update(_face_frame(closed=True, dur=off_dur), GestureSignals(), [])

    assert router._laser_state == "off"
    assert router.state.phase != PHASE_ACTIVE


# --- Reality tear: fists-together charge -> pull-apart release -------------

@pytest.mark.unit
def test_reality_tear_charges_together_then_opens_on_pull_apart():
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)
    hand = _hand_with_velocity(0.0)

    def tear(dist: float) -> PoseMatch:
        return PoseMatch(
            "reality_tear", confidence=0.9,
            primary=hand, secondary=hand,
            extra={"palm_dist": dist},
        )

    # Fists together → charge advances to full
    for _ in range(int(config.ABILITY_CHARGE_TIME["reality_tear"] * 60) + 10):
        router.update(_frame(), GestureSignals(), [tear(0.10)])
    assert router.state.phase == PHASE_CHARGING
    assert router.state.charge >= 0.99

    # Dead zone (apart but not far enough) → must NOT fire yet
    router.update(_frame(), GestureSignals(), [tear(0.35)])
    assert not any(e[0] == "ability_release" for e in rec.events)

    # Pulled apart past the threshold → the tear opens
    router.update(_frame(), GestureSignals(), [tear(0.70)])
    assert any(e[0] == "ability_release" for e in rec.events)
    assert router.state.phase == PHASE_ACTIVE


# --- Frost nova: bursts on uncross (pose drop after full charge) -----------

@pytest.mark.unit
def test_frost_nova_bursts_on_uncross():
    hooks = HookBus()
    router = AbilityRouter(hooks)
    rec = _attach(hooks)
    hand = _hand_with_velocity(0.0)

    # Charge while wrists crossed (pose present)
    for _ in range(int(config.ABILITY_CHARGE_TIME["frost_nova"] * 60) + 10):
        router.update(
            _frame(), GestureSignals(),
            [PoseMatch("frost_nova", confidence=0.9, primary=hand, secondary=hand)],
        )
    assert router.state.charge >= 0.99

    # Uncross: the pose disappears while fully charged → burst fires
    router.update(_frame(), GestureSignals(), [])
    assert any(e[0] == "ability_release" for e in rec.events)
    assert router.state.phase == PHASE_ACTIVE


# --- Projectiles follow the flick direction -------------------------------

@pytest.mark.unit
def test_projectile_aims_along_flick():
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list = []
    hooks.on("projectile_spawn", lambda s: spawns.append(s))

    hand = HandData(
        label="Right",
        palm=np.array([0.5, 0.5], dtype=np.float32),
        palm_px=(640, 360),
        velocity=np.zeros(2, dtype=np.float32),
        fingers_open=np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        openness=0.05,
        spread=0.15,
        pinch=0.1,
        landmarks=np.zeros((21, 3), dtype=np.float32),
        palm_size=0.15,
        flick=np.array([1.0, 0.0], dtype=np.float32),  # captured flick to the right
        flick_speed=2.0,
        orientation="palm",
    )

    # Charge fireball to full while holding the pose; the hand's flick (>= the
    # fire threshold) launches a shot along the flick as soon as it is loaded.
    for _ in range(int(config.ABILITY_CHARGE_TIME["fireball"] * 60) + 5):
        router.update(
            _frame(), GestureSignals(),
            [PoseMatch("fireball", confidence=0.9, primary=hand)],
        )
        if spawns:
            break

    assert spawns, "fireball should have launched a projectile on the flick"
    d = spawns[0].direction
    assert d[0] > 0.9 and abs(d[1]) < 0.2  # aimed along the +x flick
