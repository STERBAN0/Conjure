"""Tests for the fireball repeater in AbilityRouter.

Fireball charges ONCE, then fires on every fast finger flick for as long as the
index-up pose is held (unlimited shots, no re-charge). A flick must clear
FIREBALL_FIRE_FLICK_SPEED to count, so small jitter / slow movement never
misfires. Dropping the pose unloads it without firing.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.hooks import HookBus
from core.state import (
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    FrameState,
    GestureSignals,
    HandData,
    ProjectileSpawn,
)
from gestures.poses import PoseMatch
from gestures.router import AbilityRouter
from tests.conftest import _make_hand

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame(dt: float = 1 / 60.0, t: float = 0.0) -> FrameState:
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    return FrameState(frame_bgr=blank, timestamp=t, dt=dt, hands=[], face=None)


def _fireball_hand(
    flick_speed: float = 0.0, index_tip_speed: float = 0.0
) -> HandData:
    """Index-up fireball hand with an optional captured flick / fingertip flick
    (both pointing to the right). `index_tip_speed` simulates flicking only the
    finger while the palm stays put (zero palm velocity)."""
    return _make_hand(
        label="Right",
        palm_xy=(0.5, 0.5),
        fingers_open=(0.3, 0.95, 0.05, 0.05, 0.05),
        velocity=np.zeros(2, dtype=np.float32),
        flick=np.array([1.0, 0.0], dtype=np.float32),
        flick_speed=flick_speed,
        index_tip_velocity=np.array([index_tip_speed, 0.0], dtype=np.float32),
        orientation="palm",
    )


def _match(hand: HandData) -> PoseMatch:
    return PoseMatch(name="fireball", confidence=1.0, primary=hand)


def _charge_to_full(router: AbilityRouter) -> None:
    """Bring the router to a loaded (fully-charged) fireball, no flick yet."""
    still = _fireball_hand(flick_speed=0.0)
    router.update(_frame(), GestureSignals(), [_match(still)])
    # One big dt ramps the charge straight to 1.0 (still no flick → no shot).
    router.update(
        _frame(dt=config.ABILITY_CHARGE_TIME["fireball"] + 0.05),
        GestureSignals(),
        [_match(still)],
    )
    assert router.state.charge >= 1.0
    assert router.state.phase == PHASE_CHARGING


# ---------------------------------------------------------------------------
# Fires on a fast flick; stays loaded
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_fires_on_fast_flick() -> None:
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    fast = _fireball_hand(flick_speed=config.FIREBALL_FIRE_FLICK_SPEED * 1.5)
    router.update(_frame(), GestureSignals(), [_match(fast)])

    assert len(spawns) == 1
    assert spawns[0].kind == "fireball"
    # Still loaded and holding — not released into ACTIVE/cooldown.
    assert router.state.phase == PHASE_CHARGING


# ---------------------------------------------------------------------------
# Unlimited shots while the pose is held (no re-charge)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_unlimited_shots_while_held() -> None:
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    fast = _fireball_hand(flick_speed=config.FIREBALL_FIRE_FLICK_SPEED * 1.5)

    for _ in range(3):
        # Flick to fire.
        router.update(_frame(), GestureSignals(), [_match(fast)])
        # Let the refire cooldown elapse (no flick) before the next shot.
        router.update(
            _frame(dt=config.FIREBALL_REFIRE_COOLDOWN + 0.02),
            GestureSignals(),
            [_match(_fireball_hand(flick_speed=0.0))],
        )

    assert len(spawns) == 3, f"expected 3 shots without re-charging, got {len(spawns)}"
    assert router.state.phase == PHASE_CHARGING


# ---------------------------------------------------------------------------
# Fires on a pure index-fingertip flick (palm not moving)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_fires_on_fingertip_flick_only() -> None:
    """Flicking just the index finger (zero palm velocity, no captured flick)
    fires the fireball — the trigger reads the fingertip velocity, so you don't
    have to shove the whole hand."""
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    # Palm still + no captured flick, but the fingertip flicks right.
    tip = _make_hand(
        label="Right",
        palm_xy=(0.5, 0.5),
        fingers_open=(0.3, 0.95, 0.05, 0.05, 0.05),
        velocity=np.zeros(2, dtype=np.float32),
        flick=np.zeros(2, dtype=np.float32),
        flick_speed=0.0,
        index_tip_velocity=np.array(
            [config.FIREBALL_FIRE_FLICK_SPEED * 1.5, 0.0], dtype=np.float32
        ),
        orientation="palm",
    )
    router.update(_frame(), GestureSignals(), [_match(tip)])

    assert len(spawns) == 1, "a fingertip-only flick should fire the fireball"
    # And the shot follows the finger (to the right), not straight up.
    assert spawns[0].direction[0] > 0.9


# ---------------------------------------------------------------------------
# One flick = one shot, even though flick_speed stays latched for ~0.40s
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_one_shot_per_flick() -> None:
    """A single flick held latched across many frames fires exactly once.

    flick_speed stays latched for HAND_FLICK_DECAY_SECONDS (0.40s), longer than
    the refire cooldown, so a purely time-based gate double-fires from one flick.
    The edge-trigger must disarm on fire and not re-fire until the flick subsides.
    """
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    fast = _fireball_hand(flick_speed=config.FIREBALL_FIRE_FLICK_SPEED * 2.0)
    # 40 frames @ 1/60 s = 0.67 s, well past the refire cooldown, but it is all
    # one latched flick — the level never drops, so it must not re-arm.
    for _ in range(40):
        router.update(_frame(dt=1 / 60.0), GestureSignals(), [_match(fast)])

    assert len(spawns) == 1, f"one flick must fire exactly one shot, got {len(spawns)}"


# ---------------------------------------------------------------------------
# Slow movement below the threshold must NOT misfire
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_slow_move_does_not_misfire() -> None:
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    slow = _fireball_hand(flick_speed=config.FIREBALL_FIRE_FLICK_SPEED * 0.5)
    for _ in range(30):
        router.update(_frame(), GestureSignals(), [_match(slow)])

    assert spawns == [], "slow movement below the threshold must not fire"


# ---------------------------------------------------------------------------
# Dropping the pose unloads without firing
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_fireball_pose_drop_unloads_without_firing() -> None:
    hooks = HookBus()
    router = AbilityRouter(hooks)
    spawns: list[ProjectileSpawn] = []
    hooks.on("projectile_spawn", spawns.append)

    _charge_to_full(router)
    # Drop the pose entirely for longer than the lost-grace window.
    for _ in range(config.POSE_LOST_GRACE_FRAMES + 3):
        router.update(_frame(), GestureSignals(), [])

    assert spawns == [], "dropping the pose should unload, not fire"
    assert router.state.phase == PHASE_COOLDOWN
