"""Tests for ProjectileField — always-on effect that manages in-flight projectiles.

We exercise the spawn → update → burst lifecycle without touching render(),
which requires a pygame Surface and is outside the scope of unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from core.hooks import HookBus
from core.state import PHASE_IDLE, AbilityState, GestureSignals, ProjectileSpawn
from effects.projectiles import ProjectileField

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signals() -> GestureSignals:
    return GestureSignals()


def _make_ability() -> AbilityState:
    return AbilityState(name="", phase=PHASE_IDLE, charge=0.0)


def _make_field(width: int = 1280, height: int = 720) -> tuple[ProjectileField, HookBus]:
    """Return a fresh (ProjectileField, HookBus) pair."""
    hooks = HookBus()
    field = ProjectileField(width, height, hooks)
    return field, hooks


def _spawn(
    hooks: HookBus,
    *,
    kind: str = "rasengan",
    origin: tuple[float, float] = (640.0, 360.0),
    direction: tuple[float, float] = (1.0, 0.0),
    speed_px: float | None = None,
    radius_px: float | None = None,
) -> None:
    """Emit a projectile_spawn event through the bus."""
    if speed_px is None:
        speed_px = config.PROJECTILE_RASENGAN_SPEED_PX
    if radius_px is None:
        radius_px = config.PROJECTILE_RASENGAN_RADIUS_PX
    spawn = ProjectileSpawn(
        kind=kind,
        origin_px=origin,
        direction=np.array(direction, dtype=np.float32),
        speed_px=speed_px,
        intensity=1.0,
        radius_px=radius_px,
    )
    hooks.emit("projectile_spawn", spawn)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_spawn_adds_projectile_to_list():
    # Arrange
    field, hooks = _make_field()

    # Act
    _spawn(hooks)

    # Assert
    assert len(field._projectiles) == 1
    assert field._projectiles[0].kind == "rasengan"


@pytest.mark.unit
def test_spawn_multiple_projectiles_accumulate():
    # Arrange
    field, hooks = _make_field()

    # Act
    _spawn(hooks, kind="rasengan")
    _spawn(hooks, kind="fireball")
    _spawn(hooks, kind="rasengan")

    # Assert
    assert len(field._projectiles) == 3


@pytest.mark.unit
def test_update_advances_projectile_position():
    # Arrange
    field, hooks = _make_field()
    _spawn(hooks, origin=(0.0, 360.0), direction=(1.0, 0.0),
           speed_px=config.PROJECTILE_RASENGAN_SPEED_PX)
    initial_x = float(field._projectiles[0].pos[0])

    # Act — one frame at dt=1/60
    dt = 1.0 / 60.0
    field.update(_make_signals(), dt, _make_ability())

    # Assert — x advanced by speed * dt
    expected_x = initial_x + config.PROJECTILE_RASENGAN_SPEED_PX * dt
    assert abs(float(field._projectiles[0].pos[0]) - expected_x) < 0.01


@pytest.mark.unit
def test_projectile_removed_and_burst_spawned_when_past_right_edge():
    # Arrange — spawn at a position that will cross the right edge in one big step
    width, height = 1280, 720
    field, hooks = _make_field(width, height)
    margin = config.PROJECTILE_EDGE_MARGIN_PX
    # Place the projectile just inside the margin on the right
    start_x = float(width) + margin - 1.0
    _spawn(hooks, origin=(start_x, float(height) / 2.0),
           direction=(1.0, 0.0), speed_px=config.PROJECTILE_RASENGAN_SPEED_PX)

    # Act — advance one large dt so it crosses the edge
    field.update(_make_signals(), 0.1, _make_ability())

    # Assert — projectile gone, burst particles created
    assert len(field._projectiles) == 0
    assert len(field._bursts) > 0


@pytest.mark.unit
def test_projectile_removed_when_past_left_edge():
    # Arrange — spawn moving left, already past the left margin
    field, hooks = _make_field()
    margin = config.PROJECTILE_EDGE_MARGIN_PX
    start_x = -margin + 1.0
    _spawn(hooks, origin=(start_x, 360.0),
           direction=(-1.0, 0.0), speed_px=config.PROJECTILE_RASENGAN_SPEED_PX)

    # Act
    field.update(_make_signals(), 0.1, _make_ability())

    # Assert
    assert len(field._projectiles) == 0


@pytest.mark.unit
def test_projectile_inside_screen_stays_alive():
    # Arrange — spawn well inside the screen, one tiny step
    field, hooks = _make_field()
    _spawn(hooks, origin=(640.0, 360.0), direction=(1.0, 0.0), speed_px=100.0)

    # Act — a single 1/60 s frame won't push it past the edge
    field.update(_make_signals(), 1 / 60.0, _make_ability())

    # Assert
    assert len(field._projectiles) == 1


@pytest.mark.unit
def test_burst_particles_age_out():
    # Arrange — push a projectile past the edge to spawn bursts.
    # Use a tiny dt so the projectile crosses the edge without also expiring
    # the newly-created burst particles in the same frame.  (Burst max_age is
    # ~0.2–0.4 s; a 1-second dt would age them to death immediately.)
    width, height = 1280, 720
    field, hooks = _make_field(width, height)
    margin = config.PROJECTILE_EDGE_MARGIN_PX
    # Start already past the edge so even a tiny dt triggers the burst.
    _spawn(hooks, origin=(float(width) + margin + 1.0, 360.0),
           direction=(1.0, 0.0), speed_px=config.PROJECTILE_RASENGAN_SPEED_PX)
    field.update(_make_signals(), 1 / 60.0, _make_ability())  # triggers burst
    assert len(field._bursts) > 0, "burst particles should appear after edge crossing"

    # Act — advance time far beyond max burst lifetime (typically < 0.5 s)
    field.update(_make_signals(), 5.0, _make_ability())

    # Assert — all burst particles have expired
    assert len(field._bursts) == 0


@pytest.mark.unit
def test_cap_drops_oldest_when_exceeded():
    # Arrange — spawn PROJECTILE_MAX_ACTIVE projectiles, then one more
    field, hooks = _make_field()
    cap = config.PROJECTILE_MAX_ACTIVE

    for _ in range(cap):
        _spawn(hooks)

    assert len(field._projectiles) == cap

    # Remember the oldest projectile's kind tag (we'll use origin to distinguish)
    # We'll tag the first by its initial x position being unique
    _spawn(hooks, origin=(999.0, 100.0))  # the +1 spawn

    # Assert — still at cap, oldest (first) was dropped
    assert len(field._projectiles) == cap
    # The last spawned (origin 999,100) should be present
    last_x_values = [float(p.pos[0]) for p in field._projectiles]
    assert 999.0 in last_x_values


@pytest.mark.unit
def test_update_is_idempotent_with_no_projectiles():
    # Arrange — empty field
    field, hooks = _make_field()

    # Act — call update several times with no projectiles
    for _ in range(10):
        field.update(_make_signals(), 1 / 60.0, _make_ability())

    # Assert — nothing broke, still empty
    assert len(field._projectiles) == 0
    assert len(field._bursts) == 0


@pytest.mark.unit
def test_fireball_projectile_spawned_with_correct_kind():
    # Arrange
    field, hooks = _make_field()

    # Act
    _spawn(hooks, kind="fireball",
           speed_px=config.PROJECTILE_FIREBALL_SPEED_PX,
           radius_px=config.PROJECTILE_FIREBALL_RADIUS_PX)

    # Assert
    assert field._projectiles[0].kind == "fireball"
    assert abs(field._projectiles[0].radius - config.PROJECTILE_FIREBALL_RADIUS_PX) < 0.01
