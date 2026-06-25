"""Fireball — turbulent ember sphere at the active palm.

Visual specification:
- LAYER_FG. Warm palette (orange/ember).
- Gesture: one hand with ONLY the index finger up; the ember sphere forms just
  above the INDEX FINGERTIP (config.FIREBALL_FINGERTIP_LIFT_PX), not the palm.
- Charging: a turbulent sphere of fire particles grows from a small ember.
- Release: brief launch puff (small radial glow at the fingertip).
  The flying fireball is handled by ProjectileField; charge once, then flick the
  finger to shoot repeatedly while the index-up pose is held.
- Palette: FIREBALL_CORE_COLOR (near-white yellow), FIREBALL_OUTER_COLOR (deep orange).
"""

from __future__ import annotations

import math
import random

import pygame

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
    AbilityState,
    FrameState,
    GestureSignals,
)
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    ease_in_out_quad,
    radial_glow,
)

_PUFF_DURATION = 0.20
_INDEX_TIP = 8  # MediaPipe index fingertip landmark


class _Ember:
    __slots__ = ("angle", "radius", "ang_velocity", "life", "max_life", "size")

    def __init__(
        self,
        angle: float,
        radius: float,
        ang_velocity: float,
        life: float,
        size: float,
    ) -> None:
        self.angle = angle
        self.radius = radius
        self.ang_velocity = ang_velocity
        self.life = life
        self.max_life = life
        self.size = size


class FireballEffect(Effect):
    layer = LAYER_FG
    name = "fireball"
    ability_name = "fireball"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0xF11E)
        self._embers: list[_Ember] = []
        self._puff_age: float = -1.0  # seconds since launch puff; -1 = inactive

    # --- Lifecycle ----------------------------------------------------------

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        self._embers.clear()
        self._puff_age = -1.0

    def on_release(self, intensity: float, frame: FrameState) -> None:
        self._puff_age = 0.0

    def on_exit(self) -> None:
        self._embers.clear()
        self._puff_age = -1.0

    # --- Update -------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        # Step existing embers
        for e in self._embers:
            e.angle += e.ang_velocity * dt
            e.life -= dt
        self._embers = [e for e in self._embers if e.life > 0]

        if ability.phase in (PHASE_CHARGING, PHASE_ACTIVE):
            target_count = int(
                config.FIREBALL_PARTICLE_COUNT
                * max(ability.charge, ability.intensity)
            )
            radius = self._sphere_radius(ability)
            while len(self._embers) < target_count:
                self._embers.append(_Ember(
                    angle=self._rng.uniform(0, math.tau),
                    radius=self._rng.uniform(radius * 0.2, radius),
                    ang_velocity=(
                        self._rng.uniform(4.0, 12.0)
                        * self._rng.choice((-1.0, 1.0))
                    ),
                    life=self._rng.uniform(0.15, 0.45),
                    size=self._rng.uniform(2, 4),
                ))

        # Advance puff age
        if self._puff_age >= 0.0:
            self._puff_age += dt

    # --- Render -------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        cx, cy = self._center(ability)

        if ability.phase in (PHASE_CHARGING, PHASE_ACTIVE):
            self._render_sphere(target, ability, (cx, cy))
        elif ability.phase == PHASE_RELEASING:
            fade = max(0.0, 1.0 - ability.phase_age / 0.18)
            if fade > 0.01:
                radial_glow(
                    target, (cx, cy),
                    radius=int(self._sphere_radius(ability) * 1.5),
                    color=config.FIREBALL_OUTER_COLOR,
                    alpha=int(200 * fade),
                )

        # Launch puff
        if 0.0 <= self._puff_age < _PUFF_DURATION:
            t = self._puff_age / _PUFF_DURATION
            puff_alpha = int(230 * (1.0 - t))
            puff_radius = int(config.FIREBALL_RADIUS_PEAK * (0.9 + 1.6 * t))
            radial_glow(
                target, (cx, cy),
                radius=puff_radius,
                color=config.FIREBALL_OUTER_COLOR,
                alpha=puff_alpha,
                layers=10,
            )
            radial_glow(
                target, (cx, cy),
                radius=int(puff_radius * 0.45),
                color=config.FIREBALL_CORE_COLOR,
                alpha=min(255, int(puff_alpha * 1.3)),
                layers=6,
            )

    # --- Private ------------------------------------------------------------

    def _render_sphere(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        center: tuple[int, int],
    ) -> None:
        cx, cy = center
        radius = self._sphere_radius(ability)
        charge = ability.charge

        # Outer heat glow
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 1.6),
            color=config.FIREBALL_OUTER_COLOR,
            alpha=int(160 * charge),
        )
        # Hot inner core
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 0.65),
            color=config.FIREBALL_CORE_COLOR,
            alpha=int(230 * charge),
        )
        # Turbulent outer shell rings (phase-shifted for flicker)
        for ring_scale, alpha_base in ((0.7, 70), (1.0, 90), (1.25, 55)):
            flicker = 0.85 + 0.15 * math.sin(ability.age * 23.0 + ring_scale * 9.0)
            additive_ring(
                target, (cx, cy),
                radius=int(radius * ring_scale),
                color=config.FIREBALL_OUTER_COLOR,
                alpha=int(alpha_base * charge * flicker),
                width=max(1, int(1 + 2 * charge)),
            )
        # Swirling turbulent arms
        for arm in range(4):
            pts = []
            for j in range(20):
                t = j / 19.0
                phi = (
                    ability.age * 9.0
                    + arm * math.tau / 4.0
                    + t * 4.5
                    # turbulence ripple
                    + 0.4 * math.sin(ability.age * 17.0 + j * 0.7)
                )
                r = radius * t
                pts.append((
                    cx + math.cos(phi) * r,
                    cy + math.sin(phi) * r * 0.8,
                ))
            additive_polyline(
                target, pts,
                color=config.FIREBALL_CORE_COLOR,
                width=max(1, int(2 + 2 * charge)),
                alpha=int(100 * charge),
            )
        # Ember particles
        for e in self._embers:
            ex = int(cx + math.cos(e.angle) * e.radius)
            ey = int(cy + math.sin(e.angle) * e.radius)
            life_ratio = e.life / max(1e-6, e.max_life)
            # short trailing streak
            tail_angle = e.angle - math.copysign(0.35, e.ang_velocity)
            tx = int(cx + math.cos(tail_angle) * e.radius)
            ty = int(cy + math.sin(tail_angle) * e.radius)
            additive_polyline(
                target, [(tx, ty), (ex, ey)],
                color=config.FIREBALL_OUTER_COLOR,
                width=max(1, int(e.size)),
                alpha=int(130 * life_ratio),
            )
            additive_circle(
                target, (ex, ey),
                size=max(1, int(e.size * (0.5 + 0.5 * life_ratio))),
                color=config.FIREBALL_CORE_COLOR,
                alpha=int(210 * life_ratio),
            )

    def _sphere_radius(self, ability: AbilityState) -> float:
        return (
            config.FIREBALL_RADIUS_BASE
            + (config.FIREBALL_RADIUS_PEAK - config.FIREBALL_RADIUS_BASE)
            * ease_in_out_quad(ability.charge)
        )

    def _center(self, ability: AbilityState) -> tuple[int, int]:
        h = ability.primary_hand
        if h is None:
            return (self.width // 2, self.height // 2)
        # Fireball forms just ABOVE the index fingertip (suspended in the air),
        # matching the spawn point in the router. Falls back to the palm centre
        # only if landmarks are unavailable.
        if h.landmarks is not None:
            tip = h.landmarks[_INDEX_TIP, :2]
            cx = int(float(tip[0]) * self.width)
            cy = int(float(tip[1]) * self.height - config.FIREBALL_FINGERTIP_LIFT_PX)
            return cx, cy
        return int(h.palm[0] * self.width), int(h.palm[1] * self.height)
