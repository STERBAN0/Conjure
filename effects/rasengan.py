"""Rasengan — spinning sphere of chakra in the active palm.

Visual specification:
- Dense, fast-rotating blue sphere of particles in the active palm.
- Particles trace short curved trails (faded streaks).
- Outer shell more defined than Chidori — orderly, not chaotic.
- Charge: sphere grows from a small point.
- Release: the sphere leaves the hand the instant it's thrown — a brief launch
  puff (radial glow at the hand) marks the departure, and the flying sphere is
  then carried by ProjectileField. Nothing lingers on the palm after the throw.
"""

from __future__ import annotations

import math
import random

import pygame

import config
from core.state import (
    PHASE_CHARGING,
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

_PUFF_DURATION = 0.22


# NOTE: _Streak is structurally near-identical to _Ember in effects/fireball.py.
# They are intentionally kept separate to avoid coupling two independent effects
# through a shared base class — extract to effects/utils.py only if a third
# user appears.
class _Streak:
    __slots__ = ("angle", "radius", "ang_velocity", "life", "max_life", "size")

    def __init__(self, angle, radius, ang_velocity, life, size):
        self.angle = float(angle)
        self.radius = float(radius)
        self.ang_velocity = float(ang_velocity)
        self.life = float(life)
        self.max_life = float(life)
        self.size = float(size)


class RasenganEffect(Effect):
    layer = LAYER_FG
    name = "rasengan"
    ability_name = "rasengan"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0xCA0)
        self._streaks: list[_Streak] = []
        self._puff_age: float = -1.0   # seconds since launch puff; -1 = inactive

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        self._streaks.clear()
        self._puff_age = -1.0

    def on_release(self, intensity: float, frame: FrameState) -> None:
        self._puff_age = 0.0

    def on_exit(self) -> None:
        self._streaks.clear()
        self._puff_age = -1.0

    # ------------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        # Always step
        for s in self._streaks:
            s.angle += s.ang_velocity * dt
            s.life -= dt
        self._streaks = [s for s in self._streaks if s.life > 0]

        if ability.phase == PHASE_CHARGING:
            target = int(config.RASENGAN_PARTICLE_COUNT * max(ability.charge, ability.intensity))
            radius = self._sphere_radius(ability)
            while len(self._streaks) < target:
                self._streaks.append(_Streak(
                    angle=self._rng.uniform(0, math.tau),
                    radius=self._rng.uniform(radius * 0.25, radius),
                    ang_velocity=self._rng.uniform(6.0, 14.0)
                    * self._rng.choice((-1.0, 1.0)),
                    life=self._rng.uniform(0.2, 0.5),
                    size=self._rng.uniform(2, 3),
                ))

        # Advance puff age
        if self._puff_age >= 0.0:
            self._puff_age += dt

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        cx, cy = self._center(ability)

        # The sphere only lives in the hand while charging. The instant it's
        # thrown (phase -> ACTIVE) it leaves the hand entirely: no fading ball on
        # the palm, just the launch puff below. This is what makes the rasengan
        # depart WITH the projectile instead of lingering for ~half a second.
        if ability.phase == PHASE_CHARGING:
            self._render_sphere(target, ability, (cx, cy))

        # Launch puff: brief concentric radial glow at the hand on release
        if 0.0 <= self._puff_age < _PUFF_DURATION:
            t = self._puff_age / _PUFF_DURATION
            puff_alpha = int(220 * (1.0 - t))
            puff_radius = int(config.RASENGAN_RADIUS_PEAK * (0.8 + 1.4 * t))
            radial_glow(
                target, (cx, cy),
                radius=puff_radius,
                color=config.RASENGAN_OUTER_COLOR,
                alpha=puff_alpha,
                layers=10,
            )
            radial_glow(
                target, (cx, cy),
                radius=int(puff_radius * 0.5),
                color=config.RASENGAN_CORE_COLOR,
                alpha=min(255, int(puff_alpha * 1.3)),
                layers=6,
            )

    # ------------------------------------------------------------------

    def _render_sphere(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        center: tuple[int, int],
    ) -> None:
        cx, cy = center
        radius = self._sphere_radius(ability)
        # Outer glow
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 1.5),
            color=config.RASENGAN_OUTER_COLOR,
            alpha=int(170 * ability.charge),
        )
        # Hot core
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 0.7),
            color=config.RASENGAN_CORE_COLOR,
            alpha=int(220 * ability.charge),
        )
        for ring_scale, alpha_scale in ((0.75, 80), (1.05, 110), (1.28, 60)):
            additive_ring(
                target,
                (cx, cy),
                radius=int(radius * ring_scale),
                color=(190, 235, 255),
                alpha=int(alpha_scale * ability.charge),
                width=max(1, int(1 + 2 * ability.charge)),
            )
        # Vortex arms give the sphere readable rotation even when particles blur.
        for arm in range(3):
            pts = []
            for j in range(26):
                t = j / 25.0
                phi = ability.age * 11.0 + arm * math.tau / 3.0 + t * 5.2
                r = radius * t
                pts.append((
                    cx + math.cos(phi) * r,
                    cy + math.sin(phi) * r * 0.72,
                ))
            additive_polyline(
                target,
                pts,
                color=config.RASENGAN_CORE_COLOR,
                width=max(1, int(2 + 2 * ability.charge)),
                alpha=int(110 * ability.charge),
            )
        # Defined orbital shell — N circles arranged at sphere radius
        shell_n = config.RASENGAN_SHELL_DENSITY
        for i in range(shell_n):
            phi = (i / shell_n) * math.tau + ability.age * 6.0
            r = radius * (0.85 + 0.15 * math.sin(ability.age * 12.0 + i))
            x = int(cx + math.cos(phi) * r)
            y = int(cy + math.sin(phi) * r)
            additive_circle(
                target, (x, y),
                size=2,
                color=config.RASENGAN_CORE_COLOR,
                alpha=180,
            )
        # Streaks
        for s in self._streaks:
            x = int(cx + math.cos(s.angle) * s.radius)
            y = int(cy + math.sin(s.angle) * s.radius)
            life_ratio = s.life / max(1e-6, s.max_life)
            tail_angle = s.angle - math.copysign(0.42, s.ang_velocity)
            tx = cx + math.cos(tail_angle) * s.radius
            ty = cy + math.sin(tail_angle) * s.radius
            additive_polyline(
                target,
                [(tx, ty), (x, y)],
                color=config.RASENGAN_OUTER_COLOR,
                width=max(1, int(s.size)),
                alpha=int(140 * life_ratio),
            )
            additive_circle(
                target, (x, y),
                size=max(1, int(s.size * (0.5 + 0.5 * life_ratio))),
                color=config.RASENGAN_OUTER_COLOR,
                alpha=int(220 * life_ratio),
            )

    def _sphere_radius(self, ability: AbilityState) -> float:
        return (
            config.RASENGAN_RADIUS_BASE
            + (config.RASENGAN_RADIUS_PEAK - config.RASENGAN_RADIUS_BASE)
            * ease_in_out_quad(ability.charge)
        )

    def _center(self, ability: AbilityState) -> tuple[int, int]:
        h = ability.primary_hand
        if h is None:
            return (self.width // 2, self.height // 2)
        return int(h.palm[0] * self.width), int(h.palm[1] * self.height)
