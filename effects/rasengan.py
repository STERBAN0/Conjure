"""Rasengan — spinning sphere of chakra in the active palm.

Visual specification:
- Dense, fast-rotating blue sphere of particles in the active palm.
- Particles trace short curved trails (faded streaks).
- Outer shell more defined than Chidori — orderly, not chaotic.
- Charge: sphere grows from a small point.
- Release: sphere expands rapidly, dissipates in a radial particle burst.
"""

from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np
import pygame

import config
from core.state import (
    AbilityState,
    FrameState,
    GestureSignals,
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
)
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    ease_in_out_quad,
    radial_glow,
)


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
        self._burst: list[_Streak] = []   # expansion burst on release
        self._burst_active = False

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        self._streaks.clear()
        self._burst.clear()
        self._burst_active = False

    def on_release(self, intensity: float, frame: FrameState) -> None:
        # Spawn a radial burst of fast outward streaks
        for _ in range(int(80 * (0.5 + 0.5 * intensity))):
            a = self._rng.uniform(0, math.tau)
            self._burst.append(_Streak(
                angle=a,
                radius=10.0,
                ang_velocity=self._rng.uniform(2.0, 6.0),
                life=self._rng.uniform(0.35, 0.6),
                size=self._rng.uniform(2, 4),
            ))
        self._burst_active = True

    def on_exit(self) -> None:
        self._streaks.clear()
        self._burst.clear()
        self._burst_active = False

    # ------------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        # Always step
        for s in self._streaks:
            s.angle += s.ang_velocity * dt
            s.life -= dt
        self._streaks = [s for s in self._streaks if s.life > 0]

        if ability.phase in (PHASE_CHARGING, PHASE_ACTIVE):
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

        # Burst handling
        if self._burst_active:
            for b in self._burst:
                b.radius += 700.0 * dt
                b.life -= dt
            self._burst = [b for b in self._burst if b.life > 0]
            if not self._burst:
                self._burst_active = False

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        cx, cy = self._center(ability)

        if ability.phase == PHASE_CHARGING or ability.phase == PHASE_ACTIVE:
            self._render_sphere(target, ability, (cx, cy))
        elif ability.phase == PHASE_RELEASING:
            # Sphere fades fast
            fade = max(0.0, 1.0 - ability.phase_age / 0.2)
            if fade > 0.01:
                radial_glow(
                    target, (cx, cy),
                    radius=int(self._sphere_radius(ability) * 1.4),
                    color=config.RASENGAN_OUTER_COLOR,
                    alpha=int(200 * fade),
                )

        # Burst streaks (always draw if any are alive)
        for b in self._burst:
            x = int(cx + math.cos(b.angle) * b.radius)
            y = int(cy + math.sin(b.angle) * b.radius)
            life_ratio = b.life / max(1e-6, b.max_life)
            tx = int(cx + math.cos(b.angle) * max(0.0, b.radius - 55.0))
            ty = int(cy + math.sin(b.angle) * max(0.0, b.radius - 55.0))
            additive_polyline(
                target,
                [(tx, ty), (x, y)],
                color=config.RASENGAN_CORE_COLOR,
                width=max(1, int(b.size)),
                alpha=int(150 * life_ratio),
            )
            additive_circle(
                target, (x, y),
                size=max(1, int(b.size * (0.4 + 0.6 * life_ratio))),
                color=config.RASENGAN_OUTER_COLOR,
                alpha=int(220 * life_ratio),
            )

    # ------------------------------------------------------------------

    def _render_sphere(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        center: Tuple[int, int],
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

    def _center(self, ability: AbilityState) -> Tuple[int, int]:
        h = ability.primary_hand
        if h is None:
            return (self.width // 2, self.height // 2)
        return int(h.palm[0] * self.width), int(h.palm[1] * self.height)
