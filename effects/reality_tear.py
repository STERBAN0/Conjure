"""Reality tear — jagged glowing fracture pulled open by clawed hands.

Bound to the `reality_tear` ability (clawed-pose pair). The tear opens
with sustained pose, jitters along its length, and snaps shut with a
brief flash on exit.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

import config
from core.state import AbilityState, GestureSignals
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    dark_polyline,
    draw_screen_flash,
    radial_glow,
)


class RealityTearEffect(Effect):
    layer = LAYER_FG
    name = "reality_tear"
    ability_name = "reality_tear"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0x7EA2)
        self._noise_phase = 0.0
        self._flash = 0.0

    def on_exit(self) -> None:
        # Snap-shut flash carries strength of the last frame
        self._flash = 0.7

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        self._noise_phase += dt * 8.0
        self._flash = max(0.0, self._flash - dt * 3.0)

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        if ability.primary_hand is None or ability.secondary_hand is None:
            if self._flash > 0:
                draw_screen_flash(target, (220, 200, 255), int(80 * self._flash))
            return

        a = ability.primary_hand.palm * np.array([self.width, self.height])
        b = ability.secondary_hand.palm * np.array([self.width, self.height])

        # Strength ramps with ability age, capped.
        strength = float(np.clip(ability.age * 1.6, 0.0, 1.0))

        # Extrapolate beyond hands a touch for drama
        center = (a + b) * 0.5
        diff = b - a
        length = float(np.linalg.norm(diff)) + 1e-6
        ux, uy = diff[0] / length, diff[1] / length
        nx, ny = -uy, ux
        half = length * 0.6 * (0.7 + 0.6 * strength)
        p0 = center - np.array([ux, uy]) * half
        p1 = center + np.array([ux, uy]) * half

        amp = config.REALITY_TEAR_BASE_AMP + (
            config.REALITY_TEAR_PEAK_AMP - config.REALITY_TEAR_BASE_AMP
        ) * strength
        n = config.REALITY_TEAR_SEGMENTS
        ts = np.linspace(0.0, 1.0, n + 1)

        # 1D smooth noise
        noise = np.zeros_like(ts)
        for k, freq in enumerate((4.0, 9.0, 17.0)):
            noise += np.sin(ts * freq * math.pi * 2 + self._noise_phase * (1 + k)) / (k + 1)
        noise *= amp / 2.0

        points: list[tuple[float, float]] = []
        for t, val in zip(ts, noise, strict=False):
            base = p0 * (1 - t) + p1 * t
            points.append((float(base[0] + nx * val), float(base[1] + ny * val)))

        # Dark inner slit first, then additive chromatic edges. This gives the
        # tear contrast against bright camera frames instead of just tinting.
        dark_polyline(
            target,
            points,
            color=(0, 0, 0),
            width=max(3, int(10 + 18 * strength)),
            alpha=int(170 * strength),
        )

        radial_glow(
            target,
            (int(center[0]), int(center[1])),
            radius=int(80 + 180 * strength),
            color=(160, 100, 255),
            alpha=int(70 * strength),
            layers=9,
        )

        # Outer chromatic glow — red shifted one way, blue the other
        if strength > 0.2:
            shift = 5
            additive_polyline(
                target, [(p[0] + nx * shift, p[1] + ny * shift) for p in points],
                color=(255, 90, 180), width=10, alpha=int(150 * strength),
            )
            additive_polyline(
                target, [(p[0] - nx * shift, p[1] - ny * shift) for p in points],
                color=(120, 180, 255), width=10, alpha=int(150 * strength),
            )

        branch_indices = range(4, len(points) - 4, 6)
        for i, idx in enumerate(branch_indices):
            base = np.asarray(points[idx], dtype=np.float32)
            side = -1.0 if i % 2 == 0 else 1.0
            phase = self._noise_phase * (1.2 + 0.15 * i)
            branch_len = (35 + 60 * strength) * (0.65 + 0.35 * math.sin(phase))
            direction = (
                np.array([nx, ny]) * side
                + np.array([ux, uy]) * math.sin(phase + i) * 0.45
            )
            direction /= np.linalg.norm(direction) + 1e-6
            end = base + direction * branch_len
            mid = base + direction * branch_len * 0.55 + np.array([ux, uy]) * (
                math.sin(phase * 1.7) * 18 * strength
            )
            additive_polyline(
                target,
                [
                    (float(base[0]), float(base[1])),
                    (float(mid[0]), float(mid[1])),
                    (float(end[0]), float(end[1])),
                ],
                color=(255, 120, 210) if side < 0 else (120, 210, 255),
                width=max(1, int(1 + 3 * strength)),
                alpha=int(120 * strength),
            )

        # Hot inner core
        additive_polyline(
            target, points,
            color=(255, 245, 230),
            width=max(2, int(2 + 4 * strength)),
            alpha=255,
        )

        for _ in range(int(8 + 18 * strength)):
            p = np.asarray(points[self._rng.randrange(0, len(points))])
            jitter = np.array([nx, ny]) * self._rng.uniform(-18, 18) * strength
            spark = p + jitter
            additive_circle(
                target,
                (int(spark[0]), int(spark[1])),
                size=self._rng.randint(1, 3),
                color=(255, 235, 255),
                alpha=int(100 + 120 * strength),
            )

        if self._flash > 0:
            draw_screen_flash(target, (220, 200, 255), int(80 * self._flash))
