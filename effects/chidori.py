"""Chidori — Sasuke's lightning blade (melee version).

Visual specification:
- Dense cluster of bright cyan-white electric arcs emanating from the
  active palm. Arcs are jagged polylines with random branches; redrawn
  every frame so they flicker.
- Inner core near-white, outer layer cyan, occasional magenta highlights.
- Audio level modulates arc count and length so it crackles louder with
  ambient noise.
- Charge phase: arcs start sparse and short, grow denser/longer.
- Release phase: melee strike — brief bright flash, intensified arcs + a
  concentrated impact glow at the palm. Lightning STAYS on the hand;
  no forward beam, no projectile.
"""

from __future__ import annotations

import math
import random

import numpy as np
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
    draw_screen_flash,
    ease_in_out_quad,
    jagged_branch,
    radial_glow,
)


class ChidoriEffect(Effect):
    layer = LAYER_FG
    name = "chidori"
    ability_name = "chidori"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0xC417)
        self._flash = 0.0        # 0..1, decays after release (melee strike flash)
        self._strike_age = 0.0   # seconds since release, for burst ramp-down

    # --- Lifecycle ----------------------------------------------------------

    def on_release(self, intensity: float, frame: FrameState) -> None:
        self._flash = max(self._flash, intensity)
        self._strike_age = 0.0

    def on_exit(self) -> None:
        self._flash = 0.0
        self._strike_age = 0.0

    # --- Tick / draw --------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        self._flash = max(0.0, self._flash - dt * 4.5)
        if ability.phase in (PHASE_RELEASING, PHASE_ACTIVE):
            self._strike_age += dt

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        hand = ability.primary_hand
        if hand is None:
            return

        cx = int(hand.palm[0] * self.width)
        cy = int(hand.palm[1] * self.height)
        palm_anchor = np.array([cx, cy], dtype=np.float32)
        finger_anchors: list[tuple[int, int]] = []
        for idx in (8, 12):
            if hand.landmarks.shape[0] > idx:
                lm = hand.landmarks[idx]
                finger_anchors.append(
                    (int(lm[0] * self.width), int(lm[1] * self.height))
                )

        charge_eased = ease_in_out_quad(ability.charge)
        audio_boost = 0.6 + 0.4 * float(signals.audio_level)

        if ability.phase in (PHASE_CHARGING, PHASE_ACTIVE, PHASE_RELEASING):
            # Strike intensity: ramps up on release, then decays
            if ability.phase in (PHASE_RELEASING, PHASE_ACTIVE):
                strike_boost = max(0.0, 1.0 - self._strike_age / 0.35)
            else:
                strike_boost = 0.0

            arc_intensity = max(charge_eased, strike_boost)

            # 1) Underlying glow at the palm — intensifies on strike
            glow_r = int(20 + 90 * arc_intensity * audio_boost)
            radial_glow(
                target, (cx, cy),
                radius=glow_r,
                color=config.CHIDORI_GLOW_COLOR,
                alpha=int(180 * arc_intensity),
            )
            # Extra impact glow ring on strike
            if strike_boost > 0.0:
                radial_glow(
                    target, (cx, cy),
                    radius=int(60 + 80 * strike_boost),
                    color=config.CHIDORI_CORE_COLOR,
                    alpha=int(200 * strike_boost),
                    layers=12,
                )
            additive_ring(
                target,
                (cx, cy),
                radius=int(18 + 52 * arc_intensity),
                color=config.CHIDORI_HIGHLIGHT_COLOR,
                alpha=int(90 * arc_intensity),
                width=max(1, int(1 + 3 * arc_intensity)),
            )
            for anchor in finger_anchors:
                radial_glow(
                    target,
                    anchor,
                    radius=int(18 + 36 * arc_intensity),
                    color=config.CHIDORI_GLOW_COLOR,
                    alpha=int(110 * arc_intensity),
                    layers=6,
                )
                additive_polyline(
                    target,
                    [(cx, cy), anchor],
                    color=config.CHIDORI_CORE_COLOR,
                    width=max(1, int(1 + 3 * arc_intensity)),
                    alpha=int(180 * arc_intensity),
                )

            # 2) Arcs radiating outward — burst count/length on strike
            arc_count = int(
                config.CHIDORI_ARC_COUNT_BASE
                + (config.CHIDORI_ARC_COUNT_PEAK - config.CHIDORI_ARC_COUNT_BASE)
                * arc_intensity * audio_boost
            )
            # On strike: 1.5x arc density for a brief moment
            if strike_boost > 0.2:
                arc_count = int(arc_count * (1.0 + 0.6 * strike_boost))
            arc_len = (
                config.CHIDORI_ARC_LENGTH_BASE_PX
                + (config.CHIDORI_ARC_LENGTH_PEAK_PX
                   - config.CHIDORI_ARC_LENGTH_BASE_PX) * arc_intensity
            )

            for _ in range(arc_count):
                origin = (cx, cy)
                if finger_anchors and self._rng.random() < 0.65:
                    origin = self._rng.choice(finger_anchors)
                anchor_vec = np.asarray(origin, dtype=np.float32) - palm_anchor
                if np.linalg.norm(anchor_vec) > 8.0:
                    base_angle = math.atan2(float(anchor_vec[1]), float(anchor_vec[0]))
                    angle = base_angle + self._rng.uniform(-1.25, 1.25)
                else:
                    angle = self._rng.uniform(0, math.tau)
                length = arc_len * self._rng.uniform(0.55, 1.0)
                self._draw_arc(
                    target, origin,
                    angle=angle,
                    length=length,
                    intensity=arc_intensity,
                )

            # 3) Small dancing core sparks
            for _ in range(int(10 + 22 * arc_intensity)):
                a = self._rng.uniform(0, math.tau)
                r = self._rng.uniform(0, 18 + 18 * arc_intensity)
                px = int(cx + math.cos(a) * r)
                py = int(cy + math.sin(a) * r)
                additive_circle(
                    target, (px, py),
                    size=self._rng.randint(1, 3),
                    color=config.CHIDORI_CORE_COLOR,
                    alpha=240,
                )

        # 4) Release flash — bright white for the melee strike impact
        if self._flash > 0:
            draw_screen_flash(
                target, color=(255, 255, 255), alpha=int(140 * self._flash)
            )

    # --- Internal -----------------------------------------------------------

    def _draw_arc(
        self,
        target: pygame.Surface,
        origin: tuple[int, int],
        angle: float,
        length: float,
        intensity: float,
    ) -> None:
        direction = (math.cos(angle), math.sin(angle))
        path = jagged_branch(
            origin, direction, length,
            segment_len=config.CHIDORI_SEGMENT_LEN_PX,
            jitter=config.CHIDORI_JITTER_PX * (0.5 + 0.5 * intensity),
            rng=self._rng,
        )
        # Outer cyan glow (thicker)
        additive_polyline(
            target, path,
            color=config.CHIDORI_GLOW_COLOR,
            width=6,
            alpha=int(150 * intensity),
        )
        # Inner white core
        additive_polyline(
            target, path,
            color=config.CHIDORI_CORE_COLOR,
            width=2,
            alpha=255,
        )
        # Branches at random
        if self._rng.random() < config.CHIDORI_BRANCH_PROB and len(path) > 4:
            split = self._rng.randint(2, len(path) - 2)
            split_origin = path[split]
            branch_angle = angle + self._rng.uniform(-1.0, 1.0)
            branch_dir = (math.cos(branch_angle), math.sin(branch_angle))
            branch = jagged_branch(
                split_origin, branch_dir,
                length=length * self._rng.uniform(0.3, 0.6),
                segment_len=config.CHIDORI_SEGMENT_LEN_PX,
                jitter=config.CHIDORI_JITTER_PX * 0.6,
                rng=self._rng,
            )
            additive_polyline(
                target, branch,
                color=config.CHIDORI_HIGHLIGHT_COLOR,
                width=2,
                alpha=int(180 * intensity),
            )
