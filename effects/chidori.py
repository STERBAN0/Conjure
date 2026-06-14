"""Chidori — Sasuke's lightning blade.

Visual specification:
- Dense cluster of bright cyan-white electric arcs emanating from the
  active palm. Arcs are jagged polylines with random branches; redrawn
  every frame so they flicker.
- Inner core near-white, outer layer cyan, occasional magenta highlights.
- Audio level modulates arc count and length so it crackles louder when
  there's ambient noise.
- Charge phase: arcs start sparse and short, grow denser/longer.
- Release phase: arcs converge into a forward beam, brief screen flash,
  then a fast fade.
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
        self._flash = 0.0          # 0..1, decays after release
        self._beam_intensity = 0.0  # post-release converging beam strength

    # --- Lifecycle ----------------------------------------------------------

    def on_release(self, intensity: float, frame: FrameState) -> None:
        self._flash = max(self._flash, intensity)
        self._beam_intensity = max(self._beam_intensity, intensity)

    def on_exit(self) -> None:
        self._flash = 0.0
        self._beam_intensity = 0.0

    # --- Tick / draw --------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        self._flash = max(0.0, self._flash - dt * 4.0)
        self._beam_intensity = max(0.0, self._beam_intensity - dt * 1.6)

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
        finger_anchors: list[Tuple[int, int]] = []
        for idx in (8, 12):
            if hand.landmarks.shape[0] > idx:
                lm = hand.landmarks[idx]
                finger_anchors.append(
                    (int(lm[0] * self.width), int(lm[1] * self.height))
                )

        charge_eased = ease_in_out_quad(ability.charge)
        audio_boost = 0.6 + 0.4 * float(signals.audio_level)

        if ability.phase in (PHASE_CHARGING, PHASE_ACTIVE, PHASE_RELEASING):
            # 1) Underlying glow at the palm
            glow_r = int(20 + 90 * charge_eased * audio_boost)
            radial_glow(
                target, (cx, cy),
                radius=glow_r,
                color=config.CHIDORI_GLOW_COLOR,
                alpha=int(180 * charge_eased),
            )
            additive_ring(
                target,
                (cx, cy),
                radius=int(18 + 52 * charge_eased),
                color=config.CHIDORI_HIGHLIGHT_COLOR,
                alpha=int(90 * charge_eased),
                width=max(1, int(1 + 3 * charge_eased)),
            )
            for anchor in finger_anchors:
                radial_glow(
                    target,
                    anchor,
                    radius=int(18 + 36 * charge_eased),
                    color=config.CHIDORI_GLOW_COLOR,
                    alpha=int(110 * charge_eased),
                    layers=6,
                )
                additive_polyline(
                    target,
                    [(cx, cy), anchor],
                    color=config.CHIDORI_CORE_COLOR,
                    width=max(1, int(1 + 3 * charge_eased)),
                    alpha=int(180 * charge_eased),
                )

            # 2) Arcs radiating outward
            arc_count = int(
                config.CHIDORI_ARC_COUNT_BASE
                + (config.CHIDORI_ARC_COUNT_PEAK - config.CHIDORI_ARC_COUNT_BASE)
                * charge_eased * audio_boost
            )
            arc_len = (
                config.CHIDORI_ARC_LENGTH_BASE_PX
                + (config.CHIDORI_ARC_LENGTH_PEAK_PX
                   - config.CHIDORI_ARC_LENGTH_BASE_PX) * charge_eased
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
                    intensity=charge_eased,
                )

            # 3) Small dancing core sparks
            for _ in range(int(10 + 22 * charge_eased)):
                a = self._rng.uniform(0, math.tau)
                r = self._rng.uniform(0, 18 + 18 * charge_eased)
                px = int(cx + math.cos(a) * r)
                py = int(cy + math.sin(a) * r)
                additive_circle(
                    target, (px, py),
                    size=self._rng.randint(1, 3),
                    color=config.CHIDORI_CORE_COLOR,
                    alpha=240,
                )

        # 4) Release flash + forward beam
        if self._flash > 0:
            draw_screen_flash(
                target, color=(255, 255, 255), alpha=int(120 * self._flash)
            )
        if self._beam_intensity > 0 and ability.phase != PHASE_CHARGING:
            self._draw_beam(target, (cx, cy), self._beam_intensity)

    # --- Internal -----------------------------------------------------------

    def _draw_arc(
        self,
        target: pygame.Surface,
        origin: Tuple[int, int],
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

    def _draw_beam(
        self,
        target: pygame.Surface,
        origin: Tuple[int, int],
        intensity: float,
    ) -> None:
        # Forward beam: a horizontal-ish straight bar from origin off the
        # frame edge, aimed at the user's "forward". Without true forward
        # vector estimation we approximate "into the screen" as "outward
        # from the centre of the screen along the origin->edge ray".
        cx, cy = origin
        ex, ey = self.width // 2, self.height // 2
        dx, dy = cx - ex, cy - ey
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        end = (
            cx + ux * self.width,
            cy + uy * self.height,
        )
        path = [origin, end]
        additive_polyline(
            target, path, config.CHIDORI_GLOW_COLOR,
            width=int(40 * intensity), alpha=int(160 * intensity),
        )
        additive_polyline(
            target, path, config.CHIDORI_CORE_COLOR,
            width=int(14 * intensity), alpha=int(220 * intensity),
        )
