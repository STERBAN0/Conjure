"""Heads-up display.

Two layers of info, each toggle-able from main:

  hud_minimal (default ON): the user-facing layer.
    - Big readable label of the active ability, top-centre.
    - Charge ring around the active hand(s) while CHARGING.
    - Faint "READY" indicator when fully charged + waiting for release.
    - Cooldown ring fades after release.

  hud_debug (default OFF, toggle D): for development.
    - Continuous signal numbers.
    - Pose match list with confidences.
    - Hand landmark dots + velocity arrows.
    - FPS.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pygame

from core.state import (
    AbilityState,
    FrameState,
    GestureSignals,
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_RELEASING,
)
from gestures.poses import PoseMatch


_PHASE_COLORS = {
    PHASE_CHARGING: (140, 200, 255),
    PHASE_ACTIVE: (255, 240, 180),
    PHASE_RELEASING: (255, 255, 255),
    PHASE_COOLDOWN: (200, 80, 120),
}

_ABILITY_LABELS = {
    "chidori": "CHIDORI",
    "kamehameha": "KAMEHAMEHA",
    "rasengan": "RASENGAN",
    "space_stretch": "SPACE STRETCH",
    "reality_tear": "REALITY TEAR",
}


class HUD:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self._fps_smoothed = 60.0

    # ------------------------------------------------------------------

    def render_minimal(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        matches: Iterable[PoseMatch],
    ) -> None:
        # Active ability banner
        if ability.active:
            label = _ABILITY_LABELS.get(ability.name, ability.name.upper())
            color = _PHASE_COLORS.get(ability.phase, (220, 230, 255))
            self._draw_text_with_glow(
                target, label, self.font_big,
                position=(self.width // 2, 60),
                color=color,
                anchor="midtop",
            )

            phase_label = ability.phase.upper()
            self._draw_text_with_glow(
                target, phase_label, self.font_small,
                position=(self.width // 2, 110),
                color=color,
                anchor="midtop",
                alpha=180,
            )

        # Charge ring on the primary hand
        if ability.phase == PHASE_CHARGING and ability.primary_hand is not None:
            self._draw_charge_ring(
                target,
                center_norm=ability.primary_hand.palm,
                charge=ability.charge,
                color=(140, 220, 255),
            )
            if ability.secondary_hand is not None:
                self._draw_charge_ring(
                    target,
                    center_norm=ability.secondary_hand.palm,
                    charge=ability.charge,
                    color=(140, 220, 255),
                )

        # READY indicator when charged + waiting for release motion
        if (
            ability.phase == PHASE_CHARGING
            and ability.charge >= 0.999
            and ability.primary_hand is not None
        ):
            cx = int(ability.primary_hand.palm[0] * self.width)
            cy = int(ability.primary_hand.palm[1] * self.height) - 80
            self._draw_text_with_glow(
                target, "READY", self.font_med,
                position=(cx, cy),
                color=(255, 240, 120),
                anchor="center",
            )

    # ------------------------------------------------------------------

    def render_debug(
        self,
        target: pygame.Surface,
        frame: FrameState,
        signals: GestureSignals,
        ability: AbilityState,
        matches: Iterable[PoseMatch],
        fps: float,
    ) -> None:
        # Smoothed FPS
        self._fps_smoothed = self._fps_smoothed * 0.9 + fps * 0.1

        hand_labels = [
            f"{h.label}:{h.tracking_confidence:.2f}" for h in frame.hands
        ]
        lines = [
            f"fps          : {self._fps_smoothed:5.1f}",
            f"hands        : {hand_labels}",
            f"span         : {signals.span:5.2f}    expansion: {signals.expansion:+.2f}",
            f"grip         : {signals.grip:5.2f}    motion   : {signals.motion_energy:5.2f}",
            f"rotation     : {signals.rotation:+.2f} rad/s",
            f"time scale   : {signals.time_scale:5.2f}",
            f"audio lvl    : {signals.audio_level:5.2f}",
            "",
            f"ability      : {ability.name or '(idle)'}",
            f"phase        : {ability.phase}",
            f"charge       : {ability.charge:5.2f}    age: {ability.age:5.2f}s",
            f"intensity    : {ability.intensity:5.2f}",
            "",
            "matches:",
        ]
        for m in sorted(matches, key=lambda x: -x.confidence):
            lines.append(f"  {m.name:<14} {m.confidence:.2f}")
        lines.append("")
        lines.append("ESC/Q quit  H hud  D debug  S screenshot")

        bg = pygame.Surface((360, 22 * len(lines) + 18), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        target.blit(bg, (10, 10))
        for i, line in enumerate(lines):
            target.blit(
                self.font_small.render(line, True, (220, 230, 255)),
                (20, 18 + i * 18),
            )

        # Hand landmark dots
        for h in frame.hands:
            for lm in h.landmarks:
                x = int(lm[0] * self.width)
                y = int(lm[1] * self.height)
                pygame.draw.circle(target, (255, 255, 255), (x, y), 2)
            color = (120, 220, 255) if h.label == "Right" else (255, 180, 120)
            pygame.draw.circle(target, color, h.palm_px, 8, 2)

    # ------------------------------------------------------------------

    def _draw_charge_ring(
        self,
        target: pygame.Surface,
        center_norm: np.ndarray,
        charge: float,
        color: tuple[int, int, int],
    ) -> None:
        cx = int(center_norm[0] * self.width)
        cy = int(center_norm[1] * self.height)
        radius = 60
        thickness = 5

        ring = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
        # Background track
        pygame.draw.circle(
            ring, (255, 255, 255, 50),
            (radius + 2, radius + 2), radius, thickness,
        )
        # Charge arc — pygame doesn't have a clean "arc" w/ alpha + thickness
        # so we approximate with N segments.
        segs = 64
        active_segs = int(segs * charge)
        for i in range(active_segs):
            a0 = -math.pi / 2 + (i / segs) * math.tau
            a1 = -math.pi / 2 + ((i + 1) / segs) * math.tau
            x0 = (radius + 2) + math.cos(a0) * radius
            y0 = (radius + 2) + math.sin(a0) * radius
            x1 = (radius + 2) + math.cos(a1) * radius
            y1 = (radius + 2) + math.sin(a1) * radius
            pygame.draw.line(ring, (*color, 230), (x0, y0), (x1, y1), thickness)
        target.blit(
            ring, (cx - radius - 2, cy - radius - 2),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    def _draw_text_with_glow(
        self,
        target: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        position: tuple[int, int],
        color: tuple[int, int, int],
        anchor: str = "topleft",
        alpha: int = 255,
    ) -> None:
        # Glow: render text on a small surface, blur with a few offset blits.
        rendered = font.render(text, True, color)
        rect = rendered.get_rect(**{anchor: position})

        glow = pygame.Surface(rendered.get_size(), pygame.SRCALPHA)
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)):
            tinted = font.render(text, True, color)
            tinted.set_alpha(60)
            glow.blit(tinted, (dx + 2, dy + 2),
                      special_flags=pygame.BLEND_RGBA_ADD)
        target.blit(
            glow, rect.move(-2, -2).topleft,
            special_flags=pygame.BLEND_RGBA_ADD,
        )
        rendered.set_alpha(alpha)
        target.blit(rendered, rect.topleft)
