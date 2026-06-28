"""Heads-up display.

Two layers of info, each toggle-able from main:

  hud_minimal (default ON): the user-facing layer.
    - Big readable label of the active ability, top-centre.
    - Charge ring around the active hand(s) while CHARGING.
    - Slim linear charge bar + integer % under the label.
    - Fire-hint prompt while charging ("THRUST TO STRIKE", etc.).
    - Faint "READY" indicator when fully charged + waiting for release.
    - Cooldown sweep indicator during PHASE_COOLDOWN.
    - Compact roster strip along the right edge (config.HUD_SHOW_ROSTER).

  hud_debug (default OFF, toggle D): for development.
    - Continuous signal numbers.
    - Pose match list with confidences.
    - Hand landmark dots + velocity arrows.
    - Face mesh "mask" + eye open/shut state + gaze vector (Laser Eyes).
    - FPS.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pygame

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_RELEASING,
    AbilityState,
    FrameState,
    GestureSignals,
)
from gestures.poses import PoseMatch

# Deep, production-grade fills — deliberately DARK (ink-toned) so the text reads
# as solid dark glyphs on light/busy backgrounds; _draw_text_with_outline wraps
# them in a bright near-opaque halo + a dark drop-shadow so they ALSO stay legible
# on dark backgrounds. Dark fill + bright ring = visible on any webcam feed.
_PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    PHASE_CHARGING: (16, 40, 104),
    PHASE_ACTIVE: (120, 50, 4),
    PHASE_RELEASING: (118, 22, 42),
    PHASE_COOLDOWN: (92, 16, 46),
}

# Shared dark text fills used outside the phase palette.
_TEXT_READY: tuple[int, int, int] = (126, 58, 2)        # fully-charged / fire prompt
_TEXT_CHARGING: tuple[int, int, int] = (16, 40, 108)    # charging fire hint
_TEXT_PCT: tuple[int, int, int] = (16, 38, 96)          # charge percentage
_TEXT_LASER_ON: tuple[int, int, int] = (120, 24, 24)    # laser-eyes status (ON)
_TEXT_LASER_OFF: tuple[int, int, int] = (50, 54, 66)    # laser-eyes status (OFF)

# Debug skeleton: MediaPipe hand bone connections + per-finger state colours.
_HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
)
_FINGER_TIP_IDX: tuple[int, ...] = (4, 8, 12, 16, 20)
_FINGER_NAMES: tuple[str, ...] = ("th", "ix", "md", "rg", "pk")
_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "ext": (120, 255, 140),   # extended → green
    "amb": (255, 220, 90),    # ambiguous (dead-zone) → yellow
    "fold": (255, 110, 110),  # folded → red
}


def _finger_state_label(value: float) -> str:
    """Match gestures.poses dead-zone classification for the debug readout."""
    if value >= config.SINGLE_FINGER_EXTENDED:
        return "ext"
    if value <= config.SINGLE_FINGER_FOLDED:
        return "fold"
    return "amb"

_ABILITY_LABELS: dict[str, str] = {
    "chidori": "CHIDORI",
    "kamehameha": "KAMEHAMEHA",
    "rasengan": "RASENGAN",
    "fireball": "FIREBALL",
    "frost_nova": "FROST NOVA",
    "laser_eyes": "LASER EYES",
    "space_stretch": "SPACE STRETCH",
    "reality_tear": "REALITY TEAR",
    "time_freeze": "TIME FREEZE",
}

# How-to-fire text shown while the ability is charging so the user always
# knows what motion triggers the release.
_ABILITY_FIRE_HINT: dict[str, str] = {
    "chidori": "HOLD",
    "kamehameha": "PUSH AT SCREEN TO FIRE",
    "rasengan": "FLICK TO THROW",
    "fireball": "FLICK FINGER TO SHOOT",
    "frost_nova": "SPREAD TO BURST",
    "laser_eyes": "EYES SHUT TO CHARGE · OPEN TO FIRE",
    "space_stretch": "JUST PULL APART",
    "reality_tear": "HOLD",
    "time_freeze": "HOLD",
}

# Ordered list for the right-edge roster strip.
_ROSTER_ENTRIES: list[tuple[str, str]] = [
    ("chidori", "Chidori"),
    ("kamehameha", "Kamehameha"),
    ("rasengan", "Rasengan"),
    ("fireball", "Fireball"),
    ("frost_nova", "Frost Nova"),
    ("laser_eyes", "Laser Eyes"),
    ("space_stretch", "Space Stretch"),
    ("reality_tear", "Reality Tear"),
    ("time_freeze", "Time Freeze"),
]

# Canonical keyboard controls — the single source of truth for BOTH the on-screen
# controls overlay (toggled with K) and the debug panel's compact legend, so the
# two can never drift apart. Each entry is (key, short_label, description); the
# debug legend uses the short label, the overlay uses the full description.
_CONTROLS: tuple[tuple[str, str, str], ...] = (
    ("K", "controls", "show / hide this controls list"),
    ("O", "options", "audio options — mute / volume slider"),
    ("H", "hud", "toggle the minimal HUD"),
    ("D", "debug", "toggle the debug overlay"),
    ("M", "manual", "open the hand-sign manual  (←/→ to page)"),
    ("L", "laser", "toggle Laser Eyes on / off"),
    ("R", "clear", "clear the Laser Eyes drawing"),
    ("S", "screenshot", "save a screenshot to ./screenshots"),
    ("ESC", "close", "close the manual / this list"),
    ("Q", "quit", "quit Conjure"),
)


def _controls_legend_lines() -> list[str]:
    """Two compact lines for the debug panel, derived from `_CONTROLS`.

    Split in half so the panel grows DOWN rather than WIDE (keeps each line
    narrower than the long debug rows that govern the panel width).
    """
    parts = [f"{key} {short}" for key, short, _desc in _CONTROLS]
    mid = (len(parts) + 1) // 2
    return [" | ".join(parts[:mid]), " | ".join(parts[mid:])]


class HUD:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.font_big = pygame.font.SysFont("consolas", 36, bold=True)
        self.font_med = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14)
        self._fps_smoothed = 60.0

        # --- render caches ---------------------------------------------------
        # _text_cache: (text, color, font_id, alpha, halo_color, halo_px,
        #               shadow_color, shadow_offset) → pre-composited Surface
        self._text_cache: dict[tuple, pygame.Surface] = {}

        # _charge_ring_cache: (color, charge_bucket_pct, radius, thickness)
        #   → pre-drawn ring Surface (before blitting).
        # charge_bucket_pct is charge rounded to the nearest 1 % (int 0–100).
        self._charge_ring_cache: dict[tuple, pygame.Surface] = {}

        # _cd_label: pre-rendered "CD" text surface (colour stays constant).
        cd_color = _PHASE_COLORS[PHASE_COOLDOWN]
        self._cd_label: pygame.Surface = self.font_small.render("CD", True, cd_color)
        # _cd_ring_cache: fraction_bucket (int 0–100) → pre-drawn arc Surface
        self._cd_ring_cache: dict[int, pygame.Surface] = {}

        # _debug_cache: last rendered debug panel + the key that produced it.
        self._debug_panel: pygame.Surface | None = None
        self._debug_panel_key: tuple | None = None

        # _controls_panel: the K-overlay is static, so build it once on first use.
        self._controls_panel: pygame.Surface | None = None

    # ------------------------------------------------------------------
    # Public API (keep existing signatures intact)
    # ------------------------------------------------------------------

    def render_minimal(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        matches: Iterable[PoseMatch],
        face_enabled: bool = True,
    ) -> None:
        matches_list = list(matches)

        # Active ability banner
        if ability.active:
            label = _ABILITY_LABELS.get(ability.name, ability.name.upper())
            color = _PHASE_COLORS.get(ability.phase, (220, 230, 255))
            self._draw_text_with_outline(
                target, label, self.font_big,
                position=(self.width // 2, 60),
                color=color,
                anchor="midtop",
            )
            self._draw_text_with_outline(
                target, ability.phase.upper(), self.font_small,
                position=(self.width // 2, 110),
                color=color,
                anchor="midtop",
                alpha=180,
            )

        # Charge rings on the active hand(s) — laser_eyes has no hand
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

        # Charge bar + % under the label, and fire-hint prompt
        if ability.phase == PHASE_CHARGING and ability.active:
            self._draw_charge_bar(target, ability)
            self._draw_fire_hint(target, ability)

        # READY indicator when fully charged, waiting for release motion
        if (
            ability.phase == PHASE_CHARGING
            and ability.charge >= 0.999
            and ability.primary_hand is not None
        ):
            cx = int(ability.primary_hand.palm[0] * self.width)
            cy = int(ability.primary_hand.palm[1] * self.height) - 80
            self._draw_text_with_outline(
                target, "READY", self.font_med,
                position=(cx, cy),
                color=_TEXT_READY,
                anchor="center",
            )

        # Cooldown sweep in bottom-centre
        if ability.phase == PHASE_COOLDOWN:
            self._draw_cooldown_indicator(target, ability)

        # Right-edge roster (shows all abilities, highlights detected ones)
        if config.HUD_SHOW_ROSTER:
            self._draw_roster(target, matches_list, ability)

        # Laser Eyes status indicator (top-left): ON when face tracking is live,
        # OFF when the face model is unavailable. It's a status read-out only — the
        # ability itself is gesture-driven (eyes shut to charge, open to fire).
        laser_status = "LASER EYES: ON" if face_enabled else "LASER EYES: OFF"
        laser_color = _TEXT_LASER_ON if face_enabled else _TEXT_LASER_OFF
        self._draw_text_with_outline(
            target, laser_status, self.font_small,
            position=(10, 10),
            color=laser_color,
            anchor="topleft",
        )

        # Discoverability hint for the controls overlay (top-right) so the K key
        # isn't invisible. Black fill + the bright halo from _draw_text_with_outline
        # keeps it readable on bright, dark, or busy backgrounds alike.
        self._draw_text_with_outline(
            target, "PRESS K FOR CONTROLS", self.font_small,
            position=(self.width - 10, 10),
            color=(8, 10, 16),
            anchor="topright",
        )

    # ------------------------------------------------------------------

    def render_controls(self, target: pygame.Surface) -> None:
        """Centred overlay listing every keyboard command (toggled with K).

        The panel is static, so it's composited once and cached — subsequent
        frames pay only a single blit.
        """
        if self._controls_panel is None:
            self._controls_panel = self._build_controls_panel()
        panel = self._controls_panel
        rect = panel.get_rect(center=(self.width // 2, self.height // 2))
        target.blit(panel, rect)

    def _build_controls_panel(self) -> pygame.Surface:
        """Compose the controls overlay surface from the canonical _CONTROLS list."""
        title_color = (236, 243, 255)
        key_color = (140, 220, 255)
        desc_color = (208, 220, 238)

        title = self.font_med.render("CONTROLS", True, title_color)
        rows = [
            (
                self.font_med.render(key, True, key_color),
                self.font_small.render(desc, True, desc_color),
            )
            for key, _short, desc in _CONTROLS
        ]

        pad = 22
        row_pitch = 30
        gap = 16  # space between the key column and the description column
        key_col_w = max(k.get_width() for k, _ in rows) + gap
        desc_col_w = max(d.get_width() for _, d in rows)
        content_w = max(title.get_width(), key_col_w + desc_col_w)
        panel_w = content_w + pad * 2
        panel_h = pad * 2 + title.get_height() + 16 + row_pitch * len(rows)

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((8, 12, 22, 226))
        pygame.draw.rect(panel, (92, 132, 182), panel.get_rect(), width=1)

        panel.blit(title, ((panel_w - title.get_width()) // 2, pad))
        y = pad + title.get_height() + 16
        for key_surf, desc_surf in rows:
            panel.blit(key_surf, (pad, y))
            # Vertically centre the description against the (taller) key glyph.
            desc_y = y + (key_surf.get_height() - desc_surf.get_height()) // 2
            panel.blit(desc_surf, (pad + key_col_w, desc_y))
            y += row_pitch
        return panel

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
        self._fps_smoothed = self._fps_smoothed * 0.9 + fps * 0.1

        hand_labels = [
            f"{h.label}:{h.tracking_confidence:.2f}" for h in frame.hands
        ]
        face = frame.face
        if face is not None and getattr(face, "present", False):
            face_line = (
                f"face         : eyes={'SHUT' if face.both_eyes_closed else 'open '} "
                f"closed={face.eyes_closed_duration:4.2f}s "
                f"gaze=({float(face.gaze[0]):+.2f},{float(face.gaze[1]):+.2f})"
            )
        else:
            face_line = "face         : (not tracked)"
        lines = [
            f"fps          : {self._fps_smoothed:5.1f}",
            f"hands        : {hand_labels}",
            f"span         : {signals.span:5.2f}    expansion: {signals.expansion:+.2f}",
            f"grip         : {signals.grip:5.2f}    motion   : {signals.motion_energy:5.2f}",
            f"rotation     : {signals.rotation:+.2f} rad/s",
            f"time scale   : {signals.time_scale:5.2f}",
            f"audio lvl    : {signals.audio_level:5.2f}",
            face_line,
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
        # Controls legend, derived from the canonical _CONTROLS list so it stays
        # in lockstep with the K overlay. Two lines keep the panel narrow.
        lines.extend(_controls_legend_lines())

        # Cache the debug panel by its content so the ~20 font renders + surface
        # alloc only happen when the text actually changes (typically once per
        # second for the FPS line; everything else changes only when state changes).
        panel_key = tuple(lines)
        if panel_key != self._debug_panel_key:
            rendered = [self.font_small.render(line, True, (220, 230, 255)) for line in lines]
            line_pitch = 18
            panel_w = max((s.get_width() for s in rendered), default=0) + 20
            panel_h = line_pitch * len(lines) + 16
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            panel.fill((0, 0, 0, 160))
            for i, surf in enumerate(rendered):
                panel.blit(surf, (10, 2 + i * line_pitch))
            self._debug_panel = panel
            self._debug_panel_key = panel_key

        if self._debug_panel is not None:
            target.blit(self._debug_panel, (10, 10))

        # Per-hand skeleton + colour-coded finger states.
        for h in frame.hands:
            self._draw_hand_debug(target, h)

        # Face mesh "mask" + eye state + gaze, the laser-eyes equivalent of the
        # hand skeleton (only present while face tracking is on).
        if face is not None and getattr(face, "present", False):
            self._draw_face_debug(target, face)

    def _draw_face_debug(self, target: pygame.Surface, face) -> None:
        """Draw the face mesh, eye open/shut state, and gaze vector.

        This is the Laser Eyes counterpart to the hand skeleton: every mesh
        landmark is a faint dot (the "mask"), the two eyes are ringed green when
        open / red when shut (charging), and an arrow shows the gaze direction
        the beams follow. ``pygame.draw`` clips to the surface, so landmarks that
        fall slightly outside the frame are handled safely.
        """
        w, hgt = self.width, self.height
        closed = bool(face.both_eyes_closed)

        # Mesh dots — the "mask".
        if face.landmarks is not None:
            mesh_color = (90, 200, 160)
            for lm in face.landmarks:
                pygame.draw.circle(
                    target, mesh_color,
                    (int(float(lm[0]) * w), int(float(lm[1]) * hgt)), 1,
                )

        # Face centre marker.
        fc = face.face_center
        pygame.draw.circle(
            target, (255, 220, 90),
            (int(float(fc[0]) * w), int(float(fc[1]) * hgt)), 3, 1,
        )

        # Eyes — ringed, coloured by closed state.
        eye_color = (255, 110, 110) if closed else (120, 255, 140)
        for eye in (face.left_eye_px, face.right_eye_px):
            pygame.draw.circle(target, eye_color, eye, 8, 2)

        # Gaze vector from the eye midpoint (the direction the beams travel).
        mx = (face.left_eye_px[0] + face.right_eye_px[0]) // 2
        my = (face.left_eye_px[1] + face.right_eye_px[1]) // 2
        gaze = face.gaze
        ex = int(mx + float(gaze[0]) * config.HUD_DEBUG_GAZE_ARROW_PX)
        ey = int(my + float(gaze[1]) * config.HUD_DEBUG_GAZE_ARROW_PX)
        pygame.draw.line(target, (120, 220, 255), (mx, my), (ex, ey), 2)
        pygame.draw.circle(target, (120, 220, 255), (ex, ey), 4)

    def _draw_hand_debug(self, target: pygame.Surface, h) -> None:
        """Draw one hand's bone skeleton, fingertip states, and a readout.

        Fingertips and the text are colour-coded green/yellow/red for
        extended / ambiguous / folded so you can see exactly which finger the
        classifier thinks is up or down.
        """
        w, hgt = self.width, self.height
        pts = [(int(lm[0] * w), int(lm[1] * hgt)) for lm in h.landmarks]

        for a, b in _HAND_BONES:
            pygame.draw.line(target, (90, 140, 180), pts[a], pts[b], 2)
        for p in pts:
            pygame.draw.circle(target, (200, 220, 255), p, 2)
        for tip, value in zip(_FINGER_TIP_IDX, h.fingers_open, strict=False):
            state = _finger_state_label(float(value))
            pygame.draw.circle(target, _STATE_COLORS[state], pts[tip], 6)

        palm_color = (120, 220, 255) if h.label == "Right" else (255, 180, 120)
        pygame.draw.circle(target, palm_color, h.palm_px, 9, 2)

        # Finger-state readout pinned under the palm.
        cells = "  ".join(
            f"{n}:{float(v):.2f}{_finger_state_label(float(v))[0].upper()}"
            for n, v in zip(_FINGER_NAMES, h.fingers_open, strict=False)
        )
        text = f"{h.label} c={h.tracking_confidence:.2f}  {cells}"
        surf = self.font_small.render(text, True, (235, 242, 255))
        bx = max(4, min(h.palm_px[0] - surf.get_width() // 2, w - surf.get_width() - 4))
        by = min(h.palm_px[1] + 56, hgt - 22)
        bg = pygame.Surface((surf.get_width() + 8, surf.get_height() + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        target.blit(bg, (bx - 4, by - 2))
        target.blit(surf, (bx, by))

    # ------------------------------------------------------------------
    # Private helpers
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

        # Cache the ring surface by (color, charge %) — charge changes at most
        # 100 unique values, so the cache stays tiny while eliminating the 64-seg
        # loop + SRCALPHA alloc on every frame the ring is visible.
        charge_bucket = int(min(charge, 1.0) * 100)
        cache_key = (color, charge_bucket, radius, thickness)
        ring = self._charge_ring_cache.get(cache_key)
        if ring is None:
            ring = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
            # Background track
            pygame.draw.circle(
                ring, (255, 255, 255, 50),
                (radius + 2, radius + 2), radius, thickness,
            )
            # Charge arc approximated with N line segments
            segs = 64
            active_segs = int(segs * charge_bucket / 100)
            for i in range(active_segs):
                a0 = -math.pi / 2 + (i / segs) * math.tau
                a1 = -math.pi / 2 + ((i + 1) / segs) * math.tau
                x0 = (radius + 2) + math.cos(a0) * radius
                y0 = (radius + 2) + math.sin(a0) * radius
                x1 = (radius + 2) + math.cos(a1) * radius
                y1 = (radius + 2) + math.sin(a1) * radius
                pygame.draw.line(ring, (*color, 230), (x0, y0), (x1, y1), thickness)
            self._charge_ring_cache[cache_key] = ring

        target.blit(
            ring, (cx - radius - 2, cy - radius - 2),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    def _draw_charge_bar(
        self,
        target: pygame.Surface,
        ability: AbilityState,
    ) -> None:
        """Slim horizontal bar + integer % just below the phase label."""
        bar_w = 220
        bar_h = 6
        bx = self.width // 2 - bar_w // 2
        by = 130  # just below the phase label at y=110

        # Track (faint)
        track = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        track.fill((255, 255, 255, 40))
        target.blit(track, (bx, by))

        # Fill
        fill_w = int(bar_w * min(ability.charge, 1.0))
        if fill_w > 0:
            bar_color = _PHASE_COLORS.get(ability.phase, (140, 200, 255))
            fill = pygame.Surface((fill_w, bar_h), pygame.SRCALPHA)
            fill.fill((*bar_color, 200))
            target.blit(fill, (bx, by))

        # Percentage
        pct = int(min(ability.charge, 1.0) * 100)
        self._draw_text_with_outline(
            target, f"{pct}%", self.font_small,
            position=(self.width // 2, by + bar_h + 4),
            color=_TEXT_PCT,
            anchor="midtop",
        )

    def _draw_fire_hint(
        self,
        target: pygame.Surface,
        ability: AbilityState,
    ) -> None:
        """Show the fire-method hint near the active hand while charging."""
        hint = _ABILITY_FIRE_HINT.get(ability.name)
        if not hint:
            return

        # Anchor near the primary hand; laser_eyes has no hand so use screen bottom
        if ability.primary_hand is not None:
            cx = int(ability.primary_hand.palm[0] * self.width)
            cy = int(ability.primary_hand.palm[1] * self.height) + 75
            cy = min(cy, self.height - 40)
        else:
            cx = self.width // 2
            cy = self.height - 80

        # Deep orange-gold when fully charged, deep blue while charging.
        color: tuple[int, int, int] = (
            _TEXT_READY if ability.charge >= 0.999 else _TEXT_CHARGING
        )
        alpha = 140 + int(115 * min(ability.charge, 1.0))
        self._draw_text_with_outline(
            target, f"► {hint}", self.font_med,
            position=(cx, cy),
            color=color,
            anchor="center",
            alpha=alpha,
        )

    def _draw_cooldown_indicator(
        self,
        target: pygame.Surface,
        ability: AbilityState,
    ) -> None:
        """Small arc sweep at screen bottom-centre showing cooldown remaining."""
        cooldown_total = config.ABILITY_COOLDOWN.get(ability.name, 0.5)
        if cooldown_total <= 0:
            return
        fraction = max(0.0, 1.0 - ability.phase_age / cooldown_total)

        cx, cy = self.width // 2, self.height - 50
        radius = 22
        thickness = 4

        # Cache the arc surface by fraction bucket (1 % steps).
        fraction_bucket = int(fraction * 100)
        surf = self._cd_ring_cache.get(fraction_bucket)
        if surf is None:
            surf = pygame.Surface((radius * 2 + 6, radius * 2 + 6), pygame.SRCALPHA)
            pygame.draw.circle(
                surf, (255, 255, 255, 30), (radius + 3, radius + 3), radius, thickness,
            )
            segs = 48
            active_segs = int(segs * fraction_bucket / 100)
            cd_color = _PHASE_COLORS[PHASE_COOLDOWN]
            for i in range(active_segs):
                a0 = -math.pi / 2 + (i / segs) * math.tau
                a1 = -math.pi / 2 + ((i + 1) / segs) * math.tau
                x0 = (radius + 3) + math.cos(a0) * radius
                y0 = (radius + 3) + math.sin(a0) * radius
                x1 = (radius + 3) + math.cos(a1) * radius
                y1 = (radius + 3) + math.sin(a1) * radius
                pygame.draw.line(surf, (*cd_color, 200), (x0, y0), (x1, y1), thickness)
            self._cd_ring_cache[fraction_bucket] = surf

        target.blit(
            surf, (cx - radius - 3, cy - radius - 3),
            special_flags=pygame.BLEND_RGBA_ADD,
        )
        # Use the pre-rendered "CD" label (rendered once in __init__).
        label = self._cd_label
        target.blit(label, (cx - label.get_width() // 2, cy - label.get_height() // 2))

    def _draw_roster(
        self,
        target: pygame.Surface,
        matches: list[PoseMatch],
        ability: AbilityState,
    ) -> None:
        """Compact right-edge strip listing all abilities with detection highlight."""
        match_names: set[str] = {m.name for m in matches if m.confidence >= 0.40}

        row_h = 22
        pad_x = 10
        strip_w = 162
        strip_h = row_h * len(_ROSTER_ENTRIES) + 30
        strip_x = self.width - strip_w - 6
        strip_y = (self.height - strip_h) // 2

        bg = pygame.Surface((strip_w, strip_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 110))
        target.blit(bg, (strip_x, strip_y))

        # Header hint
        header = self.font_small.render("M — manual", True, (140, 160, 180))
        target.blit(header, (strip_x + pad_x, strip_y + 4))

        for idx, (aid, aname) in enumerate(_ROSTER_ENTRIES):
            row_y = strip_y + 24 + idx * row_h
            is_active = ability.active and ability.name == aid
            is_matched = aid in match_names

            if is_active:
                hl = pygame.Surface((strip_w, row_h - 2), pygame.SRCALPHA)
                ac = _PHASE_COLORS.get(ability.phase, (140, 200, 255))
                hl.fill((*ac, 55))
                target.blit(hl, (strip_x, row_y))
                text_color: tuple[int, int, int] = _PHASE_COLORS.get(
                    ability.phase, (220, 240, 255)
                )
            elif is_matched:
                hl = pygame.Surface((strip_w, row_h - 2), pygame.SRCALPHA)
                hl.fill((140, 200, 255, 30))
                target.blit(hl, (strip_x, row_y))
                text_color = (180, 220, 255)
            else:
                text_color = (120, 135, 155)

            dot_c = text_color if (is_matched or is_active) else (55, 65, 80)
            pygame.draw.circle(
                target, dot_c, (strip_x + pad_x, row_y + row_h // 2), 3,
            )
            lbl = self.font_small.render(aname, True, text_color)
            target.blit(lbl, (strip_x + pad_x + 10, row_y + 2))

    def _draw_text_with_outline(
        self,
        target: pygame.Surface,
        text: str,
        font: pygame.font.Font,
        position: tuple[int, int],
        color: tuple[int, int, int],
        anchor: str = "topleft",
        alpha: int = 255,
        halo_color: tuple[int, int, int] = (244, 247, 252),
        halo_px: int = 1,
        shadow_color: tuple[int, int, int] = (4, 6, 12),
        shadow_offset: int = 2,
    ) -> None:
        """Render a dark, production-grade label that reads on ANY background.

        The fill (`color`) is a dark ink tone, so it contrasts on light/busy
        feeds. A bright, near-opaque 1-px halo rings it (legible on dark feeds)
        and a dark drop-shadow adds depth — together the glyphs always have both a
        light and a dark edge, so neither a bright nor a dark background can wash
        them out.

        Results are cached by (text, color, font id, alpha, halo_color, halo_px,
        shadow_color, shadow_offset) so repeated identical calls — which are the
        norm while the ability banner stays the same for several seconds — pay only
        one blit instead of 3 renders + 8 halo copies.
        """
        cache_key = (
            text, color, id(font), alpha,
            halo_color, halo_px, shadow_color, shadow_offset,
        )
        composite = self._text_cache.get(cache_key)
        if composite is None:
            # Build a temporary rendered surface to measure size.
            _sample = font.render(text, True, color)
            w = _sample.get_width() + abs(shadow_offset) + halo_px * 2
            h = _sample.get_height() + abs(shadow_offset) + halo_px * 2
            composite = pygame.Surface((w, h), pygame.SRCALPHA)

            # offsets relative to composite origin (halo_px margin at top-left)
            ox, oy = halo_px, halo_px

            # Drop-shadow
            shadow = font.render(text, True, shadow_color)
            shadow.set_alpha(int(alpha * 0.8))
            composite.blit(shadow, (ox + shadow_offset, oy + shadow_offset))

            # Halo copies
            halo = font.render(text, True, halo_color)
            halo_alpha = int(alpha * 0.85)
            for ddx in range(-halo_px, halo_px + 1):
                for ddy in range(-halo_px, halo_px + 1):
                    if ddx == 0 and ddy == 0:
                        continue
                    hc = halo.copy()
                    hc.set_alpha(halo_alpha)
                    composite.blit(hc, (ox + ddx, oy + ddy))

            # Foreground fill
            _sample.set_alpha(alpha)
            composite.blit(_sample, (ox, oy))

            self._text_cache[cache_key] = composite

        # Position the composite surface via the requested anchor.
        rect = composite.get_rect(**{anchor: position})
        # Composite origin includes the halo_px margin, so shift back by it.
        target.blit(composite, (rect.left - halo_px, rect.top - halo_px))
