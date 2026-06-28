"""In-app audio options overlay (toggled with the O key).

A small centred panel with two controls, driven by the mouse:

  - a **Mute audio** checkbox, and
  - a **volume slider** (shown only while not muted) that scales the SFX
    master volume from 0–100 %.

Both feed a :class:`audio.sounds.SoundManager` through its public
``set_master_volume`` / ``toggle_muted`` API, so changes are applied live.

The panel is deliberately self-contained: ``handle_event`` owns all the
mouse hit-testing and returns ``True`` when it consumes an event, so the
orchestrator can ``continue`` past it. Fonts are created lazily on first
render, which keeps construction and the interaction logic importable and
testable without a display surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from audio.sounds import SoundManager

# Panel geometry (logical pixels — pygame's SCALED mode maps mouse events back
# into this same logical space, so hit-testing matches what's drawn). The panel
# is wide enough for the footer hint to sit comfortably on one line.
_PANEL_W = 560
_PANEL_H = 220
_PAD = 34

# Panel-local y offsets, shared by both render() and the screen-space hit rects
# built in __init__ so the drawing and the click targets can never drift apart.
_TITLE_Y = 26
_CHECKBOX_Y = 84
_SLIDER_Y = 152
_FOOTER_FROM_BOTTOM = 30

_CHECKBOX_SIZE = 26
_SLIDER_W = 380
_SLIDER_H = 6
_KNOB_R = 9

# Colours, matched to effects/hud.py's controls overlay (square 1px border).
_BG = (8, 12, 22, 232)
_BORDER = (92, 132, 182)
_TITLE_C = (236, 243, 255)
_TEXT_C = (208, 220, 238)
_DIM_C = (120, 135, 155)
_ACCENT = (140, 220, 255)
_HINT_C = (120, 140, 170)


def volume_from_mouse_x(mouse_x: int, track_left: int, track_width: int) -> float:
    """Map an x pixel on the slider track to a 0.0–1.0 volume (clamped).

    Pure helper so the slider maths can be unit-tested without pygame state.
    """
    if track_width <= 0:
        return 0.0
    frac = (mouse_x - track_left) / track_width
    if frac < 0.0:
        return 0.0
    if frac > 1.0:
        return 1.0
    return frac


class OptionsPanel:
    """Mouse-driven audio options overlay (mute checkbox + volume slider)."""

    def __init__(self, width: int, height: int, sound: SoundManager) -> None:
        self._width = width
        self._height = height
        self._sound = sound
        self._open = False
        self._dragging = False

        # Fonts are created lazily (see _ensure_fonts) so construction needs no
        # display/font init — keeps handle_event testable headless.
        self._font_title: pygame.font.Font | None = None
        self._font_row: pygame.font.Font | None = None
        self._font_hint: pygame.font.Font | None = None

        # --- fixed geometry (screen space) ----------------------------------
        px = (width - _PANEL_W) // 2
        py = (height - _PANEL_H) // 2
        self._panel_rect = pygame.Rect(px, py, _PANEL_W, _PANEL_H)

        # Checkbox box + a generous click target spanning its label.
        box_x = px + _PAD
        box_y = py + _CHECKBOX_Y
        self._checkbox_box = pygame.Rect(box_x, box_y, _CHECKBOX_SIZE, _CHECKBOX_SIZE)
        self._checkbox_hit = pygame.Rect(box_x - 6, box_y - 8, 300, _CHECKBOX_SIZE + 16)

        # Slider track + a taller grab area so the knob is easy to catch.
        self._slider_left = px + _PAD
        self._slider_width = _SLIDER_W
        self._slider_y = py + _SLIDER_Y
        self._slider_hit = pygame.Rect(
            self._slider_left - _KNOB_R,
            self._slider_y - 14,
            _SLIDER_W + _KNOB_R * 2,
            28,
        )

    # -- public API ----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self._open = not self._open
        if not self._open:
            self._dragging = False

    def handle_event(self, event: object) -> bool:
        """Consume a mouse event while open. Returns True if it was handled.

        Non-mouse events (e.g. KEYDOWN) always return False so the caller's
        keyboard handling — including O/ESC to close — still runs.
        """
        if not self._open:
            return False

        etype = getattr(event, "type", None)

        if etype == pygame.MOUSEBUTTONDOWN and getattr(event, "button", None) == 1:
            pos = event.pos
            if self._checkbox_hit.collidepoint(pos):
                self._sound.toggle_muted()
                return True
            if not self._sound.is_muted and self._slider_hit.collidepoint(pos):
                self._dragging = True
                self._set_volume_from_x(pos[0])
                return True
            # Swallow any other click that lands on the panel so it doesn't fall
            # through to the app behind it.
            return bool(self._panel_rect.collidepoint(pos))

        if etype == pygame.MOUSEMOTION and self._dragging:
            self._set_volume_from_x(event.pos[0])
            return True

        if etype == pygame.MOUSEBUTTONUP and getattr(event, "button", None) == 1:
            if self._dragging:
                self._dragging = False
                return True

        return False

    def render(self, target: pygame.Surface) -> None:
        if not self._open:
            return
        self._ensure_fonts()

        panel = pygame.Surface((_PANEL_W, _PANEL_H), pygame.SRCALPHA)
        panel.fill(_BG)
        # Square 1px border to match the dark fill's corners and the K controls
        # overlay — no border_radius (which left the fill's corners poking out).
        pygame.draw.rect(panel, _BORDER, panel.get_rect(), width=1)

        # Title.
        title = self._font_title.render("AUDIO OPTIONS", True, _TITLE_C)
        panel.blit(title, ((_PANEL_W - title.get_width()) // 2, _TITLE_Y))

        muted = self._sound.is_muted

        # --- mute checkbox (panel-local coords, square to match the panel) ---
        box = pygame.Rect(_PAD, _CHECKBOX_Y, _CHECKBOX_SIZE, _CHECKBOX_SIZE)
        pygame.draw.rect(panel, _ACCENT, box, width=2)
        if muted:
            # Filled box + a tick so it clearly reads as "checked / muted".
            pygame.draw.rect(panel, _ACCENT, box.inflate(-8, -8))
            pygame.draw.lines(
                panel, (8, 12, 22), False,
                [(box.left + 6, box.centery),
                 (box.centerx - 1, box.bottom - 7),
                 (box.right - 5, box.top + 7)],
                3,
            )
        label = self._font_row.render("Mute audio", True, _TEXT_C)
        panel.blit(label, (box.right + 16, box.centery - label.get_height() // 2))

        # --- volume slider (only when not muted) ---
        track_cy = _SLIDER_Y + _SLIDER_H // 2
        if muted:
            note = self._font_row.render("(sound effects muted)", True, _DIM_C)
            panel.blit(note, (_PAD, _SLIDER_Y - 8))
        else:
            vol = self._sound.master_volume
            track = pygame.Rect(_PAD, _SLIDER_Y, _SLIDER_W, _SLIDER_H)
            pygame.draw.rect(panel, (255, 255, 255, 40), track, border_radius=3)
            fill_w = int(_SLIDER_W * vol)
            if fill_w > 0:
                pygame.draw.rect(
                    panel, _ACCENT,
                    pygame.Rect(_PAD, _SLIDER_Y, fill_w, _SLIDER_H),
                    border_radius=3,
                )
            knob_x = _PAD + fill_w
            pygame.draw.circle(panel, (240, 250, 255), (knob_x, track_cy), _KNOB_R)
            pygame.draw.circle(panel, _BORDER, (knob_x, track_cy), _KNOB_R, 1)
            pct = self._font_row.render(f"{int(round(vol * 100))}%", True, _TEXT_C)
            panel.blit(
                pct,
                (_PAD + _SLIDER_W + 22, track_cy - pct.get_height() // 2),
            )

        # Footer hint, centred (panel is wide enough for it on one line).
        hint = self._font_hint.render(
            "click the box to mute  ·  drag the slider  ·  O / ESC to close",
            True, _HINT_C,
        )
        panel.blit(
            hint,
            ((_PANEL_W - hint.get_width()) // 2, _PANEL_H - _FOOTER_FROM_BOTTOM),
        )

        target.blit(panel, self._panel_rect.topleft)

    # -- internals -----------------------------------------------------------

    def _set_volume_from_x(self, mouse_x: int) -> None:
        level = volume_from_mouse_x(mouse_x, self._slider_left, self._slider_width)
        self._sound.set_master_volume(level)

    def _ensure_fonts(self) -> None:
        if self._font_title is None:
            self._font_title = pygame.font.SysFont("consolas", 22, bold=True)
            self._font_row = pygame.font.SysFont("consolas", 18, bold=True)
            self._font_hint = pygame.font.SysFont("consolas", 13)
