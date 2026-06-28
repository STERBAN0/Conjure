"""Tests for effects/options_panel.py — slider maths + mouse interaction.

The panel is built so construction and event handling need no display surface
(fonts are created lazily at render time), which keeps these tests headless.
A tiny fake SoundManager records the calls the panel makes against it.
"""

from __future__ import annotations

import types

import pygame
import pytest

from effects.options_panel import OptionsPanel, volume_from_mouse_x


class _FakeSound:
    """Minimal stand-in exposing the SoundManager surface the panel uses."""

    def __init__(self, volume: float = 0.8, muted: bool = False) -> None:
        self._v = volume
        self._m = muted
        self.calls: list[tuple[str, object]] = []

    @property
    def master_volume(self) -> float:
        return self._v

    @property
    def is_muted(self) -> bool:
        return self._m

    def set_master_volume(self, v: float) -> None:
        self._v = max(0.0, min(1.0, v))
        self.calls.append(("set_master_volume", self._v))

    def toggle_muted(self) -> bool:
        self._m = not self._m
        self.calls.append(("toggle_muted", self._m))
        return self._m


def _down(pos):
    return types.SimpleNamespace(type=pygame.MOUSEBUTTONDOWN, button=1, pos=pos)


def _motion(pos):
    return types.SimpleNamespace(type=pygame.MOUSEMOTION, pos=pos)


def _up(pos):
    return types.SimpleNamespace(type=pygame.MOUSEBUTTONUP, button=1, pos=pos)


class TestVolumeFromMouseX:
    def test_left_edge_is_zero(self):
        assert volume_from_mouse_x(100, 100, 200) == 0.0

    def test_right_edge_is_one(self):
        assert volume_from_mouse_x(300, 100, 200) == 1.0

    def test_midpoint_is_half(self):
        assert volume_from_mouse_x(200, 100, 200) == pytest.approx(0.5)

    def test_clamps_outside_track(self):
        assert volume_from_mouse_x(0, 100, 200) == 0.0
        assert volume_from_mouse_x(9999, 100, 200) == 1.0

    def test_zero_width_track_is_safe(self):
        assert volume_from_mouse_x(50, 100, 0) == 0.0


class TestPanelInteraction:
    def _open_panel(self, **kw):
        sound = _FakeSound(**kw)
        panel = OptionsPanel(1280, 720, sound)
        panel.toggle()  # open it
        return panel, sound

    def test_closed_panel_ignores_events(self):
        sound = _FakeSound()
        panel = OptionsPanel(1280, 720, sound)
        assert panel.handle_event(_down((640, 360))) is False
        assert sound.calls == []

    def test_clicking_checkbox_toggles_mute(self):
        panel, sound = self._open_panel()
        assert panel.handle_event(_down(panel._checkbox_hit.center)) is True
        assert ("toggle_muted", True) in sound.calls

    def test_clicking_slider_sets_volume(self):
        panel, sound = self._open_panel(volume=0.2)
        mid_x = panel._slider_left + panel._slider_width // 2
        assert panel.handle_event(_down((mid_x, panel._slider_y))) is True
        assert sound.master_volume == pytest.approx(0.5, abs=0.02)

    def test_drag_updates_then_release_stops(self):
        panel, sound = self._open_panel(volume=0.0)
        panel.handle_event(_down((panel._slider_left, panel._slider_y)))
        # Drag past the right edge → clamps to full volume.
        panel.handle_event(
            _motion((panel._slider_left + panel._slider_width + 80, panel._slider_y))
        )
        assert sound.master_volume == 1.0
        panel.handle_event(_up((0, 0)))
        # After release, further motion must not change the volume.
        panel.handle_event(_motion((panel._slider_left, panel._slider_y)))
        assert sound.master_volume == 1.0

    def test_slider_ignored_while_muted(self):
        panel, sound = self._open_panel(muted=True)
        mid_x = panel._slider_left + panel._slider_width // 2
        panel.handle_event(_down((mid_x, panel._slider_y)))
        assert all(c[0] != "set_master_volume" for c in sound.calls)
