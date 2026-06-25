"""Tests for the on-screen controls list (the K overlay) and its debug legend.

The K overlay and the debug-panel legend both render from one canonical list,
``_CONTROLS``, so these tests guard against the two drifting out of sync and
against documented keys (like the L laser toggle) going missing.
"""
from __future__ import annotations

import pygame

from effects.hud import _CONTROLS, HUD, _controls_legend_lines


def test_controls_list_includes_the_overlay_key() -> None:
    keys = {key for key, _short, _desc in _CONTROLS}
    assert "K" in keys  # the key that opens this very list


def test_controls_list_includes_laser_toggle() -> None:
    # L toggles Laser Eyes / face tracking; it must be documented in the list.
    keys = {key for key, _short, _desc in _CONTROLS}
    assert "L" in keys


def test_controls_keys_are_unique() -> None:
    keys = [key for key, _short, _desc in _CONTROLS]
    assert len(keys) == len(set(keys))


def test_debug_legend_covers_every_control_on_two_lines() -> None:
    lines = _controls_legend_lines()
    assert len(lines) == 2
    assert all(line.strip() for line in lines)
    joined = " ".join(lines)
    for key, _short, _desc in _CONTROLS:
        assert key in joined


def test_controls_overlay_panel_builds() -> None:
    # Smoke test: the cached panel composites without error and is non-empty.
    pygame.font.init()
    hud = HUD(640, 480)
    target = pygame.Surface((640, 480), pygame.SRCALPHA)
    hud.render_controls(target)
    panel = hud._controls_panel
    assert panel is not None
    assert panel.get_width() > 0 and panel.get_height() > 0
