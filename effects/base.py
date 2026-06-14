"""Base class for all Aether effects.

An effect is an object owned by the renderer. It has two roles:

  1. It optionally claims an ability (set ``ability_name``). The renderer
     only ticks effects whose ability is in flight (or always-on effects
     that leave ``ability_name`` empty).
  2. It draws to the screen each frame. Either as a foreground overlay
     (LAYER_FG) or as a background frame mutator (LAYER_BG, by overriding
     ``pre_process_frame``).

Effects can subscribe to ability lifecycle events on the hook bus by
overriding the ``on_*`` methods. The renderer is responsible for wiring
those callbacks up at construction time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from core.state import AbilityState, FrameState, GestureSignals

LAYER_BG = "background"
LAYER_FG = "foreground"


class Effect:
    """Base class. Override the lifecycle hooks you care about."""

    layer: str = LAYER_FG
    name: str = "effect"
    # If set, this effect is gated on the router's active ability matching.
    # An empty string means "always run".
    ability_name: str = ""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    # --- Lifecycle hooks (optional) -----------------------------------------

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        """Called once when the router enters this effect's ability."""

    def on_charge(
        self, charge: float, frame: FrameState, signals: GestureSignals
    ) -> None:
        """Called every frame while the ability charges."""

    def on_release(self, intensity: float, frame: FrameState) -> None:
        """Called once when charge completes and the release motion fires."""

    def on_active(self, frame: FrameState, signals: GestureSignals) -> None:
        """Called every frame during the post-release sustain phase."""

    def on_exit(self) -> None:
        """Called once when the ability exits (cooldown / pose loss)."""

    # --- Draw / tick --------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        """Advance simulation. ``ability`` is the router's snapshot."""

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        """Draw foreground content onto ``target``."""

    def pre_process_frame(
        self,
        frame_bgr: np.ndarray,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> np.ndarray:
        """LAYER_BG only: warp the BGR frame and return the result."""
        return frame_bgr

    # --- Convenience --------------------------------------------------------

    def is_gated(self) -> bool:
        """True if this effect runs only when its ability is active."""
        return bool(self.ability_name)
