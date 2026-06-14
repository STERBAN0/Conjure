"""Effects renderer.

Pipeline per frame:
    1. Background effects mutate the BGR camera frame (e.g. SpaceStretch).
    2. Frame is converted to a pygame surface and blitted.
    3. Foreground effects draw their layers on top, additively.

Effects are gated on the ability router. Effects with ``ability_name``
only run when the router is in their ability; effects with no
``ability_name`` run every frame.

The renderer subscribes effects to ability lifecycle events on the hook
bus, so effects can clean up their internal state on enter/exit.
"""

from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np
import pygame

import config
from core.hooks import HookBus
from core.state import (
    AbilityState,
    FrameState,
    GestureSignals,
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
)
from effects.base import LAYER_BG, LAYER_FG, Effect
from effects.chidori import ChidoriEffect
from effects.kamehameha import KamehamehaEffect
from effects.rasengan import RasenganEffect
from effects.reality_tear import RealityTearEffect
from effects.space_stretch import SpaceStretchEffect

log = logging.getLogger(__name__)


class EffectsRenderer:
    def __init__(self, width: int, height: int, hooks: HookBus) -> None:
        self.width = width
        self.height = height
        self.hooks = hooks
        self.effects: list[Effect] = []

    def add(self, effect: Effect) -> None:
        self.effects.append(effect)
        # Wire ability lifecycle hooks.
        if effect.is_gated():
            self._wire_lifecycle(effect)

    def _wire_lifecycle(self, effect: Effect) -> None:
        name = effect.ability_name

        def filtered(handler: Callable, *args, **kwargs):
            return handler(*args, **kwargs)

        self.hooks.on(
            "ability_enter",
            lambda n, frame, signals: effect.on_enter(frame, signals)
            if n == name else None,
        )
        self.hooks.on(
            "ability_charge",
            lambda n, charge, frame, signals: effect.on_charge(charge, frame, signals)
            if n == name else None,
        )
        self.hooks.on(
            "ability_release",
            lambda n, intensity, frame: effect.on_release(intensity, frame)
            if n == name else None,
        )
        self.hooks.on(
            "ability_active",
            lambda n, frame, signals: effect.on_active(frame, signals)
            if n == name else None,
        )
        self.hooks.on(
            "ability_exit",
            lambda n: effect.on_exit() if n == name else None,
        )

    def update_and_render(
        self,
        frame: FrameState,
        signals: GestureSignals,
        ability: AbilityState,
        target: pygame.Surface,
    ) -> None:
        signals.time_scale = 1.0
        scaled_dt = frame.dt

        # 1. BG effects mutate the BGR frame
        bgr = frame.frame_bgr
        active_bg: list[Effect] = []
        for e in self.effects:
            if e.layer != LAYER_BG:
                continue
            if not self._effect_active(e, ability):
                continue
            e.update(signals, scaled_dt, ability)
            bgr = e.pre_process_frame(bgr, signals, ability)
            active_bg.append(e)

        # 2. Camera -> pygame surface
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = cv2.resize(
                rgb, (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
        # frombuffer reads the array's buffer directly; ascontiguousarray is a
        # no-op when cvtColor/resize already produced a contiguous frame, so
        # this avoids the per-frame full-frame copy that tobytes() forced.
        rgb = np.ascontiguousarray(rgb)
        cam_surface = pygame.image.frombuffer(
            rgb, (self.width, self.height), "RGB",
        )
        target.blit(cam_surface, (0, 0))

        # 2b. BG effects may also draw a foreground guide layer after their
        # frame warp. This is how effects like SpaceStretch make the warped
        # field legible instead of being a subtle camera smear.
        for e in active_bg:
            e.render(target, signals, ability)

        # 3. FG effects compose additively
        for e in self.effects:
            if e.layer != LAYER_FG:
                continue
            if not self._effect_active(e, ability):
                continue
            e.update(signals, scaled_dt, ability)
            e.render(target, signals, ability)

        self.hooks.emit("frame_rendered", frame, signals, ability)

    @staticmethod
    def _effect_active(effect: Effect, ability: AbilityState) -> bool:
        if not effect.is_gated():
            return True
        if ability.name != effect.ability_name:
            return False
        return ability.phase in (PHASE_CHARGING, PHASE_ACTIVE, PHASE_RELEASING)


def default_renderer(width: int, height: int, hooks: HookBus) -> EffectsRenderer:
    """The canonical effect roster Aether ships with.

    Order matters for FG layering: things that should sit beneath others
    are added first.
    """
    r = EffectsRenderer(width, height, hooks)
    # Background warp first
    r.add(SpaceStretchEffect(width, height))
    # Foreground effects
    r.add(KamehamehaEffect(width, height))
    r.add(RasenganEffect(width, height))
    r.add(ChidoriEffect(width, height))
    r.add(RealityTearEffect(width, height))
    return r
