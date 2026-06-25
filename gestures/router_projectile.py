"""Projectile-spawning helpers mixin for AbilityRouter.

Contains the projectile spawn, launch-origin, and aim-direction helpers.
Pulled out of router.py to keep that file under the 800-line ceiling.
All methods reference ``self.*`` and are mixed into ``AbilityRouter``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

import config
from core.state import FrameState, HandData, ProjectileSpawn

if TYPE_CHECKING:
    from gestures.router import AbilityDef

log = logging.getLogger(__name__)


class _ProjectileMixin:
    """Mixin providing projectile-spawn logic for AbilityRouter."""

    def _spawn_projectile_if_needed(
        self, ability: AbilityDef, frame: FrameState
    ) -> None:
        """Emit a projectile_spawn hook event if this ability has a projectile kind."""
        if ability.projectile_kind is None:
            return

        hand = self.state.primary_hand
        h, w = frame.frame_bgr.shape[:2]

        if hand is not None:
            origin_px = self._launch_origin_px(ability, hand, w, h)
            direction = self._aim_direction(
                hand, index_tip=(ability.projectile_kind == "fireball")
            )
        else:
            origin_px = (float(w) * 0.5, float(h) * 0.5)
            direction = np.array([0.0, -1.0], dtype=np.float32)

        if ability.projectile_kind == "rasengan":
            speed = config.PROJECTILE_RASENGAN_SPEED_PX
            radius = config.PROJECTILE_RASENGAN_RADIUS_PX
        elif ability.projectile_kind == "fireball":
            speed = config.PROJECTILE_FIREBALL_SPEED_PX
            radius = config.PROJECTILE_FIREBALL_RADIUS_PX
        else:
            log.warning(
                "router: unknown projectile_kind %r — using fallback speed/radius",
                ability.projectile_kind,
            )
            speed = 800.0
            radius = 40.0

        spawn = ProjectileSpawn(
            kind=ability.projectile_kind,
            origin_px=origin_px,
            direction=direction,
            speed_px=speed,
            intensity=self.state.intensity,
            radius_px=radius,
        )
        self.hooks.emit("projectile_spawn", spawn)
        log.info(
            "router: spawn projectile %s at (%.0f,%.0f) dir=(%.2f,%.2f)",
            ability.projectile_kind, origin_px[0], origin_px[1],
            direction[0], direction[1],
        )

    @staticmethod
    def _launch_origin_px(
        ability: AbilityDef, hand: HandData, w: int, h: int
    ) -> tuple[float, float]:
        """Where the projectile is born. Fireball is the index finger pointing
        up, so it launches from just ABOVE the INDEX FINGERTIP (suspended in the
        air a little); everything else from the palm."""
        if ability.projectile_kind == "fireball" and hand.landmarks is not None:
            tip = hand.landmarks[8, :2]  # INDEX_TIP
            return (
                float(tip[0]) * w,
                float(tip[1]) * h - config.FIREBALL_FINGERTIP_LIFT_PX,
            )
        return (float(hand.palm_px[0]), float(hand.palm_px[1]))

    @staticmethod
    def _aim_direction(hand: HandData, *, index_tip: bool = False) -> np.ndarray:
        """Direction a thrown projectile should travel. Prefer the captured flick
        (the hand has usually already stopped by fire time, so the instantaneous
        velocity is unreliable), then live velocity, then straight up. When
        `index_tip` (the fireball), the live fingertip velocity wins if it's the
        stronger signal — the fireball is thrown by flicking the finger, so the
        shot should follow the finger, not the near-still palm."""
        flick = np.asarray(hand.flick, dtype=np.float32)
        if hand.flick_speed > 0.0 and float(np.linalg.norm(flick)) > 1e-6:
            v = flick
        else:
            v = np.asarray(hand.velocity, dtype=np.float32)
        if index_tip:
            tip_v = np.asarray(hand.index_tip_velocity, dtype=np.float32)
            if float(np.linalg.norm(tip_v)) > float(np.linalg.norm(v)):
                v = tip_v
        mag = float(np.linalg.norm(v))
        if mag < 1e-6:
            return np.array([0.0, -1.0], dtype=np.float32)
        return np.array([float(v[0]) / mag, float(v[1]) / mag], dtype=np.float32)
