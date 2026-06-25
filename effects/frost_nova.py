"""Frost Nova — expanding frost ring and crystalline ice shards.

Visual specification:
- LAYER_FG.
- Charging: frost particles gather between the hands (orbiting cold
  sparkles in the FROST_OUTER_COLOR palette).
- Release / active: expanding frost ring (FROST_CORE_COLOR additive ring)
  grows to FROST_NOVA_RING_MAX_RADIUS_PX, plus FROST_SHARD_COUNT crystalline
  shards radiating outward as jagged ice polylines.
- Cool palette: FROST_CORE_COLOR (near-white ice), FROST_OUTER_COLOR (sky blue).
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

import pygame

import config
from core.state import (
    PHASE_CHARGING,
    AbilityState,
    FrameState,
    GestureSignals,
)
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    jagged_path,
    radial_glow,
)

log = logging.getLogger(__name__)

_RING_EXPAND_SECONDS = 0.55   # time for ring to reach max radius
_SHARD_LIFE = 0.8             # seconds shards persist
_VEIL_SECONDS = 0.9           # full-screen ice-blue wash flash + fade
_VEIL_MAX_ALPHA = 150         # peak opacity of the ice wash (0..255)
_SNOW_COUNT = 64              # falling snowflakes spawned on burst
_SNOW_LIFE = 1.6              # seconds a snowflake lives


@dataclass
class _Shard:
    angle: float
    length: float
    pts: list[tuple[int, int]] = field(default_factory=list)
    life: float = _SHARD_LIFE
    max_life: float = _SHARD_LIFE


@dataclass
class _Snowflake:
    x: float
    y: float
    vx: float       # horizontal drift px/s
    vy: float       # fall speed px/s
    size: float
    drift_phase: float
    life: float = _SNOW_LIFE


class FrostNovaEffect(Effect):
    layer = LAYER_FG
    name = "frost_nova"
    ability_name = "frost_nova"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0xF0057)
        self._shards: list[_Shard] = []
        self._snow: list[_Snowflake] = []
        self._burst_age: float = -1.0   # -1 = no burst active
        self._burst_cx: int = 0
        self._burst_cy: int = 0

        # Full-screen ice-blue wash, pre-filled once; alpha set per frame.
        self._veil = pygame.Surface((width, height))
        self._veil.fill(config.FROST_OUTER_COLOR)

    # --- Lifecycle ----------------------------------------------------------

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        self._shards.clear()
        self._snow.clear()
        self._burst_age = -1.0

    def on_release(self, intensity: float, frame: FrameState) -> None:
        cx, cy = self._midpoint_from_frame(frame)
        self._burst_cx = cx
        self._burst_cy = cy
        self._burst_age = 0.0
        self._shards.clear()
        self._spawn_snow()
        # Spawn crack shards radiating outward. Each shard is grown ALL THE WAY to
        # the screen edge along its angle (slightly past, ×0.9–1.1) so the WHOLE
        # frame cracks — not just a disc around the burst centre.
        for i in range(config.FROST_SHARD_COUNT):
            angle = (i / config.FROST_SHARD_COUNT) * math.tau
            edge = self._edge_distance(cx, cy, angle)
            length = edge * self._rng.uniform(0.9, 1.1)
            # Build jagged polyline points for the shard
            end_x = cx + math.cos(angle) * length
            end_y = cy + math.sin(angle) * length
            raw_pts = jagged_path(
                (cx, cy), (int(end_x), int(end_y)),
                segment_len=26.0,
                jitter=10.0,
                rng=self._rng,
            )
            # Only keep the outer portion so shards appear to grow from ring
            shard = _Shard(
                angle=angle,
                length=length,
                pts=raw_pts,
                life=_SHARD_LIFE,
                max_life=_SHARD_LIFE,
            )
            self._shards.append(shard)

    def on_exit(self) -> None:
        self._shards.clear()
        self._snow.clear()
        self._burst_age = -1.0

    # --- Update -------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        if self._burst_age >= 0.0:
            self._burst_age += dt
        for sh in self._shards:
            sh.life -= dt
        self._shards = [sh for sh in self._shards if sh.life > 0]

        # Drift snowflakes down (with a gentle horizontal sway).
        for sf in self._snow:
            sf.life -= dt
            sf.drift_phase += dt * 2.2
            sf.x += (sf.vx + math.sin(sf.drift_phase) * 18.0) * dt
            sf.y += sf.vy * dt
        self._snow = [
            sf for sf in self._snow
            if sf.life > 0 and sf.y < self.height + 24
        ]

    def _spawn_snow(self) -> None:
        """Seed snowflakes across the frame, already mid-fall, on the burst."""
        self._snow.clear()
        rng = self._rng
        for _ in range(_SNOW_COUNT):
            self._snow.append(_Snowflake(
                x=rng.uniform(0, self.width),
                y=rng.uniform(-self.height * 0.25, self.height * 0.6),
                vx=rng.uniform(-30.0, 30.0),
                vy=rng.uniform(70.0, 180.0),
                size=rng.uniform(2.0, 5.0),
                drift_phase=rng.uniform(0.0, math.tau),
                life=_SNOW_LIFE * rng.uniform(0.7, 1.0),
            ))

    # --- Render -------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        # 1. Charging: frost sparkle gathering between hands
        if ability.phase == PHASE_CHARGING:
            self._render_charge_sparkles(target, ability)

        # 2. Burst: expanding ring + shards
        if self._burst_age >= 0.0:
            self._render_burst(target)

    # --- Private helpers ----------------------------------------------------

    def _render_charge_sparkles(
        self, target: pygame.Surface, ability: AbilityState
    ) -> None:
        ph = ability.primary_hand
        sh = ability.secondary_hand
        if ph is None:
            return
        if sh is not None:
            cx = int((ph.palm[0] + sh.palm[0]) * 0.5 * self.width)
            cy = int((ph.palm[1] + sh.palm[1]) * 0.5 * self.height)
        else:
            cx = int(ph.palm[0] * self.width)
            cy = int(ph.palm[1] * self.height)

        count = int(12 + 30 * ability.charge)
        rng = self._rng
        for _ in range(count):
            a = rng.uniform(0, math.tau)
            r = rng.uniform(8, 50 + 40 * ability.charge)
            px = int(cx + math.cos(a) * r)
            py = int(cy + math.sin(a) * r)
            additive_circle(
                target, (px, py),
                size=rng.randint(1, 3),
                color=config.FROST_CORE_COLOR,
                alpha=int(180 * ability.charge),
            )
        radial_glow(
            target, (cx, cy),
            radius=int(20 + 60 * ability.charge),
            color=config.FROST_OUTER_COLOR,
            alpha=int(140 * ability.charge),
            layers=6,
        )

    def _render_burst(self, target: pygame.Surface) -> None:
        cx, cy = self._burst_cx, self._burst_cy

        # Full-screen ice-blue wash: the whole frame flashes cold, then fades.
        veil_t = self._burst_age / _VEIL_SECONDS
        if veil_t < 1.0:
            veil_alpha = int(_VEIL_MAX_ALPHA * (1.0 - veil_t))
            if veil_alpha > 0:
                self._veil.set_alpha(veil_alpha)
                target.blit(self._veil, (0, 0))

        # Ring
        t_ring = min(1.0, self._burst_age / _RING_EXPAND_SECONDS)
        ring_radius = int(config.FROST_NOVA_RING_MAX_RADIUS_PX * t_ring)
        ring_fade = 1.0 - t_ring * t_ring
        if ring_fade > 0.02 and ring_radius > 0:
            additive_ring(
                target, (cx, cy),
                radius=ring_radius,
                color=config.FROST_CORE_COLOR,
                alpha=int(220 * ring_fade),
                width=max(2, int(8 * ring_fade)),
            )
            additive_ring(
                target, (cx, cy),
                radius=max(1, ring_radius - 6),
                color=config.FROST_OUTER_COLOR,
                alpha=int(120 * ring_fade),
                width=max(1, int(3 * ring_fade)),
            )

        # Shards (grow outward from origin)
        for sh in self._shards:
            life_ratio = sh.life / max(1e-6, sh.max_life)
            # Reveal shard progressively based on how early in burst we are
            reveal = min(1.0, (1.0 - life_ratio) * 3.0 + 0.3)
            visible_count = max(2, int(len(sh.pts) * reveal))
            pts_visible = sh.pts[:visible_count]
            if len(pts_visible) < 2:
                continue
            # Outer blue glow
            additive_polyline(
                target, pts_visible,
                color=config.FROST_OUTER_COLOR,
                width=4,
                alpha=int(160 * life_ratio),
            )
            # White ice core
            additive_polyline(
                target, pts_visible,
                color=config.FROST_CORE_COLOR,
                width=1,
                alpha=int(230 * life_ratio),
            )

        # Falling snowflakes drifting down the whole frame.
        for sf in self._snow:
            life_ratio = max(0.0, min(1.0, sf.life / _SNOW_LIFE))
            additive_circle(
                target, (int(sf.x), int(sf.y)),
                size=max(1, int(sf.size)),
                color=config.FROST_CORE_COLOR,
                alpha=int(220 * life_ratio),
            )

    def _edge_distance(self, cx: int, cy: int, angle: float) -> float:
        """Distance from (cx, cy) to the frame edge along *angle*.

        Lets each crack shard reach the screen border regardless of where the
        burst is centred, so the whole frame ends up cracked.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        candidates: list[float] = []
        if dx > 1e-6:
            candidates.append((self.width - cx) / dx)
        elif dx < -1e-6:
            candidates.append((0 - cx) / dx)
        if dy > 1e-6:
            candidates.append((self.height - cy) / dy)
        elif dy < -1e-6:
            candidates.append((0 - cy) / dy)
        valid = [c for c in candidates if c > 0]
        return min(valid) if valid else float(max(self.width, self.height))

    def _midpoint_from_frame(self, frame: FrameState) -> tuple[int, int]:
        # Extract hand positions at the time of release via the typed attribute.
        # FrameState.ability is optional (may be None or absent on older paths).
        try:
            ab: AbilityState = frame.ability  # type: ignore[union-attr]
        except AttributeError:
            log.warning("frost_nova: FrameState has no 'ability' attribute — using screen centre")
            return self.width // 2, self.height // 2
        if ab is None:
            return self.width // 2, self.height // 2
        ph = ab.primary_hand
        sh = ab.secondary_hand
        if ph is not None and sh is not None:
            mx = int((ph.palm[0] + sh.palm[0]) * 0.5 * self.width)
            my = int((ph.palm[1] + sh.palm[1]) * 0.5 * self.height)
            return mx, my
        if ph is not None:
            return int(ph.palm[0] * self.width), int(ph.palm[1] * self.height)
        return self.width // 2, self.height // 2
