"""TimeShatter — glass-break animation that plays when time_freeze releases.

Visual specification:
- LAYER_FG, always-on (ability_name = "").
- Subscribes to the ``ability_exit`` hook; triggers only when exiting
  ``time_freeze``.
- Phase 1 — cracks (first half of animation): TIME_SHATTER_CRACK_LINES jagged
  polylines radiate from the screen centre, coloured TIME_SHATTER_CRACK_COLOR,
  drawn additively so they glow on the frozen frame.
- Phase 2 — shards (second half, blended with cracks): TIME_SHATTER_SHARDS
  triangular shards fall under TIME_SHATTER_GRAVITY_PX gravity and fade out.
- Total duration: TIME_SHATTER_SECONDS.

Performance notes:
- No per-frame full-window SRCALPHA allocation. All drawing routes through
  ``effects.utils`` helpers (shared scratch surface pattern).
- RNG is seeded once at trigger time for deterministic output.
- Shard triangles are rendered as two additive polylines (outline + fill
  approximation via radial glow at centroid) — cheap and visually convincing.
"""

from __future__ import annotations

import logging
import math
import random

import pygame

import config
from core.hooks import HookBus
from core.state import AbilityState, GestureSignals
from effects.base import LAYER_FG, Effect
from effects.utils import additive_polyline, jagged_path

log = logging.getLogger(__name__)

_RGB = tuple[int, int, int]


class _Shard:
    """One triangular glass shard.  Three vertices in screen-space pixels."""

    __slots__ = ("verts", "vel_x", "vel_y", "rot", "rot_rate", "age", "max_age")

    def __init__(
        self,
        verts: list[tuple[float, float]],
        vel_x: float,
        vel_y: float,
        rot: float,
        rot_rate: float,
        max_age: float,
    ) -> None:
        self.verts = verts          # three (x, y) in screen pixels
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.rot = rot              # current rotation in radians
        self.rot_rate = rot_rate    # radians / second
        self.age: float = 0.0
        self.max_age = max_age


class TimeShatter(Effect):
    """Always-on FG effect — glass-shatter reveal when time_freeze exits."""

    layer = LAYER_FG
    name = "time_shatter"
    ability_name = ""  # always-on

    def __init__(self, width: int, height: int, hooks: HookBus) -> None:
        super().__init__(width, height)
        self._animating: bool = False
        self._elapsed: float = 0.0
        self._duration: float = float(config.TIME_SHATTER_SECONDS)
        self._rng: random.Random = random.Random()
        # Crack geometry is fixed for one shatter animation; regenerated on trigger.
        self._crack_pts: list[list[tuple[float, float]]] = []
        self._shards: list[_Shard] = []
        hooks.on("ability_exit", self._on_ability_exit)

    # --- Hook callback -------------------------------------------------------

    def _on_ability_exit(self, name: str) -> None:
        if name != "time_freeze":
            return
        self._start_animation()

    # --- Animation init ------------------------------------------------------

    def _start_animation(self) -> None:
        """Seed RNG, build crack geometry and shard list, start timer."""
        self._rng.seed(0xC0FFEE)
        self._elapsed = 0.0
        self._animating = True
        cx = self.width / 2.0
        cy = self.height / 2.0
        self._crack_pts = self._build_cracks(cx, cy)
        self._shards = self._build_shards(cx, cy)
        log.debug("time_shatter: animation started")

    def _build_cracks(
        self, cx: float, cy: float
    ) -> list[list[tuple[float, float]]]:
        """Generate TIME_SHATTER_CRACK_LINES radial jagged polylines."""
        count = int(config.TIME_SHATTER_CRACK_LINES)
        rng = self._rng
        # Max reach: corner of the screen from centre.
        reach = math.hypot(self.width, self.height) * 0.6
        cracks: list[list[tuple[float, float]]] = []
        for i in range(count):
            base_angle = (math.tau * i / count) + rng.uniform(-0.15, 0.15)
            length = reach * rng.uniform(0.55, 1.0)
            end = (
                cx + math.cos(base_angle) * length,
                cy + math.sin(base_angle) * length,
            )
            pts = jagged_path(
                (cx, cy),
                end,
                segment_len=24.0,
                jitter=18.0,
                rng=rng,
            )
            cracks.append(pts)
        return cracks

    def _build_shards(self, cx: float, cy: float) -> list[_Shard]:
        """Generate TIME_SHATTER_SHARDS triangular shards around the centre."""
        count = int(config.TIME_SHATTER_SHARDS)
        duration = self._duration
        rng = self._rng
        shards: list[_Shard] = []
        for i in range(count):
            # Place shards in a roughly uniform grid across the screen.
            angle = (math.tau * i / count) + rng.uniform(-0.2, 0.2)
            dist = rng.uniform(self.height * 0.05, self.height * 0.45)
            px = cx + math.cos(angle) * dist
            py = cy + math.sin(angle) * dist
            size = rng.uniform(30.0, 90.0)
            verts = _triangle_verts(px, py, size, rng.uniform(0, math.tau))
            vel_x = rng.uniform(-120.0, 120.0)
            # Initial upward kick then gravity pulls down.
            vel_y = rng.uniform(-200.0, 80.0)
            rot_rate = rng.uniform(-4.0, 4.0)
            max_age = duration * rng.uniform(0.5, 1.0)
            shards.append(_Shard(
                verts=verts,
                vel_x=vel_x,
                vel_y=vel_y,
                rot=rng.uniform(0, math.tau),
                rot_rate=rot_rate,
                max_age=max_age,
            ))
        return shards

    # --- Effect.update -------------------------------------------------------

    def update(
        self,
        signals: GestureSignals,
        dt: float,
        ability: AbilityState,
    ) -> None:
        if not self._animating:
            return
        self._elapsed += dt
        if self._elapsed >= self._duration:
            self._animating = False
            self._crack_pts = []
            self._shards = []
            return
        gravity = float(config.TIME_SHATTER_GRAVITY_PX)
        for shard in self._shards:
            if shard.age >= shard.max_age:
                continue
            shard.vel_y += gravity * dt
            dx = shard.vel_x * dt
            dy = shard.vel_y * dt
            shard.verts = [(x + dx, y + dy) for x, y in shard.verts]
            shard.rot += shard.rot_rate * dt
            shard.age += dt

    # --- Effect.render -------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        if not self._animating:
            return
        t = min(1.0, self._elapsed / max(1e-6, self._duration))
        self._render_cracks(target, t)
        self._render_shards(target, t)

    def _render_cracks(self, surf: pygame.Surface, t: float) -> None:
        """Draw crack lines; brightest at t=0, fading out by t=0.6."""
        crack_color: _RGB = config.TIME_SHATTER_CRACK_COLOR  # type: ignore[assignment]
        fade = max(0.0, 1.0 - t / 0.6)
        alpha = int(220 * fade)
        if alpha <= 0:
            return
        # Draw each crack line with its own jagged path.
        for pts in self._crack_pts:
            if len(pts) < 2:
                continue
            additive_polyline(surf, pts, crack_color, 2, alpha)
            # Thin bright core line for the "hot" crack look.
            additive_polyline(surf, pts, (255, 255, 255), 1, int(alpha * 0.5))

    def _render_shards(self, surf: pygame.Surface, t: float) -> None:
        """Draw glass shards; appear from t=0.1, fall and fade to t=1.0."""
        if t < 0.05:
            return
        crack_color: _RGB = config.TIME_SHATTER_CRACK_COLOR  # type: ignore[assignment]
        for shard in self._shards:
            if shard.max_age <= 0:
                continue
            age_frac = min(1.0, shard.age / shard.max_age)
            alpha = int(180 * (1.0 - age_frac))
            if alpha <= 4:
                continue
            verts = shard.verts
            # Outline only: three additive polylines (edges of the triangle).
            # The centroid glow "dots" were removed at the user's request.
            _draw_triangle_outline(surf, verts, crack_color, alpha)


# ---------------------------------------------------------------------------
# Local geometry helpers
# ---------------------------------------------------------------------------

def _triangle_verts(
    cx: float, cy: float, size: float, rotation: float
) -> list[tuple[float, float]]:
    """Equilateral triangle centred at (cx, cy) with circumradius `size`."""
    verts: list[tuple[float, float]] = []
    for k in range(3):
        angle = rotation + math.tau * k / 3.0
        verts.append((cx + math.cos(angle) * size, cy + math.sin(angle) * size))
    return verts


def _draw_triangle_outline(
    surf: pygame.Surface,
    verts: list[tuple[float, float]],
    color: _RGB,
    alpha: int,
) -> None:
    """Draw the three edges of a triangle as additive polylines."""
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        additive_polyline(surf, [a, b], color, 1, alpha)
