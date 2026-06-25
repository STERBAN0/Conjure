"""ProjectileField — always-on foreground effect that tracks flying projectiles.

Projectiles are spawned by the ability router via the ``projectile_spawn`` hook
event and travel in a straight line until they pass ``PROJECTILE_EDGE_MARGIN_PX``
beyond any screen edge, at which point they burst into outward-streak particles.

Supported projectile kinds:
    rasengan  - blue spinning sphere (radial glow + vortex arms + shell rings)
    fireball  - orange turbulent sphere (radial glow + flickering embers + tail)

Architecture:
    ``ProjectileField`` is an always-on effect (``ability_name = ""``).
    The ``_Projectile`` inner class uses ``__slots__`` for cache-friendliness.
    ``_BurstParticle`` represents the post-impact streak particles.
    No full-window surfaces are allocated. All drawing goes through
    ``effects.utils`` primitives.
"""

from __future__ import annotations

import logging
import math
import random

import numpy as np
import pygame

import config
from core.hooks import HookBus
from core.state import AbilityState, GestureSignals, ProjectileSpawn
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    lerp_color,
    radial_glow,
)

log = logging.getLogger(__name__)

# Type alias for RGB tuples.
_RGB = tuple[int, int, int]


class _Projectile:
    """One live projectile.  __slots__ keeps per-instance overhead minimal."""

    __slots__ = (
        "pos",      # np.ndarray float32 (2,) — pixel position
        "vel",      # np.ndarray float32 (2,) — pixels/second
        "kind",     # str
        "intensity",
        "radius",   # float, visual radius px
        "age",      # float, seconds alive
        "spin",     # float, accumulated rotation radians (for rasengan arms)
        "spin_rate",# float, radians/second
        "_rng",     # random.Random — per-projectile RNG for flicker
    )

    def __init__(
        self,
        spawn: ProjectileSpawn,
        rng: random.Random,
    ) -> None:
        self.pos = np.array(spawn.origin_px, dtype=np.float32)
        self.vel = np.array(spawn.direction, dtype=np.float32) * spawn.speed_px
        self.kind = spawn.kind
        self.intensity = float(spawn.intensity)
        self.radius = float(spawn.radius_px)
        self.age = 0.0
        self.spin = rng.uniform(0.0, math.tau)
        self.spin_rate = rng.uniform(8.0, 14.0) * rng.choice((-1.0, 1.0))  # type: ignore[arg-type]
        self._rng = rng


class _MuzzleFlash:
    """Brief expanding burst drawn at the projectile spawn origin."""

    __slots__ = ("pos", "age", "max_age", "radius", "color_outer", "color_core")

    def __init__(
        self,
        pos: np.ndarray,
        max_age: float,
        radius: float,
        color_outer: _RGB,
        color_core: _RGB,
    ) -> None:
        self.pos = pos.copy()
        self.age = 0.0
        self.max_age = float(max_age)
        self.radius = float(radius)
        self.color_outer = color_outer
        self.color_core = color_core


class _BurstParticle:
    """Short-lived streak particle spawned when a projectile hits the edge."""

    __slots__ = ("pos", "vel", "age", "max_age", "color")

    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        max_age: float,
        color: _RGB,
    ) -> None:
        self.pos = pos.astype(np.float32)
        self.vel = vel.astype(np.float32)
        self.age = 0.0
        self.max_age = float(max_age)
        self.color = color


class ProjectileField(Effect):
    """Always-on foreground effect managing all in-flight projectiles."""

    layer = LAYER_FG
    name = "projectiles"
    ability_name = ""  # always-on

    def __init__(self, width: int, height: int, hooks: HookBus) -> None:
        super().__init__(width, height)
        self._projectiles: list[_Projectile] = []
        self._bursts: list[_BurstParticle] = []
        self._flashes: list[_MuzzleFlash] = []
        self._rng = random.Random(0xB00B)
        hooks.on("projectile_spawn", self._on_spawn)

    # ------------------------------------------------------------------
    # Hook callback
    # ------------------------------------------------------------------

    def _on_spawn(self, spawn: ProjectileSpawn) -> None:
        if len(self._projectiles) >= config.PROJECTILE_MAX_ACTIVE:
            # Drop the oldest to stay within the cap.
            self._projectiles.pop(0)
            log.debug("projectile_field: cap reached, dropping oldest")

        proj_rng = random.Random(self._rng.randint(0, 0xFFFFFF))
        p = _Projectile(spawn, proj_rng)
        self._projectiles.append(p)

        # Muzzle flash — brief expanding burst at the spawn origin.
        if spawn.kind == "rasengan":
            c_outer: _RGB = config.RASENGAN_OUTER_COLOR  # type: ignore[assignment]
            c_core: _RGB = config.RASENGAN_CORE_COLOR    # type: ignore[assignment]
        else:
            c_outer = config.FIREBALL_OUTER_COLOR  # type: ignore[assignment]
            c_core = config.FIREBALL_CORE_COLOR    # type: ignore[assignment]
        flash = _MuzzleFlash(
            pos=np.array(spawn.origin_px, dtype=np.float32),
            max_age=config.PROJECTILE_MUZZLE_SECONDS,
            radius=config.PROJECTILE_MUZZLE_RADIUS_PX,
            color_outer=c_outer,
            color_core=c_core,
        )
        self._flashes.append(flash)

        log.info(
            "projectile_field: spawn %s at (%.0f,%.0f)",
            spawn.kind, spawn.origin_px[0], spawn.origin_px[1],
        )

    # ------------------------------------------------------------------
    # Effect.update  — called every frame by the renderer
    # ------------------------------------------------------------------

    def update(
        self,
        signals: GestureSignals,
        dt: float,
        ability: AbilityState,
    ) -> None:
        margin = config.PROJECTILE_EDGE_MARGIN_PX
        w, h = float(self.width), float(self.height)
        alive: list[_Projectile] = []

        for p in self._projectiles:
            p.pos += p.vel * dt
            p.age += dt
            p.spin += p.spin_rate * dt

            # Burst when centre passes beyond the screen edge + margin.
            x, y = float(p.pos[0]), float(p.pos[1])
            past_edge = (
                x < -margin
                or x > w + margin
                or y < -margin
                or y > h + margin
            )
            if past_edge:
                self._spawn_burst(p)
            else:
                alive.append(p)

        self._projectiles = alive

        # Advance muzzle flashes.
        alive_flashes: list[_MuzzleFlash] = []
        for f in self._flashes:
            f.age += dt
            if f.age < f.max_age:
                alive_flashes.append(f)
        self._flashes = alive_flashes

        # Advance burst particles.
        alive_bursts: list[_BurstParticle] = []
        for bp in self._bursts:
            bp.pos += bp.vel * dt
            bp.age += dt
            if bp.age < bp.max_age:
                alive_bursts.append(bp)
        self._bursts = alive_bursts

    # ------------------------------------------------------------------
    # Effect.render — called every frame by the renderer
    # ------------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        # Muzzle flashes render first (behind the projectile body).
        for f in self._flashes:
            t = f.age / f.max_age  # 0 → 1 as flash ages
            alpha_outer = int(180 * (1.0 - t))
            alpha_core = int(220 * (1.0 - t) ** 0.5)
            r_outer = int(f.radius * (1.0 + t * 0.8))
            r_core = int(f.radius * 0.45 * (1.0 + t * 0.3))
            cx, cy = int(f.pos[0]), int(f.pos[1])
            if alpha_outer > 0:
                radial_glow(target, (cx, cy), r_outer, f.color_outer, alpha_outer, layers=8)
            if alpha_core > 0:
                additive_circle(target, (cx, cy), r_core, f.color_core, alpha_core)

        for p in self._projectiles:
            if p.kind == "rasengan":
                self._render_rasengan(target, p)
            elif p.kind == "fireball":
                self._render_fireball(target, p)
            else:
                # Fallback: simple glowing circle
                cx, cy = int(p.pos[0]), int(p.pos[1])
                radial_glow(target, (cx, cy), int(p.radius * 2), (200, 200, 200), 180)
                additive_circle(target, (cx, cy), int(p.radius), (255, 255, 255), 200)

        for bp in self._bursts:
            self._render_burst_particle(target, bp)

    # ------------------------------------------------------------------
    # Per-kind renderers
    # ------------------------------------------------------------------

    def _render_rasengan(self, surf: pygame.Surface, p: _Projectile) -> None:
        """Blue spinning chakra sphere.

        Layers:
          1. Outer radial glow (wide, dim)
          2. Mid radial glow (medium, brighter)
          3. Core (small, bright white-blue)
          4. Two additive orbital shell rings
          5. Three vortex arm polylines rotating with p.spin
        """
        cx, cy = int(p.pos[0]), int(p.pos[1])
        r = p.radius
        outer_color: _RGB = config.RASENGAN_OUTER_COLOR  # type: ignore[assignment]
        core_color: _RGB = config.RASENGAN_CORE_COLOR    # type: ignore[assignment]

        # 1 + 2. Radial glow — outer and mid
        radial_glow(surf, (cx, cy), int(r * 2.8), outer_color, 140, layers=10)
        radial_glow(surf, (cx, cy), int(r * 1.6), outer_color, 200, layers=8)
        # 3. Core
        additive_circle(surf, (cx, cy), int(r * 0.55), core_color, 230)
        # 4. Shell rings
        additive_ring(surf, (cx, cy), int(r), outer_color, 180, width=3)
        additive_ring(surf, (cx, cy), int(r * 0.7), core_color, 140, width=2)

        # 5. Vortex arms — three polylines radiating + curving.
        arm_count = 3
        arm_points = 10
        arm_len = r * 0.9
        for a in range(arm_count):
            base_angle = p.spin + (math.tau * a / arm_count)
            pts: list[tuple[float, float]] = []
            for k in range(arm_points):
                t = k / (arm_points - 1)
                arm_r = arm_len * t
                theta = base_angle + t * 1.8  # spiral outward
                pts.append((
                    cx + math.cos(theta) * arm_r,
                    cy + math.sin(theta) * arm_r,
                ))
            alpha = int(180 * (1.0 - p.age * 0.5))
            alpha = max(0, min(255, alpha))
            additive_polyline(surf, pts, outer_color, 2, alpha)

    def _render_fireball(self, surf: pygame.Surface, p: _Projectile) -> None:
        """Turbulent orange-red fireball.

        Layers:
          1. Outer radial glow (orange, wide)
          2. Mid radial glow (yellow, tighter)
          3. Core (white-yellow, bright)
          4. Flickering embers — 6 small additive circles scattered in radius
          5. Short trailing tail polyline behind the projectile
        """
        cx, cy = int(p.pos[0]), int(p.pos[1])
        r = p.radius
        outer_color: _RGB = config.FIREBALL_OUTER_COLOR  # type: ignore[assignment]
        core_color: _RGB = config.FIREBALL_CORE_COLOR    # type: ignore[assignment]

        # 1 + 2. Glow layers
        radial_glow(surf, (cx, cy), int(r * 3.0), outer_color, 120, layers=10)
        radial_glow(surf, (cx, cy), int(r * 1.8), core_color, 200, layers=8)
        # 3. Core
        additive_circle(surf, (cx, cy), int(r * 0.5), (255, 255, 200), 240)

        # 4. Flickering embers
        rng = p._rng
        ember_count = 8
        for _ in range(ember_count):
            angle = rng.uniform(0.0, math.tau)
            dist = rng.uniform(0.0, r * 0.9)
            ex = cx + math.cos(angle) * dist
            ey = cy + math.sin(angle) * dist
            ember_r = int(rng.uniform(2, 6))
            ember_alpha = int(rng.uniform(100, 200))
            ember_col = lerp_color(outer_color, core_color, rng.random())
            additive_circle(surf, (int(ex), int(ey)), ember_r, ember_col, ember_alpha)

        # 5. Trailing tail (short polyline behind motion direction)
        speed = float(np.linalg.norm(p.vel)) + 1e-6
        rev_dir = -p.vel / speed  # pointing backwards
        tail_len = r * 2.5
        tail_pts: list[tuple[float, float]] = []
        tail_segs = 8
        for k in range(tail_segs):
            t = k / (tail_segs - 1)
            tail_pts.append((
                float(cx + rev_dir[0] * tail_len * t),
                float(cy + rev_dir[1] * tail_len * t),
            ))
        tail_alpha = int(160 * max(0.0, 1.0 - p.age * 2.0))
        if tail_alpha > 0:
            additive_polyline(surf, tail_pts, outer_color, 4, tail_alpha)

    def _render_burst_particle(
        self, surf: pygame.Surface, bp: _BurstParticle
    ) -> None:
        """Outward streak particle — a short line from previous to current pos."""
        t = bp.age / bp.max_age  # 0 → 1
        alpha = int(220 * (1.0 - t) ** 2)
        if alpha <= 0:
            return
        px, py = int(bp.pos[0]), int(bp.pos[1])
        # Trail back half a frame worth of motion.
        dt_back = 0.02
        bx = int(bp.pos[0] - bp.vel[0] * dt_back)
        by = int(bp.pos[1] - bp.vel[1] * dt_back)
        additive_polyline(surf, [(bx, by), (px, py)], bp.color, 2, alpha)

    # ------------------------------------------------------------------
    # Burst spawner
    # ------------------------------------------------------------------

    def _spawn_burst(self, p: _Projectile) -> None:
        """Spawn PROJECTILE_BURST_PARTICLES radial streak particles."""
        count = config.PROJECTILE_BURST_PARTICLES
        rng = p._rng
        max_age = 0.4

        if p.kind == "rasengan":
            colors: list[_RGB] = [
                config.RASENGAN_OUTER_COLOR,  # type: ignore[list-item]
                config.RASENGAN_CORE_COLOR,   # type: ignore[list-item]
            ]
        else:
            colors = [
                config.FIREBALL_OUTER_COLOR,  # type: ignore[list-item]
                config.FIREBALL_CORE_COLOR,   # type: ignore[list-item]
            ]

        speed_base = float(np.linalg.norm(p.vel)) * 0.6
        for _ in range(count):
            angle = rng.uniform(0.0, math.tau)
            speed = rng.uniform(speed_base * 0.4, speed_base * 1.4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            col = colors[rng.randint(0, len(colors) - 1)]
            self._bursts.append(_BurstParticle(
                pos=p.pos.copy(),
                vel=vel,
                max_age=rng.uniform(max_age * 0.5, max_age),
                color=col,
            ))

        log.debug("projectile_field: burst %s at (%.0f,%.0f)", p.kind, p.pos[0], p.pos[1])

