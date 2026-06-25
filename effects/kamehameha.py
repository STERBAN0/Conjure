"""Kamehameha — charged beam between cupped hands.

Two visual phases:

  CHARGING:
    Glowing blue-white sphere between the cupped hands, growing and pulsing
    with charge. Orbiting particles trail around it. Periodic shockwave
    rings expand outward.

  ACTIVE (post-release):
    The blast travels where the cupped palms point (see _track_aim). Face the
    cup at the camera and it fires at the viewer: the screen floods with the
    beam's blue light (an expanding bloom from the sphere plus a blue veil that
    builds to a near-total engulf, then recedes on release). Tilt the cup to a
    side and the cylindrical beam shoots toward that screen edge instead, with
    little to no screen engulf. Sustains for the ability's `active_duration`.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
    AbilityState,
    FrameState,
    GestureSignals,
)
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    draw_screen_flash,
    ease_in_out_quad,
    jagged_path,
    radial_glow,
)

# Upper bound on active-phase particle count. At 12 spawns/frame × 60 fps the
# list would grow to ~3600 particles over 5 s without a cap.
_MAX_PARTICLES = 1200


class _Particle:
    __slots__ = ("pos", "vel", "life", "max_life", "size")

    def __init__(self, pos, vel, life, size):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.vel = np.asarray(vel, dtype=np.float32)
        self.life = life
        self.max_life = life
        self.size = size


class KamehamehaEffect(Effect):
    layer = LAYER_FG
    name = "kamehameha"
    ability_name = "kamehameha"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._rng = random.Random(0xCA7E)
        self._particles: list[_Particle] = []
        self._shockwaves: list[float] = []   # ages of expanding rings
        self._next_wave_t = 0.0
        self._beam_pulse = 0.0
        self._flash = 0.0
        # Aim: a smoothed screen-plane direction the blast travels, plus how
        # squarely the palms face the camera (engulf strength). Updated while
        # CHARGING and FROZEN at the moment of firing, so the beam keeps the
        # direction you aimed it. Default = straight at the viewer.
        self._aim_axis = np.array([0.0, -1.0], dtype=np.float32)
        self._aim_engulf = 1.0

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        self._particles.clear()
        self._shockwaves.clear()
        self._next_wave_t = 0.4
        self._flash = 0.0
        self._aim_axis = np.array([0.0, -1.0], dtype=np.float32)
        self._aim_engulf = 1.0

    def on_release(self, intensity: float, frame: FrameState) -> None:
        self._flash = max(self._flash, intensity)

    def on_exit(self) -> None:
        self._particles.clear()
        self._shockwaves.clear()

    # ------------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        self._flash = max(0.0, self._flash - dt * 3.5)
        self._beam_pulse += dt
        self._next_wave_t -= dt

        # Step particles + shockwaves regardless of phase.
        for p in self._particles:
            p.life -= dt
            p.pos += p.vel * dt
        self._particles = [p for p in self._particles if p.life > 0]
        self._shockwaves = [age + dt for age in self._shockwaves if age + dt < 1.0]

        if ability.phase == PHASE_CHARGING:
            # Track where the cupped palms point so the blast fires that way; the
            # aim is frozen once we leave CHARGING (i.e. at the instant of firing).
            self._track_aim(ability)
            self._update_charge(signals, dt, ability)
        elif ability.phase == PHASE_ACTIVE:
            self._update_active(signals, dt, ability)

    def _update_charge(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        cx, cy = self._sphere_center(ability)
        radius = self._sphere_radius(ability)

        # Spawn orbiting particles up to a target count
        target = int(config.KAMEHAMEHA_PARTICLE_COUNT * ability.charge)
        while len(self._particles) < target:
            a = self._rng.uniform(0, math.tau)
            r = self._rng.uniform(radius * 0.4, radius * 1.1)
            speed = self._rng.uniform(40, 110)
            tangent = (-math.sin(a) * speed, math.cos(a) * speed)
            self._particles.append(_Particle(
                pos=(cx + math.cos(a) * r, cy + math.sin(a) * r),
                vel=tangent,
                life=self._rng.uniform(0.3, 0.7),
                size=self._rng.uniform(2, 4),
            ))

        # Periodic shockwave rings
        if self._next_wave_t <= 0.0 and ability.charge > 0.4:
            self._shockwaves.append(0.0)
            self._next_wave_t = 0.45

    def _update_active(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        # Spawn beam-aligned particles streaming outward.
        cx, cy = self._sphere_center(ability)
        ax, ay = self._beam_axis(ability)
        if len(self._particles) < _MAX_PARTICLES:
            for _ in range(12):
                spread = self._rng.uniform(-25, 25)
                px = cx + spread * (-ay)
                py = cy + spread * ax
                speed = self._rng.uniform(700, 1400)
                self._particles.append(_Particle(
                    pos=(px, py),
                    vel=(ax * speed, ay * speed),
                    life=self._rng.uniform(0.2, 0.45),
                    size=self._rng.uniform(2, 5),
                ))
        if self._next_wave_t <= 0.0:
            self._shockwaves.append(0.0)
            self._next_wave_t = 0.18

    # ------------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        # Particles always render
        for p in self._particles:
            life_ratio = p.life / max(1e-6, p.max_life)
            alpha = int(255 * life_ratio)
            size = max(1, int(p.size * (0.5 + 0.5 * life_ratio)))
            x, y = int(p.pos[0]), int(p.pos[1])
            additive_circle(
                target, (x, y), size, config.KAMEHAMEHA_OUTER_COLOR, alpha
            )

        if ability.phase == PHASE_CHARGING:
            self._render_sphere(target, ability)
            self._render_shockwaves(target, ability)
        elif ability.phase == PHASE_ACTIVE:
            self._render_sphere(target, ability, scale=1.2)
            self._render_beam(target, ability, signals)
            self._render_shockwaves(target, ability)
            # Flood the screen with blue light only when the blast is aimed at the
            # viewer; a side-aimed shot just sends the beam toward that edge.
            self._render_screen_engulf(target, ability)
        elif ability.phase == PHASE_RELEASING:
            # Fading beam + receding engulf
            self._render_beam(target, ability, signals, fade=True)
            self._render_screen_engulf(target, ability)

        if self._flash > 0:
            draw_screen_flash(
                target, color=(255, 255, 255), alpha=int(140 * self._flash)
            )

    # ------------------------------------------------------------------

    def _render_sphere(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        scale: float = 1.0,
    ) -> None:
        cx, cy = self._sphere_center(ability)
        radius = int(self._sphere_radius(ability) * scale)
        pulse = 1.0 + 0.08 * math.sin(self._beam_pulse * 8.0)
        # Outer cyan glow
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 1.6 * pulse),
            color=config.KAMEHAMEHA_OUTER_COLOR,
            alpha=int(180 * ability.charge),
        )
        # Hot white core
        radial_glow(
            target, (cx, cy),
            radius=int(radius * 0.8 * pulse),
            color=config.KAMEHAMEHA_CORE_COLOR,
            alpha=255,
        )
        if ability.charge > 0.2:
            for i, alpha_scale in enumerate((145, 100, 70)):
                rr = int(radius * (1.0 + i * 0.22) * pulse)
                additive_ring(
                    target,
                    (cx, cy),
                    radius=rr,
                    color=(180, 240, 255),
                    alpha=int(alpha_scale * ability.charge),
                    width=max(1, int(2 + 2 * ability.charge)),
                )

    def _render_screen_engulf(
        self, target: pygame.Surface, ability: AbilityState
    ) -> None:
        """Flood the screen with blue light, as if the blast is coming at you.

        Ramps up over the first third of the active phase, then recedes during
        release. An expanding blue bloom punches out from the sphere toward the
        whole frame, and a blue veil engulfs everything (additive flash).
        """
        if ability.phase == PHASE_ACTIVE:
            dur = max(1e-3, config.ABILITY_ACTIVE_DURATION.get("kamehameha", 1.5))
            ramp = min(1.0, (ability.phase_age / dur) / 0.35)
        elif ability.phase == PHASE_RELEASING:
            ramp = max(0.0, 1.0 - ability.phase_age / 0.3)
        else:
            return
        # Only engulf when the blast is aimed at the screen; a side-aimed shot
        # barely floods (the directional beam is the whole show then).
        ramp *= self._aim_engulf
        if ramp <= 0.0:
            return

        intensity = max(0.6, float(ability.intensity))
        pulse = 0.9 + 0.1 * math.sin(self._beam_pulse * 10.0)
        cx, cy = self._sphere_center(ability)
        diag = (self.width ** 2 + self.height ** 2) ** 0.5

        # Expanding blue bloom from the sphere outward toward full screen.
        bloom_r = int(self._sphere_radius(ability) + ramp * diag * 0.85 * pulse)
        radial_glow(
            target, (cx, cy), bloom_r,
            color=config.KAMEHAMEHA_ENGULF_COLOR,
            alpha=int(150 * ramp * intensity),
            layers=14,
        )
        # Bright forward core punching at the viewer.
        radial_glow(
            target, (cx, cy),
            int(self._sphere_radius(ability) * (1.5 + 3.0 * ramp)),
            color=config.KAMEHAMEHA_CORE_COLOR,
            alpha=int(210 * ramp),
            layers=10,
        )
        # Blue veil engulfing the whole frame.
        veil = int(config.KAMEHAMEHA_ENGULF_MAX_ALPHA * ramp * intensity)
        if veil > 0:
            draw_screen_flash(target, config.KAMEHAMEHA_ENGULF_COLOR, veil)

    def _render_shockwaves(
        self, target: pygame.Surface, ability: AbilityState
    ) -> None:
        cx, cy = self._sphere_center(ability)
        base_r = int(self._sphere_radius(ability))
        for age in self._shockwaves:
            t = age  # 0..1
            r = int(base_r + t * 220)
            alpha = int(140 * (1 - t) * ability.charge)
            if alpha <= 0 or r <= 0:
                continue
            ring = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(
                ring, (*config.KAMEHAMEHA_OUTER_COLOR, alpha),
                (r + 2, r + 2), r, max(1, int(4 * (1 - t))),
            )
            target.blit(
                ring, (cx - r - 2, cy - r - 2),
                special_flags=pygame.BLEND_RGBA_ADD,
            )

    def _render_beam(
        self,
        target: pygame.Surface,
        ability: AbilityState,
        signals: GestureSignals,
        fade: bool = False,
    ) -> None:
        cx, cy = self._sphere_center(ability)
        ax, ay = self._beam_axis(ability)
        length = config.KAMEHAMEHA_BEAM_LENGTH_PX
        thickness = config.KAMEHAMEHA_BEAM_THICKNESS_PX

        progress = 1.0
        if ability.phase == PHASE_ACTIVE and ability.phase_age < 0.15:
            progress = ability.phase_age / 0.15
        if fade:
            progress = max(0.0, 1.0 - ability.phase_age / 0.25)

        # End point
        ex = cx + ax * length
        ey = cy + ay * length
        pulse = 0.92 + 0.08 * math.sin(self._beam_pulse * 24.0)
        thickness *= pulse

        radial_glow(
            target,
            (cx, cy),
            radius=int(thickness * 1.25 * progress),
            color=config.KAMEHAMEHA_OUTER_COLOR,
            alpha=int(160 * progress),
        )

        # Wide aura, energy body, and hot core.
        self._draw_thick_line(
            target,
            (cx, cy),
            (ex, ey),
            thickness=int(thickness * 1.75 * progress),
            color=config.KAMEHAMEHA_OUTER_COLOR,
            alpha=int(80 * progress),
        )
        self._draw_thick_line(
            target,
            (cx, cy),
            (ex, ey),
            thickness=int(thickness * 1.05 * progress),
            color=config.KAMEHAMEHA_OUTER_COLOR,
            alpha=int(180 * progress),
        )
        self._draw_thick_line(
            target,
            (cx, cy),
            (ex, ey),
            thickness=max(2, int(thickness * 0.45 * progress)),
            color=config.KAMEHAMEHA_CORE_COLOR,
            alpha=int(240 * progress),
        )

        perp = np.array([-ay, ax], dtype=np.float32)
        axis = np.array([ax, ay], dtype=np.float32)
        origin = np.array([cx, cy], dtype=np.float32)
        end = np.array([ex, ey], dtype=np.float32)

        # Flickering edge filaments make the beam feel alive instead of a
        # static cylinder.
        filament_count = 5 if progress > 0.4 else 2
        for i in range(filament_count):
            side = -1.0 if i % 2 == 0 else 1.0
            offset = side * thickness * self._rng.uniform(0.35, 0.62) * progress
            start = origin + perp * offset + axis * self._rng.uniform(0, 30)
            finish = end + perp * offset * self._rng.uniform(0.4, 1.0)
            path = jagged_path(
                (float(start[0]), float(start[1])),
                (float(finish[0]), float(finish[1])),
                segment_len=90.0,
                jitter=28.0 * progress,
                rng=self._rng,
            )
            additive_polyline(
                target,
                path,
                color=(190, 245, 255) if i < 3 else (255, 255, 255),
                width=max(1, int(2 + 3 * progress)),
                alpha=int(120 * progress),
            )

        # Speed lines crossing the beam mouth.
        for _ in range(8):
            along = self._rng.uniform(0.08, 0.8) * length * progress
            width = self._rng.uniform(-thickness * 0.8, thickness * 0.8) * progress
            p0 = origin + axis * along + perp * width
            p1 = p0 + axis * self._rng.uniform(60, 180) * progress
            additive_polyline(
                target,
                [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                color=(220, 250, 255),
                width=1,
                alpha=int(60 * progress),
            )

    def _draw_thick_line(
        self,
        target: pygame.Surface,
        start: tuple[float, float],
        end: tuple[float, float],
        thickness: int,
        color: tuple[int, int, int],
        alpha: int,
    ) -> None:
        if thickness <= 0 or alpha <= 0:
            return
        # Routes through the bbox-blitted polyline path instead of allocating a
        # full-window surface per beam layer.
        additive_polyline(target, [start, end], color, thickness, alpha)

    # ------------------------------------------------------------------

    def _sphere_center(self, ability: AbilityState) -> tuple[int, int]:
        a, b = ability.primary_hand, ability.secondary_hand
        if a is None or b is None:
            return (self.width // 2, self.height // 2)
        mx = (a.palm[0] + b.palm[0]) * 0.5
        my = (a.palm[1] + b.palm[1]) * 0.5
        return int(mx * self.width), int(my * self.height)

    def _sphere_radius(self, ability: AbilityState) -> float:
        return (
            config.KAMEHAMEHA_SPHERE_RADIUS_BASE
            + (config.KAMEHAMEHA_SPHERE_RADIUS_PEAK
               - config.KAMEHAMEHA_SPHERE_RADIUS_BASE) * ease_in_out_quad(ability.charge)
        )

    def _beam_axis(self, ability: AbilityState) -> tuple[float, float]:
        """The smoothed screen-plane direction the beam travels (see _track_aim)."""
        return float(self._aim_axis[0]), float(self._aim_axis[1])

    def _track_aim(self, ability: AbilityState) -> None:
        """Update the smoothed aim from where the cupped palms point.

        The averaged palm normal splits into a screen-plane lateral component
        (where on screen the cup points) and a toward-camera forward component.
        Facing the camera → fire at the viewer + full engulf; tilting the cup to
        a side → the beam (and a much smaller engulf) goes that way. Smoothed with
        an EMA to fight palm-normal jitter.
        """
        raw = self._compute_aim(ability)
        if raw is None:
            return
        axis, engulf = raw
        alpha = 0.3
        self._aim_axis = (
            self._aim_axis * (1.0 - alpha) + axis * alpha
        ).astype(np.float32)
        n = float(np.linalg.norm(self._aim_axis))
        if n > 1e-6:
            self._aim_axis = (self._aim_axis / n).astype(np.float32)
        self._aim_engulf = self._aim_engulf * (1.0 - alpha) + engulf * alpha

    def _compute_aim(
        self, ability: AbilityState
    ) -> tuple[np.ndarray, float] | None:
        """Raw (screen-plane axis, engulf strength 0..1) from the palm normals."""
        a, b = ability.primary_hand, ability.secondary_hand
        if a is None or b is None:
            return None
        n = (
            np.asarray(a.palm_normal, dtype=np.float32)
            + np.asarray(b.palm_normal, dtype=np.float32)
        ) * 0.5
        lat = np.array([float(n[0]), float(n[1])], dtype=np.float32)
        lat_mag = float(np.linalg.norm(lat))
        fwd = -float(n[2])   # >0 => palms face the camera (aiming at the viewer)

        # Aiming at the screen: little lateral tilt, or the palms face the camera
        # more than they point sideways. Keep the screen-outward fallback axis for
        # the (mostly washed-out) beam and engulf the screen, scaled by how flat
        # the palms face the camera.
        if lat_mag < config.KAMEHAMEHA_AIM_LATERAL_MIN or fwd >= lat_mag:
            axis = self._fallback_axis(ability)
            engulf = float(np.clip((fwd + 0.1) * 1.8, 0.45, 1.0))
            return axis, engulf

        # Aiming to a side: fire along the lateral palm direction, no screen flood.
        axis = (lat / (lat_mag + 1e-6)).astype(np.float32)
        return axis, 0.0

    def _fallback_axis(self, ability: AbilityState) -> np.ndarray:
        """Screen-outward heuristic axis (perpendicular to the inter-hand vector,
        pointing toward the furthest screen edge) used when aiming at the viewer
        or when palm normals are degenerate."""
        a, b = ability.primary_hand, ability.secondary_hand
        if a is None or b is None:
            return np.array([0.0, -1.0], dtype=np.float32)
        diff = b.palm - a.palm
        perp1 = np.array([-diff[1], diff[0]], dtype=np.float32)
        norm = float(np.linalg.norm(perp1)) + 1e-6
        perp1 /= norm
        perp2 = -perp1
        cx, cy = self._sphere_center(ability)
        screen_mid = np.array([self.width * 0.5, self.height * 0.5])
        center_px = np.array([cx, cy], dtype=np.float32)
        d1 = float(np.linalg.norm(center_px + perp1 * 200.0 - screen_mid))
        d2 = float(np.linalg.norm(center_px + perp2 * 200.0 - screen_mid))
        return perp1 if d1 >= d2 else perp2
