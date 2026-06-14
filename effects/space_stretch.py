"""Space stretch — elastic membrane sheared along the hand-to-hand axis.

The previous radial-pinch warp was the wrong metaphor: it pulled space
inward toward a point, when the user wanted "space itself is being
stretched between my hands". This effect implements the correct version.

Implementation:
- Project every pixel onto a local frame (t along the hand-to-hand axis,
  s perpendicular to it).
- Displace pixels *along* the axis as a function of t (pull pixels in the
  middle outward toward the hands as the hands separate).
- Displacement falls off in s with a Gaussian, so only pixels near the axis
  are affected — preserves the rubber-band feeling.
- Overlay faint cyan grid lines that get displaced by the same field, so
  the "fabric" reads visually.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import pygame

import config
from core.state import AbilityState, FrameState, GestureSignals
from effects.base import LAYER_BG, Effect
from effects.utils import additive_circle, additive_polyline, additive_ring, radial_glow


class SpaceStretchEffect(Effect):
    layer = LAYER_BG
    name = "space_stretch"
    ability_name = "space_stretch"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        self._base_x = xs
        self._base_y = ys
        # Identity grid surface used for the overlaid grid lines.
        self._grid_surface = self._make_grid_surface()

    # ------------------------------------------------------------------

    def pre_process_frame(
        self,
        frame_bgr: np.ndarray,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> np.ndarray:
        if ability.primary_hand is None or ability.secondary_hand is None:
            return frame_bgr

        a = ability.primary_hand.palm * np.array([self.width, self.height])
        b = ability.secondary_hand.palm * np.array([self.width, self.height])
        center = (a + b) * 0.5
        diff = b - a
        length = float(np.linalg.norm(diff)) + 1e-6
        ux, uy = diff[0] / length, diff[1] / length      # along axis
        nx, ny = -uy, ux                                  # perpendicular

        # Coords relative to center
        rx = self._base_x - center[0]
        ry = self._base_y - center[1]
        # Project
        t = rx * ux + ry * uy           # along axis
        s = rx * nx + ry * ny           # perpendicular

        # Strength scales with charge & sustained pose duration. We use
        # ability.age as a proxy so the effect feels "warming up" rather
        # than instantly maxed.
        warmup = float(np.clip(ability.age * 1.4, 0.0, 1.0))
        stretch = float(np.clip(length / max(self.width, self.height) * 1.6, 0.0, 1.0))
        amplitude = config.SPACE_STRETCH_MAX_DISPLACEMENT_PX * warmup * stretch

        # Sigmoid-ish displacement along axis: pulls outward toward the
        # hands. tanh(t / half_length) gives the right pull-toward-edges.
        half = length * 0.5
        along_displacement = np.tanh(t / max(half, 1.0)) * amplitude
        # Gaussian falloff in s
        sigma = max(self.height, self.width) * config.SPACE_STRETCH_AXIS_FALLOFF
        falloff = np.exp(-(s * s) / (sigma * sigma))
        along_displacement *= falloff

        map_x = self._base_x - ux * along_displacement
        map_y = self._base_y - uy * along_displacement

        warped = cv2.remap(
            frame_bgr,
            map_x.astype(np.float32), map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Heat-up at midpoint — additive cyan glow on the BGR frame
        glow_strength = warmup * stretch
        if glow_strength > 0.05:
            warped = self._add_midpoint_glow(warped, center, glow_strength)
        return warped

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        # Draw translucent stress lines that mirror the displacement field.
        # This makes the background warp readable even when the camera image
        # has low texture.
        if ability.primary_hand is None or ability.secondary_hand is None:
            return
        a = ability.primary_hand.palm * np.array([self.width, self.height])
        b = ability.secondary_hand.palm * np.array([self.width, self.height])
        center = (a + b) * 0.5
        diff = b - a
        length = float(np.linalg.norm(diff)) + 1e-6
        ux, uy = diff[0] / length, diff[1] / length
        nx, ny = -uy, ux

        warmup = float(np.clip(ability.age * 1.4, 0.0, 1.0))
        stretch = float(np.clip(length / max(self.width, self.height) * 1.6, 0.0, 1.0))
        intensity = warmup * stretch

        if intensity <= 0.05:
            return

        half = length * 0.5
        amplitude = config.SPACE_STRETCH_MAX_DISPLACEMENT_PX * intensity
        sigma = max(self.height, self.width) * config.SPACE_STRETCH_AXIS_FALLOFF
        membrane_half_width = min(180.0, max(70.0, length * 0.35))

        def deform(point: np.ndarray) -> tuple[float, float]:
            rel = point - center
            t = float(rel[0] * ux + rel[1] * uy)
            s = float(rel[0] * nx + rel[1] * ny)
            along = np.tanh(t / max(half, 1.0)) * amplitude
            along *= float(np.exp(-(s * s) / (sigma * sigma)))
            q = point + np.array([ux, uy]) * along
            return float(q[0]), float(q[1])

        strand_offsets = np.linspace(-membrane_half_width, membrane_half_width, 9)
        for offset in strand_offsets:
            falloff = 1.0 - abs(float(offset)) / (membrane_half_width + 1e-6)
            pts = []
            for t in np.linspace(-half * 1.12, half * 1.12, 32):
                base = center + np.array([ux, uy]) * t + np.array([nx, ny]) * offset
                pts.append(deform(base))
            additive_polyline(
                target,
                pts,
                color=config.SPACE_STRETCH_GRID_COLOR,
                width=max(1, int(1 + 2 * intensity * falloff)),
                alpha=int(70 * intensity * (0.35 + 0.65 * falloff)),
            )

        for t in np.linspace(-half * 0.85, half * 0.85, 7):
            pts = []
            for offset in np.linspace(-membrane_half_width, membrane_half_width, 18):
                base = center + np.array([ux, uy]) * t + np.array([nx, ny]) * offset
                pts.append(deform(base))
            additive_polyline(
                target,
                pts,
                color=(180, 245, 255),
                width=1,
                alpha=int(46 * intensity),
            )

        main = [deform(p) for p in (a, center, b)]
        additive_polyline(
            target,
            main,
            color=(210, 255, 255),
            width=int(2 + 5 * intensity),
            alpha=int(210 * intensity),
        )
        radial_glow(
            target,
            (int(center[0]), int(center[1])),
            radius=int(80 + 140 * intensity),
            color=config.SPACE_STRETCH_GRID_COLOR,
            alpha=int(80 * intensity),
            layers=10,
        )
        for point in (a, b):
            p = (int(point[0]), int(point[1]))
            additive_ring(
                target,
                p,
                radius=int(22 + 22 * intensity),
                color=(210, 255, 255),
                alpha=int(170 * intensity),
                width=max(1, int(2 + 3 * intensity)),
            )
            additive_circle(
                target,
                p,
                size=int(4 + 8 * intensity),
                color=(230, 255, 255),
                alpha=int(190 * intensity),
            )

    # ------------------------------------------------------------------

    def _make_grid_surface(self) -> pygame.Surface:
        """Pre-rendered grid we *don't* currently composite — kept as a
        future hook so we can add fabric lines without a per-frame cost."""
        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        spacing = config.SPACE_STRETCH_GRID_SPACING_PX
        color = (*config.SPACE_STRETCH_GRID_COLOR, 35)
        for x in range(0, self.width, spacing):
            pygame.draw.line(s, color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, spacing):
            pygame.draw.line(s, color, (0, y), (self.width, y), 1)
        return s

    def _add_midpoint_glow(
        self,
        frame_bgr: np.ndarray,
        center: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        cx, cy = int(center[0]), int(center[1])
        h, w = frame_bgr.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return frame_bgr
        # Build a glow patch and additively blend onto the BGR frame.
        radius = int(120 * strength)
        if radius <= 0:
            return frame_bgr
        x0, x1 = max(0, cx - radius), min(w, cx + radius)
        y0, y1 = max(0, cy - radius), min(h, cy + radius)
        sub = frame_bgr[y0:y1, x0:x1].astype(np.int32)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        falloff = np.exp(-d2 / max(1, radius * radius * 0.4))
        # OpenCV channel order: BGR
        sub[..., 0] += (falloff * 200 * strength).astype(np.int32)   # B
        sub[..., 1] += (falloff * 130 * strength).astype(np.int32)   # G
        sub[..., 2] += (falloff * 30 * strength).astype(np.int32)    # R (less)
        np.clip(sub, 0, 255, out=sub)
        frame_bgr[y0:y1, x0:x1] = sub.astype(np.uint8)
        return frame_bgr
