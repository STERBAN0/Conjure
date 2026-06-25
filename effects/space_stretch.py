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

Scaling (restored original):
- Warp magnitude grows directly with the pixel distance between the two open
  palms (``stretch = clip(length / max(W, H) * 1.6)``), so the rubber-band
  stretches more the wider you pull your hands apart. There is no charge — the
  warp simply happens while the open-palm pose is held, with a short warmup
  fade-in on entry.
- The warp is centered on the live midpoint of the two palms so it tracks
  them across the frame.

Performance notes:
- Displacement field computed at 1/DOWNSCALE resolution (~57k vs 0.9M px),
  then upscaled to full res with cv2.resize. Visually identical for smooth
  warps.
- All constant grids (full-res base, coarse grid, sigma) precomputed once
  in __init__. No np.mgrid per frame.
- _add_midpoint_glow uses a precomputed distance-squared map (patch coords
  relative to patch origin are fixed for a given radius; we recompute only
  on radius change, which is cheap for a small patch).
"""

from __future__ import annotations

import cv2
import numpy as np
import pygame

import config
from core.state import AbilityState, GestureSignals
from effects.base import LAYER_BG, Effect
from effects.utils import additive_circle, additive_polyline, additive_ring, radial_glow

# Coarse-grid downscale factor: displacement field is computed at
# (H/DOWNSCALE) x (W/DOWNSCALE) then upscaled to full resolution.
_DOWNSCALE: int = 4


class SpaceStretchEffect(Effect):
    layer = LAYER_BG
    name = "space_stretch"
    ability_name = "space_stretch"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)

        # Full-resolution base grid (float32, identity mapping).
        ys_full, xs_full = np.mgrid[0:height, 0:width].astype(np.float32)
        self._base_x: np.ndarray = xs_full          # (H, W)
        self._base_y: np.ndarray = ys_full          # (H, W)

        # Coarse grid (used for the expensive tanh/exp math).
        ch = (height + _DOWNSCALE - 1) // _DOWNSCALE
        cw = (width + _DOWNSCALE - 1) // _DOWNSCALE
        self._ch = ch
        self._cw = cw
        ys_c, xs_c = np.mgrid[0:ch, 0:cw].astype(np.float32)
        # Scale coarse indices back to full-pixel coordinates.
        self._coarse_x: np.ndarray = xs_c * _DOWNSCALE   # (ch, cw)
        self._coarse_y: np.ndarray = ys_c * _DOWNSCALE   # (ch, cw)

        # Precomputed sigma (depends only on fixed width/height).
        self._sigma_f32: np.float32 = np.float32(
            max(height, width) * config.SPACE_STRETCH_AXIS_FALLOFF
        )

        # Preallocated scratch buffers — all float32, avoid per-frame heap alloc.
        self._rx_c_buf: np.ndarray = np.empty((ch, cw), np.float32)    # coarse rx
        self._ry_c_buf: np.ndarray = np.empty((ch, cw), np.float32)    # coarse ry
        self._t_c_buf: np.ndarray = np.empty((ch, cw), np.float32)     # along projection
        self._s_c_buf: np.ndarray = np.empty((ch, cw), np.float32)     # perp projection
        self._along_buf: np.ndarray = np.empty((ch, cw), np.float32)   # tanh result
        self._dx_c_buf: np.ndarray = np.empty((ch, cw), np.float32)    # coarse dx / temp
        self._dy_c_buf: np.ndarray = np.empty((ch, cw), np.float32)    # coarse dy
        self._dx_full_buf: np.ndarray = np.empty((height, width), np.float32)
        self._dy_full_buf: np.ndarray = np.empty((height, width), np.float32)
        self._map_x_buf: np.ndarray = np.empty((height, width), np.float32)
        self._map_y_buf: np.ndarray = np.empty((height, width), np.float32)

    # ------------------------------------------------------------------

    def pre_process_frame(
        self,
        frame_bgr: np.ndarray,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> np.ndarray:
        if ability.primary_hand is None or ability.secondary_hand is None:
            return frame_bgr

        # Pure-Python scalar math to avoid 2-element numpy array allocs.
        ax = float(ability.primary_hand.palm[0]) * self.width
        ay = float(ability.primary_hand.palm[1]) * self.height
        bx = float(ability.secondary_hand.palm[0]) * self.width
        by = float(ability.secondary_hand.palm[1]) * self.height
        dx = bx - ax
        dy = by - ay
        length = (dx * dx + dy * dy) ** 0.5 + 1e-6
        ux, uy = dx / length, dy / length      # along axis

        # Centre on the live midpoint of the two palms so the warp tracks them.
        cx = (ax + bx) * 0.5
        cy = (ay + by) * 0.5

        # ORIGINAL scaling (the "perfect" version): amplitude grows directly with
        # the pixel distance between the hands, so the rubber-band stretches as you
        # pull your open palms apart. Warmup is a short fade-in on entry.
        warmup = float(np.clip(ability.age * 1.4, 0.0, 1.0))
        span_t = float(np.clip(length / max(self.width, self.height) * 1.6, 0.0, 1.0))
        amplitude = config.SPACE_STRETCH_MAX_DISPLACEMENT_PX * warmup * span_t

        if amplitude < 0.5:
            return frame_bgr

        # ---- Displacement on the coarse grid (all out= to avoid allocs) -----
        ux_f32 = np.float32(ux)
        uy_f32 = np.float32(uy)
        half_f32 = np.float32(max(length * 0.5, 1.0))
        amplitude_f32 = np.float32(amplitude)
        sigma_f32 = self._sigma_f32
        neg_inv_sig2 = np.float32(-1.0) / (sigma_f32 * sigma_f32)

        # rx_c = coarse_x - cx,  ry_c = coarse_y - cy  (in-place)
        cx_f32 = np.float32(cx)
        cy_f32 = np.float32(cy)
        np.subtract(self._coarse_x, cx_f32, out=self._rx_c_buf)
        np.subtract(self._coarse_y, cy_f32, out=self._ry_c_buf)

        # t_c = rx*ux + ry*uy  (along axis) — compute into t_c_buf
        np.multiply(self._rx_c_buf, ux_f32, out=self._t_c_buf)
        np.multiply(self._ry_c_buf, uy_f32, out=self._along_buf)   # temp: ry*uy
        self._t_c_buf += self._along_buf

        # s_c = rx*(-uy) + ry*ux  (perpendicular) — compute into s_c_buf
        np.multiply(self._rx_c_buf, -uy_f32, out=self._s_c_buf)
        np.multiply(self._ry_c_buf, ux_f32, out=self._along_buf)   # temp: ry*ux
        self._s_c_buf += self._along_buf

        # along_buf = tanh(t_c / half) * amplitude  — divide in-place, then tanh
        np.multiply(self._t_c_buf, np.float32(1.0) / half_f32, out=self._along_buf)
        np.tanh(self._along_buf, out=self._along_buf)
        self._along_buf *= amplitude_f32

        # dx_c_buf = exp(-s^2/sigma^2)  (Gaussian falloff)
        np.multiply(self._s_c_buf, self._s_c_buf, out=self._dx_c_buf)
        self._dx_c_buf *= neg_inv_sig2
        np.exp(self._dx_c_buf, out=self._dx_c_buf)
        # apply falloff to along_buf
        self._along_buf *= self._dx_c_buf

        # Displacement vector components on coarse grid.
        np.multiply(-ux_f32, self._along_buf, out=self._dx_c_buf)
        np.multiply(-uy_f32, self._along_buf, out=self._dy_c_buf)

        # ---- Upscale displacement to full resolution (dst= avoids alloc) ----
        cv2.resize(
            self._dx_c_buf, (self.width, self.height),
            dst=self._dx_full_buf, interpolation=cv2.INTER_LINEAR,
        )
        cv2.resize(
            self._dy_c_buf, (self.width, self.height),
            dst=self._dy_full_buf, interpolation=cv2.INTER_LINEAR,
        )

        np.add(self._base_x, self._dx_full_buf, out=self._map_x_buf)
        np.add(self._base_y, self._dy_full_buf, out=self._map_y_buf)

        warped = cv2.remap(
            frame_bgr,
            self._map_x_buf, self._map_y_buf,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Heat-up at midpoint — additive cyan glow on the BGR frame.
        glow_strength = warmup * span_t
        if glow_strength > 0.05:
            warped = self._add_midpoint_glow(warped, (cx, cy), glow_strength)
        return warped

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        # Draw translucent stress lines that mirror the displacement field.
        if ability.primary_hand is None or ability.secondary_hand is None:
            return
        a = ability.primary_hand.palm * np.array([self.width, self.height])
        b = ability.secondary_hand.palm * np.array([self.width, self.height])
        # Centre the FG membrane on the live palm midpoint, matching the BG warp.
        center = (a + b) * 0.5
        diff = b - a
        length = float(np.linalg.norm(diff)) + 1e-6
        ux, uy = diff[0] / length, diff[1] / length
        nx, ny = -uy, ux

        warmup = float(np.clip(ability.age * 1.4, 0.0, 1.0))
        span_t = float(np.clip(length / max(self.width, self.height) * 1.6, 0.0, 1.0))
        intensity = warmup * span_t

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
        center: tuple[float, float],
        strength: float,
    ) -> np.ndarray:
        cx, cy = int(center[0]), int(center[1])
        h, w = frame_bgr.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return frame_bgr
        radius = int(120 * strength)
        if radius <= 0:
            return frame_bgr
        x0, x1 = max(0, cx - radius), min(w, cx + radius)
        y0, y1 = max(0, cy - radius), min(h, cy + radius)
        ph, pw = y1 - y0, x1 - x0
        if ph <= 0 or pw <= 0:
            return frame_bgr

        # Build patch distance-squared relative to patch origin (cheap).
        ys_p = np.arange(ph, dtype=np.float32)[:, np.newaxis]
        xs_p = np.arange(pw, dtype=np.float32)[np.newaxis, :]
        d2 = (xs_p - (cx - x0)) ** 2 + (ys_p - (cy - y0)) ** 2
        falloff = np.exp(-d2 / max(1.0, radius * radius * 0.4)).astype(np.float32)

        sub = frame_bgr[y0:y1, x0:x1].astype(np.int32)
        sub[..., 0] += (falloff * (200 * strength)).astype(np.int32)  # B
        sub[..., 1] += (falloff * (130 * strength)).astype(np.int32)  # G
        sub[..., 2] += (falloff * (30 * strength)).astype(np.int32)   # R
        np.clip(sub, 0, 255, out=sub)
        frame_bgr[y0:y1, x0:x1] = sub.astype(np.uint8)
        return frame_bgr
