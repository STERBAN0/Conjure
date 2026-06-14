"""Shared rendering primitives used across effects.

Everything here is pure: takes a target Surface + parameters, draws into it,
returns nothing. No effect should reach for pygame.draw directly when one of
these helpers fits — they bake in the additive-blend conventions Aether uses.
"""

from __future__ import annotations

import colorsys
import math
import random
from typing import Sequence, Tuple

import numpy as np
import pygame

ColorRGB = Tuple[int, int, int]


def hsv_to_rgb(h: float, s: float, v: float) -> ColorRGB:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def lerp_color(a: ColorRGB, b: ColorRGB, t: float) -> ColorRGB:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def additive_circle(
    surf: pygame.Surface,
    center: Tuple[int, int],
    size: int,
    color: ColorRGB,
    alpha: int,
) -> None:
    """Draw an additive-blended circle. `size` is the radius in pixels."""
    if size <= 0 or alpha <= 0:
        return
    s = pygame.Surface((size * 2 + 2, size * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(s, (*color, max(0, min(255, alpha))), (size + 1, size + 1), size)
    surf.blit(
        s, (center[0] - size - 1, center[1] - size - 1),
        special_flags=pygame.BLEND_RGBA_ADD,
    )


def additive_ring(
    surf: pygame.Surface,
    center: Tuple[int, int],
    radius: int,
    color: ColorRGB,
    alpha: int,
    width: int = 2,
) -> None:
    """Draw an additive blended ring on its own alpha surface."""
    if radius <= 0 or alpha <= 0 or width <= 0:
        return
    pad = width + 2
    s = pygame.Surface((radius * 2 + pad * 2, radius * 2 + pad * 2), pygame.SRCALPHA)
    pygame.draw.circle(
        s,
        (*color, max(0, min(255, alpha))),
        (radius + pad, radius + pad),
        radius,
        width,
    )
    surf.blit(
        s, (center[0] - radius - pad, center[1] - radius - pad),
        special_flags=pygame.BLEND_RGBA_ADD,
    )


def radial_glow(
    surf: pygame.Surface,
    center: Tuple[int, int],
    radius: int,
    color: ColorRGB,
    alpha: int,
    layers: int = 8,
) -> None:
    """Concentric-circle radial gradient blitted additively. Cheap and pretty."""
    if radius <= 0 or alpha <= 0:
        return
    glow = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        a = int(alpha * (i / layers) ** 2 * 0.35)
        r = int(radius * (i / layers))
        pygame.draw.circle(glow, (*color, a), (radius, radius), r)
    surf.blit(
        glow, (center[0] - radius, center[1] - radius),
        special_flags=pygame.BLEND_RGBA_ADD,
    )


# --- Shared scratch surface -------------------------------------------------
# Effects draw dozens of translucent polylines per frame. Allocating a full
# 1280x720 SRCALPHA surface per primitive — and additively blitting the whole
# window — was the dominant cost whenever an effect was on screen (Chidori
# alone issues ~75 polylines/frame at peak). Instead we keep ONE persistent
# scratch surface, clear and draw only the primitive's bounding box, then blit
# only that box. Cost becomes proportional to the primitive's footprint, with
# zero per-call allocation. Visual output is identical.

_scratch: pygame.Surface | None = None


def _get_scratch(size: Tuple[int, int]) -> pygame.Surface:
    global _scratch
    if _scratch is None or _scratch.get_size() != size:
        _scratch = pygame.Surface(size, pygame.SRCALPHA)
    return _scratch


def _polyline_bbox(
    points: Sequence[Tuple[float, float]], pad: int, bounds: Tuple[int, int]
) -> pygame.Rect | None:
    """Clipped integer bounding box of ``points`` (padded by ``pad`` px), or
    None if it falls entirely outside the surface."""
    w, h = bounds
    x0 = max(0, int(math.floor(min(p[0] for p in points))) - pad)
    y0 = max(0, int(math.floor(min(p[1] for p in points))) - pad)
    x1 = min(w, int(math.ceil(max(p[0] for p in points))) + pad)
    y1 = min(h, int(math.ceil(max(p[1] for p in points))) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return pygame.Rect(x0, y0, x1 - x0, y1 - y0)


def _blit_polyline(
    surf: pygame.Surface,
    points: Sequence[Tuple[float, float]],
    rgba: Tuple[int, int, int, int],
    width: int,
    blend: int,
) -> None:
    rect = _polyline_bbox(points, width + 2, surf.get_size())
    if rect is None:
        return
    s = _get_scratch(surf.get_size())
    # Clear, draw, and blit only the dirty region. Points stay in absolute
    # surface coordinates; the clip rect confines drawing to the bbox.
    s.fill((0, 0, 0, 0), rect)
    prev_clip = s.get_clip()
    s.set_clip(rect)
    pygame.draw.lines(s, rgba, False, points, width)
    s.set_clip(prev_clip)
    surf.blit(s, rect.topleft, rect, special_flags=blend)


def additive_polyline(
    surf: pygame.Surface,
    points: Sequence[Tuple[float, float]],
    color: ColorRGB,
    width: int,
    alpha: int,
) -> None:
    """Additively-blended polyline. Drawn into a shared scratch surface and
    blitted by bounding box, so cost scales with the line's footprint rather
    than the whole window."""
    if len(points) < 2 or alpha <= 0 or width <= 0:
        return
    _blit_polyline(
        surf, points, (*color, max(0, min(255, alpha))), width,
        pygame.BLEND_RGBA_ADD,
    )


def dark_polyline(
    surf: pygame.Surface,
    points: Sequence[Tuple[float, float]],
    color: ColorRGB,
    width: int,
    alpha: int,
) -> None:
    """Normal-blended (non-additive) polyline — for dark slits that must cut
    contrast into a bright camera frame (e.g. the reality-tear void)."""
    if len(points) < 2 or alpha <= 0 or width <= 0:
        return
    _blit_polyline(surf, points, (*color, max(0, min(255, alpha))), width, 0)


def jagged_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    segment_len: float,
    jitter: float,
    rng: random.Random | None = None,
) -> list[Tuple[float, float]]:
    """Return a list of points walking from `start` toward `end` in roughly
    `segment_len` steps with random perpendicular jitter. Used for
    lightning-style polylines."""
    rng = rng or random
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy) or 1.0
    n = max(2, int(length / max(1e-3, segment_len)))
    # Unit + perpendicular
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux

    pts: list[Tuple[float, float]] = []
    for i in range(n + 1):
        t = i / n
        bx = sx + dx * t
        by = sy + dy * t
        # Bell-shaped jitter — zero at the endpoints, max in the middle.
        falloff = 4.0 * t * (1.0 - t)
        j = (rng.random() - 0.5) * 2.0 * jitter * falloff
        pts.append((bx + nx * j, by + ny * j))
    return pts


def jagged_branch(
    origin: Tuple[float, float],
    direction: Tuple[float, float],
    length: float,
    segment_len: float,
    jitter: float,
    rng: random.Random | None = None,
) -> list[Tuple[float, float]]:
    """A jagged path starting at `origin` heading along `direction` for
    `length` pixels. Direction is automatically normalised."""
    rng = rng or random
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    end = (origin[0] + dx / norm * length, origin[1] + dy / norm * length)
    return jagged_path(origin, end, segment_len, jitter, rng=rng)


_flash_cache: pygame.Surface | None = None


def draw_screen_flash(
    surf: pygame.Surface, color: ColorRGB, alpha: int
) -> None:
    """Whole-frame additive flash. Use sparingly — usually for release impact.

    The flash colour is pre-scaled by ``alpha`` and added with BLEND_RGB_ADD,
    so the intensity actually fades with ``alpha`` (the display surface has no
    per-pixel alpha, so a plain BLEND_RGBA_ADD ignored it). The full-window
    surface is cached and reused — no per-call allocation."""
    if alpha <= 0:
        return
    global _flash_cache
    size = surf.get_size()
    if _flash_cache is None or _flash_cache.get_size() != size:
        _flash_cache = pygame.Surface(size)
    f = max(0, min(255, alpha)) / 255.0
    _flash_cache.fill((int(color[0] * f), int(color[1] * f), int(color[2] * f)))
    surf.blit(_flash_cache, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


def ease_in_out_quad(t: float) -> float:
    """Smoothstep-ish easing for charge ramps."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t) if t < 1.0 else 1.0


def shake_offset(intensity: float, rng: random.Random | None = None) -> Tuple[int, int]:
    """Tiny screen-shake offset. Intensity in pixels."""
    rng = rng or random
    if intensity <= 0:
        return (0, 0)
    return (
        int(rng.uniform(-intensity, intensity)),
        int(rng.uniform(-intensity, intensity)),
    )
