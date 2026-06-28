"""Vector hand-sign diagrams for the in-app manual.

Each ``draw_*`` paints one ability's hand pose into a rect; ``DRAW_REGISTRY``
maps an ability id to its draw function. These are the fallback used by
``Manual`` when a ``docs/manual_images/<id>.png`` is missing. Extracted from
``manual.py`` to keep that module focused on the overlay/paging logic.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pygame


def _rounded_rect(
    surf: pygame.Surface,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    radius: int,
    width: int = 0,
) -> None:
    """pygame.draw.rect with border_radius, works on SRCALPHA surfaces."""
    pygame.draw.rect(surf, color, rect, width=width, border_radius=radius)


def _finger(
    surf: pygame.Surface,
    color: tuple[int, int, int],
    base: tuple[float, float],
    tip: tuple[float, float],
    w: int = 10,
) -> None:
    """Draw one cartoon finger as a rounded thick line."""
    pygame.draw.line(surf, color, (int(base[0]), int(base[1])),
                     (int(tip[0]), int(tip[1])), w)
    pygame.draw.circle(surf, color, (int(tip[0]), int(tip[1])), w // 2)
    pygame.draw.circle(surf, color, (int(base[0]), int(base[1])), w // 2)


def _palm_rect(
    rect: pygame.Rect,
    side: str = "center",   # "center", "left", "right"
    frac: float = 0.55,
) -> pygame.Rect:
    """Return a palm-sized Rect within the given Rect, for one or two hands."""
    pw = int(rect.width * frac * 0.45)
    ph = int(rect.height * frac * 0.40)
    cy = rect.centery + int(rect.height * 0.08)
    if side == "left":
        cx = rect.centerx - int(rect.width * 0.20)
    elif side == "right":
        cx = rect.centerx + int(rect.width * 0.20)
    else:
        cx = rect.centerx
    return pygame.Rect(cx - pw // 2, cy - ph // 2, pw, ph)


# ---------------------------------------------------------------------------
# Per-ability draw functions  (vector fallbacks)
# ---------------------------------------------------------------------------

_SKIN = (240, 200, 160)
_SKIN_DARK = (210, 165, 120)
_OUTLINE = (60, 40, 20)
_LIGHTNING = (220, 240, 80)
_BLUE_ORB = (80, 160, 255)
_FROST_C = (160, 220, 255)
_FIRE_C = (255, 130, 30)


def draw_fireball(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """One hand: only the INDEX finger up, ember at the fingertip — gesture: fireball."""
    pr = _palm_rect(rect)
    _rounded_rect(surf, _SKIN, pr, 14)
    _rounded_rect(surf, _OUTLINE, pr, 14, 2)

    cx, cy = pr.centerx, pr.top
    fh = int(rect.height * 0.30)
    fw = 12

    # Index finger — lone finger pointing straight up
    _finger(surf, _SKIN, (cx, cy), (cx, cy - fh), fw)
    _rounded_rect(surf, _OUTLINE,
                  pygame.Rect(cx - fw // 2, cy - fh - fw // 2, fw, fw), 4, 2)

    # Folded stubs for middle, ring, pinky
    for ox, frac in ((14, 0.45), (28, 0.40), (40, 0.35)):
        _rounded_rect(surf, _SKIN_DARK,
                      pygame.Rect(cx + ox, cy - 8, fw - 2, int(fh * frac)), 5)
        _rounded_rect(surf, _OUTLINE,
                      pygame.Rect(cx + ox, cy - 8, fw - 2, int(fh * frac)), 5, 2)

    # Thumb stub on left
    _rounded_rect(surf, _SKIN_DARK,
                  pygame.Rect(pr.left - 8, pr.centery - 6, fw - 1, int(fh * 0.4)), 5)

    # Fire ember at the fingertip
    ember_r = int(rect.width * 0.09)
    ex, ey = cx, cy - fh - ember_r - 4
    pygame.draw.circle(surf, (255, 240, 80), (ex, ey), ember_r)
    pygame.draw.circle(surf, _FIRE_C, (ex, ey), ember_r, 3)
    for angle_deg in (-30, 0, 30):
        a = math.radians(angle_deg - 90)
        fx = ex + int(math.cos(a) * (ember_r + 8))
        fy = ey + int(math.sin(a) * (ember_r + 8))
        pygame.draw.line(surf, _FIRE_C, (ex, ey), (fx, fy), 3)


def draw_rasengan(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Two hands stacked — lower palm UP cradles the orb, top hand stirs — gesture: rasengan."""
    # Lower hand — open palm facing up, centered in the lower half
    lower_cy = rect.centery + int(rect.height * 0.18)
    pw = int(rect.width * 0.48)
    ph = int(rect.height * 0.22)
    pr_low = pygame.Rect(rect.centerx - pw // 2, lower_cy - ph // 2, pw, ph)
    _rounded_rect(surf, _SKIN, pr_low, 12)
    _rounded_rect(surf, _OUTLINE, pr_low, 12, 2)

    # Fingers on lower palm pointing upward (palm-up = fingers point up from top edge)
    fh_low = int(rect.height * 0.18)
    fw = 10
    for ox in (-18, -6, 6, 18):
        _finger(surf, _SKIN,
                (pr_low.centerx + ox, pr_low.top),
                (pr_low.centerx + ox, pr_low.top - fh_low), fw)

    # Blue spinning orb resting above the lower palm, at center
    orb_r = int(rect.width * 0.14)
    orb_cx = rect.centerx
    orb_cy = pr_low.top - orb_r - 4
    pygame.draw.circle(surf, (200, 230, 255), (orb_cx, orb_cy), orb_r)
    pygame.draw.circle(surf, _BLUE_ORB, (orb_cx, orb_cy), orb_r, 4)
    pygame.draw.circle(surf, (255, 255, 255),
                       (orb_cx - orb_r // 3, orb_cy - orb_r // 3), orb_r // 4)
    pygame.draw.ellipse(surf, _BLUE_ORB,
                        pygame.Rect(orb_cx - orb_r, orb_cy - orb_r // 3,
                                    orb_r * 2, orb_r * 2 // 3), 2)

    # Upper hand — cupped, hovering just above the orb, slightly smaller
    upper_top = orb_cy - orb_r - ph // 2 - 8
    pr_up = pygame.Rect(rect.centerx - pw // 2 + 6, upper_top, pw - 12, ph)
    _rounded_rect(surf, _SKIN, pr_up, 10)
    _rounded_rect(surf, _OUTLINE, pr_up, 10, 2)

    # Fingers on upper hand curling down toward the orb (stirring motion)
    fh_up = int(rect.height * 0.12)
    for ox in (-14, -5, 5, 14):
        tip_x = pr_up.centerx + ox + ox // 3
        _finger(surf, _SKIN,
                (pr_up.centerx + ox, pr_up.bottom),
                (tip_x, pr_up.bottom + fh_up), fw - 1)


def draw_chidori(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """One hand: index + middle extended up (V sign) — gesture: chidori."""
    pr = _palm_rect(rect)
    _rounded_rect(surf, _SKIN, pr, 12)
    _rounded_rect(surf, _OUTLINE, pr, 12, 2)

    fh = int(rect.height * 0.30)
    fw = 11
    cx = pr.centerx
    cy = pr.top

    # Index finger (left of center)
    _finger(surf, _SKIN, (cx - 10, cy), (cx - 10, cy - fh), fw)
    _rounded_rect(surf, _OUTLINE,
                  pygame.Rect(cx - 10 - fw // 2, cy - fh - fw // 2, fw, fw), 4, 2)
    # Middle finger (right of center)
    _finger(surf, _SKIN, (cx + 8, cy), (cx + 8, cy - fh - 4), fw)
    _rounded_rect(surf, _OUTLINE,
                  pygame.Rect(cx + 8 - fw // 2, cy - fh - 4 - fw // 2, fw, fw), 4, 2)

    # Folded stubs for ring + pinky
    _rounded_rect(surf, _SKIN_DARK,
                  pygame.Rect(cx + 22, cy - 8, fw, int(fh * 0.4)), 5)
    _rounded_rect(surf, _SKIN_DARK,
                  pygame.Rect(cx + 36, cy - 4, fw - 2, int(fh * 0.35)), 4)

    # Thumb stub
    _rounded_rect(surf, _SKIN_DARK,
                  pygame.Rect(pr.left - 8, pr.centery - 8, fw - 1, int(fh * 0.4)), 5)

    # Lightning crackles around both fingertips
    for tip_x, tip_y in [(cx - 10, cy - fh - 4), (cx + 8, cy - fh - 8)]:
        pts = [
            (tip_x, tip_y - 6), (tip_x - 10, tip_y - 18),
            (tip_x + 4, tip_y - 18), (tip_x - 6, tip_y - 34),
            (tip_x + 12, tip_y - 20), (tip_x + 2, tip_y - 20),
            (tip_x + 10, tip_y - 8),
        ]
        if len(pts) >= 2:
            pygame.draw.lines(surf, _LIGHTNING, False, pts, 2)


def draw_time_freeze(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """One closed FIST, palm to the camera, held still — gesture: time_freeze."""
    pr = _palm_rect(rect)
    _rounded_rect(surf, _SKIN, pr, 16)
    _rounded_rect(surf, _OUTLINE, pr, 16, 2)

    cx, cy = pr.centerx, pr.top
    knuckle_w = pr.width // 4

    # Knuckle row across the top of the fist
    for i in range(4):
        kx = pr.left + i * knuckle_w + knuckle_w // 2
        _rounded_rect(surf, _SKIN_DARK,
                      pygame.Rect(kx - 6, cy - 8, 12, 10), 5)
        _rounded_rect(surf, _OUTLINE,
                      pygame.Rect(kx - 6, cy - 8, 12, 10), 5, 2)

    # Thumb stub on the side
    _rounded_rect(surf, _SKIN_DARK,
                  pygame.Rect(pr.right - 8, pr.centery - 6, 18, 10), 5)
    _rounded_rect(surf, _OUTLINE,
                  pygame.Rect(pr.right - 8, pr.centery - 6, 18, 10), 5, 2)

    # Clock face accent — hands frozen mid-tick
    clk_r = int(rect.width * 0.10)
    clk_cx = cx
    clk_cy = cy - clk_r - 10
    pygame.draw.circle(surf, (200, 210, 240), (clk_cx, clk_cy), clk_r, 3)
    # Hour and minute hands
    pygame.draw.line(surf, (80, 100, 180),
                     (clk_cx, clk_cy),
                     (clk_cx, clk_cy - int(clk_r * 0.7)), 2)
    pygame.draw.line(surf, (80, 100, 180),
                     (clk_cx, clk_cy),
                     (clk_cx + int(clk_r * 0.5), clk_cy - int(clk_r * 0.3)), 2)
    # "Frozen" tick marks to emphasise stillness
    for tick_ang in range(0, 360, 60):
        ta = math.radians(tick_ang)
        tx1 = clk_cx + int(math.cos(ta) * (clk_r - 4))
        ty1 = clk_cy + int(math.sin(ta) * (clk_r - 4))
        tx2 = clk_cx + int(math.cos(ta) * clk_r)
        ty2 = clk_cy + int(math.sin(ta) * clk_r)
        pygame.draw.line(surf, (80, 100, 180), (tx1, ty1), (tx2, ty2), 1)


def draw_kamehameha(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Two hands raised together forming a triangle apex — gesture: kamehameha."""
    for side, xd in (("left", -1), ("right", 1)):
        pr = _palm_rect(rect, side)
        _rounded_rect(surf, _SKIN, pr, 12)
        _rounded_rect(surf, _OUTLINE, pr, 12, 2)

        cx, cy_top = pr.centerx, pr.top
        fh = int(rect.height * 0.20)
        fw = 9

        # Cupped fingers pointing inward toward each other
        for ox in [-14, -5, 5, 14]:
            bx = cx + ox
            tip_x = bx - xd * int(abs(ox) * 0.5 + fh * 0.3)
            tip_y = cy_top - fh // 2
            _finger(surf, _SKIN, (bx, cy_top), (tip_x, tip_y), fw)

        # Thumb
        ty = pr.centery
        _finger(surf, _SKIN, (pr.left if xd == -1 else pr.right, ty),
                (pr.left - 12 if xd == -1 else pr.right + 12, ty - 8), fw)

    # Energy orb forming between the close hands
    orb_r = int(rect.width * 0.13)
    pygame.draw.circle(surf, (220, 240, 255, 100), rect.center, orb_r)
    pygame.draw.circle(surf, _BLUE_ORB, rect.center, orb_r, 4)
    pygame.draw.circle(surf, (255, 255, 255),
                       (rect.centerx - orb_r // 3, rect.centery - orb_r // 3),
                       orb_r // 4)


def draw_space_stretch(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Two open palms held WIDE apart — gesture: space_stretch."""
    for side in ("left", "right"):
        pr = _palm_rect(rect, side)
        _rounded_rect(surf, _SKIN, pr, 12)
        _rounded_rect(surf, _OUTLINE, pr, 12, 2)

        cx, cy = pr.centerx, pr.top
        fh = int(rect.height * 0.28)
        fw = 9

        for ox in (-14, -5, 5, 14):
            _finger(surf, _SKIN, (cx + ox, cy), (cx + ox, cy - fh), fw)
        # Thumb
        _finger(surf, _SKIN,
                (pr.left + 4 if side == "right" else pr.right - 4, pr.centery),
                (pr.left - 8 if side == "right" else pr.right + 8, pr.centery - 6), fw)

    # Wavy distortion lines between the wide-apart hands
    mid_x = rect.centerx
    for dy_off in (-20, 0, 20):
        y = rect.centery + dy_off
        pts = [(mid_x + int(math.sin(i * 0.6) * 6), y + i) for i in range(-18, 19, 3)]
        if len(pts) >= 2:
            pygame.draw.lines(surf, (120, 180, 255), False, pts, 2)


def draw_reality_tear(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Two FISTS pulled apart — gesture: reality_tear."""
    # Draw each hand as a closed fist
    for side in ("left", "right"):
        pr = _palm_rect(rect, side)
        _rounded_rect(surf, _SKIN, pr, 16)
        _rounded_rect(surf, _OUTLINE, pr, 16, 2)

        cy = pr.top
        knuckle_w = pr.width // 4

        # Knuckle row
        for i in range(4):
            kx = pr.left + i * knuckle_w + knuckle_w // 2
            _rounded_rect(surf, _SKIN_DARK,
                          pygame.Rect(kx - 5, cy - 7, 10, 9), 4)
            _rounded_rect(surf, _OUTLINE,
                          pygame.Rect(kx - 5, cy - 7, 10, 9), 4, 2)

        # Thumb stub on outer side
        thumb_x = pr.right - 6 if side == "left" else pr.left - 12
        _rounded_rect(surf, _SKIN_DARK,
                      pygame.Rect(thumb_x, pr.centery - 5, 16, 9), 4)

    # "Pulling apart" arrows between the fists
    ax = rect.centerx
    ay = rect.centery
    pygame.draw.line(surf, (180, 100, 255), (ax - 6, ay), (ax - 28, ay), 2)
    pygame.draw.line(surf, (180, 100, 255), (ax + 6, ay), (ax + 28, ay), 2)
    # Arrowheads
    for dx in (-1, 1):
        tip_x = ax + dx * 28
        for dy in (-4, 4):
            pygame.draw.line(surf, (180, 100, 255),
                             (tip_x, ay), (tip_x - dx * 8, ay + dy), 2)

    # Jagged dimensional rift in the center
    tx = rect.centerx
    ty_start = rect.centery - int(rect.height * 0.28)
    ty_end = rect.centery + int(rect.height * 0.28)
    n = 10
    pts = []
    for i in range(n + 1):
        t = i / n
        y = int(ty_start + t * (ty_end - ty_start))
        jitter = int(math.sin(i * 1.8) * 10) * (1 if i % 2 == 0 else -1)
        pts.append((tx + jitter, y))
    if len(pts) >= 2:
        pygame.draw.lines(surf, (180, 100, 255), False, pts, 3)
        pygame.draw.lines(surf, (220, 200, 255), False, pts, 1)


def draw_frost_nova(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Wrists CROSSED in an X — gesture: frost_nova (fire: spread hands)."""
    # Left hand forearm/wrist crosses over right — draw as X at center
    pr_l = pygame.Rect(
        rect.centerx - int(rect.width * 0.28) - 30,
        rect.centery - int(rect.height * 0.15),
        int(rect.width * 0.38),
        int(rect.height * 0.35),
    )
    pr_r = pygame.Rect(
        rect.centerx - int(rect.width * 0.10) + 30,
        rect.centery - int(rect.height * 0.15),
        int(rect.width * 0.38),
        int(rect.height * 0.35),
    )

    for pr, lean in ((pr_l, 18), (pr_r, -18)):
        _rounded_rect(surf, _SKIN, pr, 12)
        _rounded_rect(surf, _OUTLINE, pr, 12, 2)
        cx, cy = pr.centerx, pr.top
        fh = int(rect.height * 0.26)
        fw = 9
        for ox in (-12, -4, 5, 13):
            tip_x = cx + ox + lean // 3
            _finger(surf, _SKIN, (cx + ox, cy), (tip_x, cy - fh), fw)

    # X crossing lines at the wrist center
    cx, cy = rect.centerx, rect.centery + int(rect.height * 0.05)
    arm_len = int(rect.width * 0.12)
    pygame.draw.line(surf, _OUTLINE, (cx - arm_len, cy - arm_len), (cx + arm_len, cy + arm_len), 4)
    pygame.draw.line(surf, _OUTLINE, (cx + arm_len, cy - arm_len), (cx - arm_len, cy + arm_len), 4)

    # Frost shard asterisk at the cross point
    fx, fy = rect.centerx, rect.centery - int(rect.height * 0.05)
    for ang in range(0, 360, 45):
        a = math.radians(ang)
        r = int(rect.width * 0.14)
        ex = fx + int(math.cos(a) * r)
        ey = fy + int(math.sin(a) * r)
        pygame.draw.line(surf, _FROST_C, (fx, fy), (ex, ey), 2)
        pygame.draw.circle(surf, (220, 240, 255), (ex, ey), 3)
    pygame.draw.circle(surf, (240, 250, 255), (fx, fy), 6)
    pygame.draw.circle(surf, _FROST_C, (fx, fy), 6, 2)


def draw_laser_eyes(surf: pygame.Surface, rect: pygame.Rect) -> None:
    """Face: BOTH eyes closed to charge; OPEN to fire beams — gesture: laser_eyes."""
    head_w = int(rect.width * 0.40)
    head_h = int(rect.height * 0.55)
    head_x = rect.centerx - head_w // 2
    head_y = rect.centery - head_h // 2 - int(rect.height * 0.04)
    head_rect = pygame.Rect(head_x, head_y, head_w, head_h)

    pygame.draw.ellipse(surf, (240, 210, 170), head_rect)
    pygame.draw.ellipse(surf, _OUTLINE, head_rect, 3)

    # Closed eyes (curved arcs)
    eye_y = head_y + int(head_h * 0.38)
    for ex_off in (-int(head_w * 0.22), int(head_w * 0.22)):
        ex = rect.centerx + ex_off
        eye_w = int(head_w * 0.22)
        eye_rect = pygame.Rect(ex - eye_w // 2, eye_y - 5, eye_w, 10)
        pygame.draw.arc(surf, _OUTLINE, eye_rect, math.pi * 0.05, math.pi * 0.95, 3)

    # Intense eyebrows
    brow_y = eye_y - 12
    for bx_off, slant in ((-int(head_w * 0.22), 4), (int(head_w * 0.22), -4)):
        bx = rect.centerx + bx_off
        bw = int(head_w * 0.20)
        pygame.draw.line(surf, _OUTLINE,
                         (bx - bw // 2, brow_y + slant),
                         (bx + bw // 2, brow_y - slant), 3)

    # Mouth (focused grimace)
    mouth_y = head_y + int(head_h * 0.65)
    mouth_w = int(head_w * 0.30)
    pygame.draw.line(surf, _OUTLINE,
                     (rect.centerx - mouth_w // 2, mouth_y),
                     (rect.centerx + mouth_w // 2, mouth_y), 3)

    # Laser beams shooting from the closed eyes (ready-to-fire state)
    for ex_off in (-int(head_w * 0.22), int(head_w * 0.22)):
        ex = rect.centerx + ex_off
        ey = eye_y
        bx_end = rect.left + 8 if ex_off < 0 else rect.right - 8
        pygame.draw.line(surf, (255, 80, 80), (ex, ey), (bx_end, ey - 4), 4)
        pygame.draw.line(surf, (255, 200, 200), (ex, ey), (bx_end, ey - 4), 2)

    # (no font in raw draw helpers — callers with font access add labels below)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

DRAW_REGISTRY: dict[str, Callable[[pygame.Surface, pygame.Rect], None]] = {
    "fireball": draw_fireball,
    "rasengan": draw_rasengan,
    "chidori": draw_chidori,
    "time_freeze": draw_time_freeze,
    "laser_eyes": draw_laser_eyes,
    "kamehameha": draw_kamehameha,
    "space_stretch": draw_space_stretch,
    "reality_tear": draw_reality_tear,
    "frost_nova": draw_frost_nova,
}

# ---------------------------------------------------------------------------
# Per-ability metadata for the manual pages
# ---------------------------------------------------------------------------

# Fire-hint strings are mirrored from effects/hud.py _ABILITY_FIRE_HINT
# so the manual and the HUD always stay in sync.
