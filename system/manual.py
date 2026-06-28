"""In-app ability manual with cartoon hand-sign diagram renderer.

Public surface
--------------
DRAW_REGISTRY : dict[str, Callable[[pygame.Surface, pygame.Rect], None]]
    Maps ability id → draw function (defined in ``manual_draws``,
    re-exported here).  Import this for the exporter.

Manual(width, height)
    Paginated full-screen overlay.
    .is_open   – bool property
    .toggle()  – open / close
    .next_page()
    .prev_page()
    .render(target)  – no-op when closed

PNG-preferred rendering
-----------------------
When ``docs/manual_images/<id>.png`` exists alongside the repo root the
Manual will load it (cached) and scale-to-fit the diagram area instead of
calling the vector draw function.  If the file is absent or fails to load,
the vector draw function is used as the fallback — no crash.
"""

from __future__ import annotations

import logging
import pathlib

import pygame

# The vector hand-sign diagrams + DRAW_REGISTRY live in manual_draws; they
# are re-exported here so `from system.manual import DRAW_REGISTRY` still works.
from system.manual_draws import DRAW_REGISTRY

log = logging.getLogger("conjure.manual")

# Absolute path to the repo root so PNG lookup is cwd-independent.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PNG_DIR = _REPO_ROOT / "docs" / "manual_images"

# ---------------------------------------------------------------------------
# PNG cache  (populated lazily on first render)
# ---------------------------------------------------------------------------

# Maps ability_id → loaded pygame.Surface (or None when load failed)
_PNG_CACHE: dict[str, pygame.Surface | None] = {}


def _load_png(ability_id: str) -> pygame.Surface | None:
    """Load docs/manual_images/<id>.png once; return None on any failure."""
    if ability_id in _PNG_CACHE:
        return _PNG_CACHE[ability_id]

    path = _PNG_DIR / f"{ability_id}.png"
    if not path.exists():
        _PNG_CACHE[ability_id] = None
        return None

    try:
        surf = pygame.image.load(str(path)).convert_alpha()
        _PNG_CACHE[ability_id] = surf
        log.debug("Loaded manual PNG: %s", path)
        return surf
    except Exception:
        log.warning("Failed to load manual PNG: %s", path, exc_info=True)
        _PNG_CACHE[ability_id] = None
        return None


def _blit_png_scaled(
    target: pygame.Surface,
    png_surf: pygame.Surface,
    dest_rect: pygame.Rect,
) -> None:
    """Scale png_surf to fit dest_rect (preserving aspect ratio) and blit."""
    pw, ph = png_surf.get_size()
    scale = min(dest_rect.width / pw, dest_rect.height / ph)
    new_w = int(pw * scale)
    new_h = int(ph * scale)
    scaled = pygame.transform.smoothscale(png_surf, (new_w, new_h))
    blit_x = dest_rect.left + (dest_rect.width - new_w) // 2
    blit_y = dest_rect.top + (dest_rect.height - new_h) // 2
    target.blit(scaled, (blit_x, blit_y))


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """Greedy word-wrap so a long description spans several lines.

    Returns the text split into lines, each no wider than ``max_width`` px.
    A single word longer than ``max_width`` is kept on its own line rather
    than looping forever, so the function always terminates.
    """
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = word if not current else f"{current} {word}"
        if not current or font.size(trial)[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ---------------------------------------------------------------------------
# Cartoon draw helpers
# ---------------------------------------------------------------------------

_PAGES: list[dict[str, str]] = [
    {
        "id": "fireball",
        "name": "Fireball",
        "sign": "One hand — only the INDEX finger up",
        "fire": "Charge once, then FLICK THE FINGER to shoot (unlimited)",
    },
    {
        "id": "rasengan",
        "name": "Rasengan",
        "sign": "Two hands stacked — lower palm UP, top hand stirs in a circle",
        "fire": "FLICK to throw (drifts slowly the way you shove)",
    },
    {
        "id": "chidori",
        "name": "Chidori",
        "sign": "One hand — two-finger V (index + middle up)",
        "fire": "HOLD (active while held; ends when you drop the sign)",
    },
    {
        "id": "time_freeze",
        "name": "Time Freeze",
        "sign": "One closed FIST, palm facing the camera",
        "fire": "HOLD ~2.5 s (screen slows to a freeze, then shatters)",
    },
    {
        "id": "laser_eyes",
        "name": "Laser Eyes",
        "sign": "Close BOTH eyes (face sign — blinks don't count)",
        "fire": "Eyes shut ~1 s to charge, OPEN to fire; twin beams converge to one "
                "point you aim with your HEAD and EYES — starts on your face, reaches "
                "anywhere, draws a trail you can write with; eyes shut to stop; 'R' clears",
    },
    {
        "id": "kamehameha",
        "name": "Kamehameha",
        "sign": "Two open hands raised together, palms to camera, fingertips & thumbs touching in a triangle",
        "fire": "PUSH toward the screen to fire — it engulfs the screen in blue",
    },
    {
        "id": "space_stretch",
        "name": "Space Stretch",
        "sign": "Two OPEN palms facing each other, then pull them apart",
        "fire": "No charge — the warp just happens and grows as you pull apart",
    },
    {
        "id": "reality_tear",
        "name": "Reality Tear",
        "sign": "Two FISTS bumped together, then pull them apart",
        "fire": "Bump fists to charge, then RIP APART to tear reality",
    },
    {
        "id": "frost_nova",
        "name": "Frost Nova",
        "sign": "Wrists CROSSED in an X",
        "fire": "SPREAD/OPEN HANDS TO BURST",
    },
]


# ---------------------------------------------------------------------------
# Manual overlay class
# ---------------------------------------------------------------------------

class Manual:
    """Paginated full-screen in-app ability manual.

    Usage (wired by the orchestrator)::

        manual = Manual(width, height)
        # in the key handler:
        if event.key == pygame.K_m:
            manual.toggle()
        if event.key == pygame.K_RIGHT:
            manual.next_page()
        if event.key == pygame.K_LEFT:
            manual.prev_page()
        # in the render loop (after other HUD):
        manual.render(screen)
    """

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._open = False
        self._page = 0

        self.font_title = pygame.font.SysFont("consolas", 48, bold=True)
        self.font_head = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_body = pygame.font.SysFont("consolas", 17)
        self.font_footer = pygame.font.SysFont("consolas", 14)

    # -- Public API ----------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._open

    def toggle(self) -> None:
        self._open = not self._open
        # Always (re)open on the first ability (Fireball, 1/9) rather than
        # wherever the user last paged to before closing.
        if self._open:
            self._page = 0
        log.debug("Manual %s", "opened" if self._open else "closed")

    def next_page(self) -> None:
        if self._open:
            self._page = (self._page + 1) % len(_PAGES)

    def prev_page(self) -> None:
        if self._open:
            self._page = (self._page - 1) % len(_PAGES)

    def render(self, target: pygame.Surface) -> None:
        if not self._open:
            return

        w, h = self._width, self._height
        page = _PAGES[self._page]

        # Semi-transparent backdrop
        backdrop = pygame.Surface((w, h), pygame.SRCALPHA)
        backdrop.fill((8, 10, 24, 220))
        target.blit(backdrop, (0, 0))

        # Title (ability name)
        title_surf = self.font_title.render(page["name"].upper(), True, (220, 240, 255))
        target.blit(title_surf, (w // 2 - title_surf.get_width() // 2, 36))

        # Diagram area — centred, generous
        diag_size = min(w // 2, int(h * 0.45))
        diag_rect = pygame.Rect(
            w // 2 - diag_size // 2,
            int(h * 0.15),
            diag_size,
            diag_size,
        )

        # Diagram background panel
        panel = pygame.Surface((diag_size, diag_size), pygame.SRCALPHA)
        panel.fill((20, 28, 50, 180))
        pygame.draw.rect(panel, (60, 80, 130), panel.get_rect(), 2, border_radius=16)
        target.blit(panel, diag_rect.topleft)

        # PNG-preferred rendering: use the image if present, else vector fallback.
        png_surf = _load_png(page["id"])
        if png_surf is not None:
            _blit_png_scaled(target, png_surf, diag_rect)
        else:
            draw_fn = DRAW_REGISTRY.get(page["id"])
            if draw_fn is not None:
                diagram_surf = pygame.Surface((diag_size, diag_size), pygame.SRCALPHA)
                draw_fn(diagram_surf, diagram_surf.get_rect())
                target.blit(diagram_surf, diag_rect.topleft)

        # Text block below the diagram. Long descriptions (e.g. the Laser Eyes
        # fire instructions) WRAP within the screen width instead of running
        # off both edges — each wrapped line is centred under the previous one.
        text_y = diag_rect.bottom + 24
        line_h = 32
        body_pitch = 24
        max_text_w = int(w * 0.86)

        def _label(text: str, color: tuple[int, int, int]) -> None:
            nonlocal text_y
            s = self.font_head.render(text, True, color)
            target.blit(s, (w // 2 - s.get_width() // 2, text_y))
            text_y += line_h

        def _body(text: str, color: tuple[int, int, int] = (180, 200, 230)) -> None:
            nonlocal text_y
            for line in _wrap_text(text, self.font_body, max_text_w):
                s = self.font_body.render(line, True, color)
                target.blit(s, (w // 2 - s.get_width() // 2, text_y))
                text_y += body_pitch
            text_y += 4

        _label("Sign:", (140, 200, 255))
        _body(page["sign"])
        text_y += 4
        _label("Fire:", (255, 230, 80))
        _body(page["fire"], (255, 240, 160))

        # Page indicator
        page_txt = f"{self._page + 1} / {len(_PAGES)}"
        pg_surf = self.font_footer.render(page_txt, True, (120, 140, 170))
        target.blit(pg_surf, (w // 2 - pg_surf.get_width() // 2, h - 48))

        # Footer nav hint — ESC to close (ESC now closes; M still toggles)
        footer = "←/→  pages      ESC  close"
        ft_surf = self.font_footer.render(footer, True, (100, 120, 150))
        target.blit(ft_surf, (w // 2 - ft_surf.get_width() // 2, h - 28))
