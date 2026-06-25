"""Headless exporter — renders each ability's cartoon diagram to a PNG.

Usage::

    python scripts/export_manual.py           # skip abilities whose PNG already exists
    python scripts/export_manual.py --force   # overwrite all PNGs unconditionally

Outputs one PNG per ability into ``docs/manual_images/<id>.png``.
Imports the draw registry from ``system/manual.py`` so diagrams
are never duplicated.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys

# Must be set before pygame.init() so no display is required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402  (import after env vars)

# Ensure the project root is on sys.path when the script is run from any cwd.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from system.manual import DRAW_REGISTRY  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export_manual")

_OUT_DIR = _PROJECT_ROOT / "docs" / "manual_images"
_DIAGRAM_SIZE = 400  # pixels — square


def _export_all(force: bool = False) -> None:
    pygame.init()
    # A minimal display surface is required for font rendering even in dummy mode.
    pygame.display.set_mode((1, 1))

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for ability_id, draw_fn in DRAW_REGISTRY.items():
        out_path = _OUT_DIR / f"{ability_id}.png"
        if out_path.exists() and not force:
            log.info("skipped %s (exists)", out_path.relative_to(_PROJECT_ROOT))
            skipped += 1
            continue

        surf = pygame.Surface((_DIAGRAM_SIZE, _DIAGRAM_SIZE), pygame.SRCALPHA)
        surf.fill((20, 28, 50, 255))  # dark panel background

        draw_fn(surf, surf.get_rect())

        pygame.image.save(surf, str(out_path))
        log.info("Saved %s", out_path.relative_to(_PROJECT_ROOT))
        written += 1

    pygame.quit()
    log.info(
        "Done — %d written, %d skipped (pass --force to overwrite all)",
        written,
        skipped,
    )


if __name__ == "__main__":
    _export_all(force="--force" in sys.argv)
