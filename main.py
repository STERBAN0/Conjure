"""Aether — gesture-driven anime ability system.

Pipeline (per frame, ~60 Hz):

    Camera (thread)         -> latest BGR frame
    HandTracker             -> structured, mirror-corrected, smoothed hands
    GestureEngine           -> continuous expressive signals
    PoseRecognizer          -> discrete pose matches with confidences
    AbilityRouter           -> single-slot ability state machine
    AudioAnalyzer (thread)  -> live broadband level + 8 FFT bands
    SystemControls          -> optional master-volume gesture (Windows)
    EffectsRenderer         -> only renders effects whose ability is in flight
    HUD                     -> ability label, charge ring, optional debug
    pygame                  -> blit, present, repeat

Controls:
    ESC / Q   - quit
    H         - toggle minimal HUD
    D         - toggle debug overlay
    S         - save a screenshot to ./screenshots
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pygame

import config
from audio.analyzer import AudioAnalyzer
from core.hooks import HookBus
from core.state import FrameState
from effects.hud import HUD
from effects.renderer import default_renderer
from gestures.engine import GestureEngine
from gestures.poses import PoseRecognizer
from gestures.router import AbilityRouter
from system.controls import SystemControls
from vision.camera import Camera
from vision.hand_tracker import HandTracker

log = logging.getLogger("aether")


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    pygame.init()
    pygame.display.set_caption("Aether")
    screen = pygame.display.set_mode(
        (config.WINDOW_W, config.WINDOW_H),
        pygame.DOUBLEBUF,
    )
    clock = pygame.time.Clock()

    hooks = HookBus()
    camera = Camera()
    tracker = HandTracker()
    engine = GestureEngine()
    poses = PoseRecognizer()
    router = AbilityRouter(hooks)
    audio = AudioAnalyzer()
    system_ctl = SystemControls()
    renderer = default_renderer(config.WINDOW_W, config.WINDOW_H, hooks)
    hud = HUD(config.WINDOW_W, config.WINDOW_H)

    show_hud_minimal = True
    show_hud_debug = False
    last_t = time.monotonic()

    log.info("Aether started — ESC/Q quit, H minimal HUD, D debug, S screenshot")
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        return 0
                    if event.key == pygame.K_h:
                        show_hud_minimal = not show_hud_minimal
                    if event.key == pygame.K_d:
                        show_hud_debug = not show_hud_debug
                    if event.key == pygame.K_s:
                        _save_screenshot(screen)

            frame_bgr = camera.read()
            if frame_bgr is None:
                clock.tick(config.TARGET_FPS)
                continue

            now = time.monotonic()
            dt = max(1e-3, now - last_t)
            last_t = now

            hands = tracker.process(frame_bgr, now)
            frame = FrameState(
                frame_bgr=frame_bgr, timestamp=now, dt=dt, hands=hands,
            )
            signals = engine.update(frame)

            level, bands = audio.get()
            signals.audio_level = level
            signals.audio_bands = bands

            matches = poses.classify(frame)
            ability = router.update(frame, signals, matches)

            system_ctl.update(frame, signals, ability)

            renderer.update_and_render(frame, signals, ability, screen)

            if show_hud_minimal:
                hud.render_minimal(screen, ability, matches)
            if show_hud_debug:
                hud.render_debug(
                    screen, frame, signals, ability, matches, fps=clock.get_fps(),
                )

            pygame.display.flip()
            clock.tick(config.TARGET_FPS)

    except KeyboardInterrupt:
        return 0
    finally:
        camera.close()
        tracker.close()
        audio.close()
        pygame.quit()


def _save_screenshot(surface: pygame.Surface) -> None:
    out_dir = Path("screenshots")
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"aether_{datetime.now():%Y%m%d_%H%M%S}.png"
    pygame.image.save(surface, str(fname))
    log.info("saved screenshot %s", fname)


if __name__ == "__main__":
    sys.exit(main())
