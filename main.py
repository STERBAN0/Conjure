"""Conjure — gesture-driven anime ability system.

Pipeline (per frame, ~60 Hz):

    Camera (thread)         -> latest BGR frame
    HandTracker             -> structured, mirror-corrected, smoothed hands
    FaceTracker (cadenced)  -> eyes-closed state for Laser Eyes
    GestureEngine           -> continuous expressive signals
    PoseRecognizer          -> discrete pose matches with confidences
    AbilityRouter           -> single-slot ability state machine
    AudioAnalyzer (thread)  -> live broadband level + 8 FFT bands
    SystemControls          -> optional master-volume gesture (Windows)
    EffectsRenderer         -> only renders effects whose ability is in flight
    HUD                     -> ability label, charge ring, optional debug
    Manual                  -> in-app hand-sign manual overlay
    pygame                  -> blit, present, repeat

Controls:
    Q         - quit
    ESC       - close the manual (when open)
    H         - toggle minimal HUD
    D         - toggle debug overlay
    M         - toggle the hand-sign manual (←/→ to page)
    L         - toggle Laser Eyes / face tracking
    R         - clear the laser-eyes molten "drawing" from the screen
    S         - save a screenshot to ./screenshots
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pygame

import config
from audio.analyzer import AudioAnalyzer
from audio.sounds import SoundManager
from core.hooks import HookBus
from core.state import PHASE_ACTIVE, FaceData, FrameState
from effects.hud import HUD
from effects.renderer import default_renderer
from gestures.engine import GestureEngine
from gestures.poses import PoseRecognizer
from gestures.router import AbilityRouter
from system.controls import SystemControls
from system.manual import Manual
from vision.camera import Camera
from vision.face_tracker import FaceTracker
from vision.hand_tracker import HandTracker

log = logging.getLogger("conjure")

# Minimum frame delta (seconds) to avoid division-by-zero on the first frame
# or after a system stall; keeps dt physics-safe at all times.
_MIN_DT_SECONDS = 1e-3


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    n_pass, n_fail = pygame.init()
    if n_fail > 0:
        log.warning("pygame.init(): %d subsystem(s) failed to initialise", n_fail)
    pygame.display.set_caption("Conjure")
    # RESIZABLE enables the title-bar MAXIMIZE button and edge-dragging (Windows
    # greys those out on a fixed-size window). SCALED keeps the render target at
    # the logical WINDOW_W x WINDOW_H and lets pygame stretch it to whatever the
    # window becomes — the surface is preserved and letterboxed to keep aspect
    # ratio, so resizing/maximizing needs no coordinate changes anywhere else.
    screen = pygame.display.set_mode(
        (config.WINDOW_W, config.WINDOW_H),
        pygame.DOUBLEBUF | pygame.SCALED | pygame.RESIZABLE,
    )
    clock = pygame.time.Clock()

    # Initialise the mixer before constructing SoundManager.
    # Failure is non-fatal: SoundManager degrades gracefully if the mixer is
    # not initialised (headless environments, missing audio drivers, etc.).
    try:
        pygame.mixer.init(
            frequency=config.SOUND_MIXER_FREQUENCY,
            size=-16,
            channels=2,
        )
        pygame.mixer.set_num_channels(config.SOUND_MIXER_CHANNELS)
    except Exception as _mixer_exc:  # noqa: BLE001
        log.warning("pygame.mixer init failed (%s) — SFX disabled", _mixer_exc)

    hooks = HookBus()
    _sound_manager = SoundManager(hooks)  # subscribes to hooks; no shutdown needed
    camera = Camera()
    tracker = HandTracker()
    face_tracker = _init_face_tracker()
    engine = GestureEngine()
    poses = PoseRecognizer()
    router = AbilityRouter(hooks)
    audio = AudioAnalyzer()
    system_ctl = SystemControls()
    renderer = default_renderer(config.WINDOW_W, config.WINDOW_H, hooks)
    hud = HUD(config.WINDOW_W, config.WINDOW_H)
    manual = Manual(config.WINDOW_W, config.WINDOW_H)

    show_hud_minimal = True
    show_hud_debug = False
    last_t = time.monotonic()
    frame_count = 0
    last_face: FaceData | None = None
    prev_time_freeze = False
    face_enabled = face_tracker is not None

    log.info(
        "Conjure started — Q quit, H HUD, D debug, M manual (ESC closes), "
        "L laser eyes, R clear drawing, S screenshot"
    )
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # ESC closes the manual when open; it no longer quits.
                        if manual.is_open:
                            manual.toggle()
                    elif event.key == pygame.K_q:
                        return 0
                    elif event.key == pygame.K_h:
                        show_hud_minimal = not show_hud_minimal
                    elif event.key == pygame.K_d:
                        show_hud_debug = not show_hud_debug
                    elif event.key == pygame.K_m:
                        manual.toggle()
                    elif event.key == pygame.K_l:
                        face_enabled = not face_enabled
                        log.info(
                            "Laser Eyes / face tracking %s",
                            "ON" if face_enabled else "OFF",
                        )
                    elif event.key == pygame.K_r:
                        renderer.clear_drawings()
                        log.info("Cleared laser-eyes drawing")
                    elif event.key == pygame.K_s:
                        _save_screenshot(screen)
                    if manual.is_open:
                        if event.key in (pygame.K_RIGHT, pygame.K_PAGEDOWN):
                            manual.next_page()
                        if event.key in (pygame.K_LEFT, pygame.K_PAGEUP):
                            manual.prev_page()

            frame_bgr = camera.read()
            if frame_bgr is None:
                clock.tick(config.TARGET_FPS)
                continue

            now = time.monotonic()
            raw_dt = max(_MIN_DT_SECONDS, now - last_t)
            last_t = now
            # Time-freeze slow-mo: scale the simulation dt while the ability is
            # live (one frame of latency is imperceptible and keeps the loop
            # itself running at full rate). Camera frames still update; the
            # desaturated/tinted frame sells the "frozen" look.
            dt = raw_dt * config.TIME_FREEZE_TIME_SCALE if prev_time_freeze else raw_dt

            hands = tracker.process(frame_bgr, now)

            # Face detection runs at a reduced cadence so it never bottlenecks
            # the 60 Hz loop. The last result is reused on intervening frames.
            frame_count += 1
            if not face_enabled:
                last_face = None
            elif (
                face_tracker is not None
                and frame_count % config.FACE_DETECT_EVERY_N_FRAMES == 0
            ):
                last_face = face_tracker.process(frame_bgr, now)

            frame = FrameState(
                frame_bgr=frame_bgr,
                timestamp=now,
                dt=dt,
                hands=hands,
                face=last_face,
            )
            signals = engine.update(frame)

            level, bands = audio.get()
            signals.audio_level = level
            signals.audio_bands = bands

            matches = poses.classify(frame)
            ability = router.update(frame, signals, matches)
            # Slow-mo ONLY during the true freeze (PHASE_ACTIVE). It must NOT
            # bleed into the release/cooldown, or the 2s shatter delay would run
            # at 0.25x (8s wall-time) and the grey tint would appear to "stick".
            prev_time_freeze = (
                ability.name == "time_freeze" and ability.phase == PHASE_ACTIVE
            )

            system_ctl.update(frame, signals, ability)

            renderer.update_and_render(frame, signals, ability, screen)

            if show_hud_minimal:
                hud.render_minimal(screen, ability, matches, face_enabled)
            if show_hud_debug:
                hud.render_debug(
                    screen, frame, signals, ability, matches, fps=clock.get_fps(),
                )
            if manual.is_open:
                manual.render(screen)

            pygame.display.flip()
            clock.tick(config.TARGET_FPS)

    except KeyboardInterrupt:
        return 0
    finally:
        camera.close()
        tracker.close()
        if face_tracker is not None:
            face_tracker.close()
        audio.close()
        pygame.quit()


def _init_face_tracker() -> FaceTracker | None:
    """Construct the face tracker, degrading gracefully if it's unavailable.

    Laser Eyes is optional: if the face model hasn't been downloaded (or
    MediaPipe can't initialise it), we log a hint and run without face
    detection rather than crashing the whole app.
    """
    if not config.FACE_ENABLED:
        return None
    try:
        return FaceTracker()
    except FileNotFoundError as e:
        log.warning("Face tracking disabled (run scripts/download_model.py): %s", e)
        return None
    except Exception as e:  # pragma: no cover — defensive
        log.warning("Face tracking unavailable: %r", e)
        return None


def _save_screenshot(surface: pygame.Surface) -> None:
    out_dir = Path(__file__).resolve().parent / "screenshots"
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"conjure_{datetime.now():%Y%m%d_%H%M%S}.png"
    pygame.image.save(surface, str(fname))
    log.info("saved screenshot %s", fname)


if __name__ == "__main__":
    sys.exit(main())
