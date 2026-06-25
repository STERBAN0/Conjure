# Conjure — Claude Code Guide

Conjure is an offline webcam app that maps hand and face gestures to real-time
anime VFX using MediaPipe tracking and pygame rendering.

## Running the app

```bash
python scripts/download_model.py   # one-time: fetches hand + face MediaPipe models (~14 MB)
python main.py                     # starts the 60 Hz pipeline
```

Audio SFX are committed to `audio/sfx/` and work immediately. No generation step needed.

## Architecture

Data flows in one direction:

```
camera (vision/camera.py)
  → hand_tracker + face_tracker (vision/)
  → GestureEngine — raw hands to continuous signals (gestures/engine.py)
  → PoseRecognizer — discrete poses + temporal hysteresis (gestures/poses.py)
  → AbilityRouter — single-slot state machine: charging/active/cooldown (gestures/router.py)
  → EffectsRenderer — draws only the active effect onto the pygame surface (effects/renderer.py)
  → pygame display
```

Key files:

| File | Role |
|---|---|
| `config.py` | All tunables (thresholds, timings, colours, window size) |
| `core/state.py` | Shared dataclasses: `HandData`, `FrameState`, `GestureSignals`, `AbilityState` |
| `core/hooks.py` | Synchronous pub/sub bus used by effects and audio |
| `gestures/poses.py` | Pose predicates — `_raw_matches` (stateless geometry) + `classify` (with hysteresis) |
| `gestures/router.py` | `default_abilities()` — the roster of registered abilities |
| `effects/base.py` | `Effect` base class with lifecycle hooks: `on_enter`, `on_charge`, `on_release`, `on_exit` |
| `effects/renderer.py` | `default_renderer()` — registers all effect instances |
| `system/manual.py` | `DRAW_REGISTRY` — in-app hand-sign diagrams (M key) |
| `audio/sounds.py` | Hook-driven SFX playback |
| `system/controls.py` | Windows-only volume gesture (optional, gracefully absent on other platforms) |

## Adding a new ability

Follow this sequence — every step is required:

1. **Pose predicate** in `gestures/poses.py`. Return `Optional[PoseMatch]` from
   `_raw_matches`. Register in `classify` with `_PoseHysteresis`.

2. **Router entry** in `gestures/router.py::default_abilities()`. Set pose id,
   charge time, cooldown, active duration, and `release_motion` (`"thrust"`,
   `"spread"`, `"throw"`, `"shove"`, `"pose_release"`, or `None` for continuous).
   Throwable abilities also need `projectile_kind` (`"rasengan"` or `"fireball"`).

3. **Tunables** in `config.py` — charge time, cooldown, colours, any effect-specific constants.

4. **Effect module** at `effects/<name>.py`. Subclass `Effect`, set `ability_name`,
   override `update`, `render`, and the lifecycle hooks you need.

5. **Register** the effect instance in `effects/renderer.py::default_renderer()`.

6. **Manual entry** — add a draw function to `system/manual.py::DRAW_REGISTRY`
   and a page to `docs/MANUAL.md`.

7. **Tests** in `tests/` — at minimum: positive pose recognition, negative (wrong
   pose rejected), and a router charge → release transition test.
   Call `_raw_matches` directly in pose tests; it's stateless and needs no
   consecutive-frame setup.

## Tests and lint

```bash
python -m pytest -q           # ~137 tests, all webcam-free, ~1 s
python -m pytest -k poses     # just the pose classifier
python -m pytest --cov=.      # with coverage report
ruff check .                  # lint
```

The pose classifier and router are fully deterministic. Synthetic landmark
fixtures live in `tests/conftest.py`. Effects are visual — run `main.py` to
eyeball them.

## Key constraints

- One ability active at a time. Effects must not listen to raw signals directly;
  they go through the router.
- No always-on effects.
- No new dependencies for things numpy or the standard library already handle.
- New abilities need a manual entry (`system/manual.py` + `docs/MANUAL.md`) or
  the PR will be asked to add one.
- The face pipeline runs at a reduced cadence (`FACE_DETECT_EVERY_N_FRAMES` in
  `config.py`) to keep the main loop at 60 Hz. Do not call face detection every
  frame.
