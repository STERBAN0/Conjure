# Contributing to Conjure

Thanks for wanting to contribute. Conjure is a small, opinionated codebase
and the goal is to keep it readable while shipping new abilities and
visual upgrades that *feel* good. This guide is short on purpose.

## Quick start

### Windows (PowerShell)

```powershell
git clone https://github.com/STERBAN0/conjure
cd conjure
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/download_model.py     # fetches hand_landmarker.task + face_landmarker.task
python main.py
```

### macOS / Linux

```bash
git clone https://github.com/STERBAN0/conjure
cd conjure
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_model.py     # fetches hand_landmarker.task + face_landmarker.task
python main.py
```

The MediaPipe Tasks models live in `models/`. `hand_landmarker.task` is
required for all hand-driven abilities. `face_landmarker.task` is
required for Laser Eyes; the app degrades gracefully if it's absent.
`scripts/download_model.py` downloads both and is idempotent — re-running
skips files that are already present.

Audio SFX (`audio/sfx/`) are committed to the repo and work immediately.
No generation step needed.

## Project layout (one-screen tour)

```
config.py         all tunables — touch this first
main.py           60 Hz pipeline + input loop
core/             shared dataclasses (HandData, FrameState, AbilityState)
vision/
  camera.py       threaded webcam capture
  hand_tracker.py MediaPipe + One Euro smoothing
  face_tracker.py MediaPipe face landmarker (eye-closed state, cadenced)
gestures/
  engine.py       continuous signals (span, expansion, motion, ...)
  poses.py        discrete pose classifier + temporal hysteresis
  router.py       single-slot ability state machine
effects/
  base.py         Effect base class with ability lifecycle hooks
  utils.py        shared rendering primitives
  *.py            one effect per ability
  projectiles.py  flying projectile field (thrown abilities)
  renderer.py     composes BG/FG layers, gates effects on the router
  hud.py          minimal + debug overlays
audio/            threaded mic + 8-band FFT (optional)
system/
  controls.py     Windows master volume gesture (optional)
  manual.py       in-app hand-sign manual overlay
tests/            pytest unit tests
```

## Adding a new ability — the short version

1. **Add a pose predicate** in `gestures/poses.py`. Single-hand predicates
   take `HandData`; two-hand predicates take a `(left, right)` pair.
   Return `Optional[PoseMatch]` with confidence in 0..1.

   There are two layers in `poses.py`:
   - `_raw_matches(hands)` — pure geometry, stateless. Call this in unit
     tests: hand-craft landmark fixtures and assert on the returned
     confidence directly.
   - `classify(hands, state)` — wraps `_raw_matches` with temporal
     hysteresis (`_PoseHysteresis` per pose). Use this at runtime.
     A pose only activates after its confidence stays above
     `POSE_ENTER_THRESHOLD` for `POSE_ENTER_FRAMES` consecutive frames
     and only deactivates when it falls below the lower
     `POSE_EXIT_THRESHOLD`. Do not bypass the hysteresis layer in
     production code.

2. **Register the ability** in `gestures/router.py::default_abilities()`
   with its pose id, charge time, cooldown, active duration, and
   `release_motion`:
   - `"thrust"` — fires on a fast forward push of one hand
   - `"spread"` — fires when the hands (or hand) fly apart
   - `"throw"` — fires on a fast wrist flick; spawns a flying projectile
   - `"shove"` — fires on a fast forward push of *both* hands simultaneously
   - `"pose_release"` — fires when the pose *disappears* while charge ≥ 1.0
     (useful when the fire trigger is the absence of a sign, as with Laser Eyes)
   - `None` — continuous; the ability stays active while the pose is held

   For throwable abilities (`"throw"`), also set `projectile_kind` to one
   of the registered projectile shape IDs (currently `"rasengan"` or
   `"fireball"`). The `ProjectileField` / `ProjectileSystem` in
   `effects/projectiles.py` picks this up on the `projectile_spawn` hook
   event and draws the flying object from release point to frame edge,
   where it bursts.

   Note on `"pose_release"`: if the pose drops before the charge ring is
   full, the ability is cancelled rather than fired. The pose must be held
   continuously through the full charge.

3. **Add tunables** to `config.py` — `ABILITY_CHARGE_TIME`,
   `ABILITY_COOLDOWN`, `ABILITY_ACTIVE_DURATION`, plus any visual
   constants for the new effect.

4. **Build the effect** in `effects/<name>.py`. Subclass `Effect`, set
   `ability_name = "<name>"`, override `update`, `render`, and any
   lifecycle hooks (`on_enter`, `on_charge`, `on_release`, etc.) you need.

5. **Register the effect** in `effects/renderer.py::default_renderer()`.

6. **Add a manual entry.** Register a draw function in
   `system/manual.py::DRAW_REGISTRY` keyed by your ability name, then add
   a corresponding page to `docs/MANUAL.md` with the hand-sign description
   and any clarifying notes. The in-app manual (M key) pulls from
   `DRAW_REGISTRY`; the Markdown file is the written reference.

7. **Add tests** under `tests/`. At minimum: a positive test that the
   pose is recognised (via `_raw_matches`), a negative test that an
   unrelated pose isn't, and a router test that the charge → release
   transition fires correctly.

If you can describe the new ability in one paragraph and a screenshot,
that's a good PR.

## Code style

- Python 3.10+. Type-annotate all public function signatures.
- Format with `black` (line length 100). Lint with `ruff`.
- Prefer immutable dataclasses where data is read-only.
- Use `logging` rather than `print()` outside of one-shot scripts.
- Pose / router / effect classes should be testable without a webcam.
  All side effects live in the renderer, the camera thread, and `main.py`.

## Tests

```bash
pytest                       # everything
pytest -m unit               # fast tests only
pytest -k chidori            # specific
pytest --cov=. --cov-report=term-missing
```

Aim for 80% coverage on new modules. The pose classifier and the router
are deterministic; please test them with hand-crafted landmark fixtures
(see `tests/conftest.py`). Use `_raw_matches` directly in those fixtures —
it's stateless, so you don't need to feed it N consecutive frames to
satisfy the hysteresis.

## Things that will get a PR rejected (politely)

- Effects that listen directly to raw `signals` instead of going through
  the router. Abilities must be unambiguous.
- Always-on effects. One ability at a time.
- New dependencies for things the standard library or numpy already does.
- Effects that depend on uncommitted assets (shaders, sounds, models)
  without a path to acquire them automatically.
- Hand-rolled smoothing where `OneEuroFilter` or `EMA` would do.
- A new ability without a manual entry in `system/manual.py` and
  `docs/MANUAL.md`.

## Things that are great to PR

- A new ability with the full plumbing (pose, router entry, effect,
  manual entry, tests).
- Visual fidelity passes on existing effects (better lightning, prettier
  Kamehameha beam, smoother space-stretch falloff).
- Performance: anything that buys back frame budget at 1280×720 at 60 fps.
- GLSL backend (`moderngl`) for the heaviest effects, gated behind a flag.
- Sound effects layered on `ability_release` events.
- A pose calibration mode that records the user's canonical pose as a
  template so the recogniser tunes per-user.
- Cross-platform polish (Linux/macOS audio + camera quirks).
- Two-player split-screen.

## Reporting bugs

Please include:

- OS, Python version, webcam model
- Output of `pip freeze | grep -E "mediapipe|opencv|pygame|numpy"`
- A short clip or screenshot if it's a visual issue
- The full stack trace if it's a crash

## License

By contributing you agree your changes are licensed under MIT, the same
as the rest of Conjure.
