# Contributing to Aether

Thanks for wanting to contribute. Aether is a small, opinionated codebase
and the goal is to keep it readable while shipping new abilities and
visual upgrades that *feel* good. This guide is short on purpose.

## Quick start

```bash
git clone https://github.com/<you>/aether
cd aether
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
pip install -e ".[dev]"
python main.py
```

The MediaPipe Tasks model lives in `models/hand_landmarker.task`. The
README's setup section has a single-line download command.

## Project layout (one-screen tour)

```
config.py         all tunables — touch this first
main.py           60 Hz pipeline + input loop
core/             shared dataclasses (HandData, FrameState, AbilityState)
vision/           camera thread + MediaPipe wrapper + smoothing
gestures/
  engine.py       continuous signals (span, expansion, motion, ...)
  poses.py        discrete pose classifier (Chidori seal, Kamehameha cup, ...)
  router.py       single-slot ability state machine
effects/
  base.py         Effect base class with ability lifecycle hooks
  utils.py        shared rendering primitives
  *.py            one effect per ability
  renderer.py     composes BG/FG layers, gates effects on the router
  hud.py          minimal + debug overlays
audio/            threaded mic + 8-band FFT (optional)
system/           Windows master volume gesture (optional)
tests/            pytest unit tests
```

## Adding a new ability — the short version

1. **Add a pose predicate** in `gestures/poses.py`. Single-hand predicates
   take `HandData`; two-hand predicates take a `(left, right)` pair.
   Return `Optional[PoseMatch]` with confidence in 0..1.
2. **Register the ability** in `gestures/router.py::default_abilities()`
   with its pose id, charge time, cooldown, active duration, and release
   motion (`"thrust"`, `"spread"`, or `None` for continuous).
3. **Add tunables** to `config.py` — `ABILITY_CHARGE_TIME`,
   `ABILITY_COOLDOWN`, `ABILITY_ACTIVE_DURATION`, plus any visual
   constants for the new effect.
4. **Build the effect** in `effects/<name>.py`. Subclass `Effect`, set
   `ability_name = "<name>"`, override `update`, `render`, and any
   lifecycle hooks (`on_enter`, `on_charge`, `on_release`, etc.) you need.
5. **Register the effect** in `effects/renderer.py::default_renderer()`.
6. **Add tests** under `tests/`. At minimum: a positive test that the
   pose is recognised, a negative test that an unrelated pose isn't, and
   a router test that the charge → release transition fires correctly.

If you can describe the new ability in one paragraph and a screenshot,
that's a good PR.

## Code style

- Python 3.10+. Type-annotate all public function signatures.
- Format with `black` (line length 100). Lint with `ruff`.
- Prefer immutable dataclasses where data is read-only.
- Use `logging` rather than `print()` outside of one-shot scripts.
- Pose / router / effect classes should be **testable without a webcam**.
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
(see `tests/conftest.py`).

## Things that will get a PR rejected (politely)

- Effects that listen directly to raw `signals` instead of going through
  the router. Abilities must be unambiguous.
- Always-on effects. One ability at a time.
- New dependencies for things the standard library or numpy already does.
- Effects that depend on uncommitted assets (shaders, sounds, models)
  without a path to acquire them automatically.
- Hand-rolled smoothing where `OneEuroFilter` or `EMA` would do.

## Things that are great to PR

- A new ability with the full plumbing (pose, router entry, effect, tests).
- Visual fidelity passes on existing effects (better lightning, prettier
  Kamehameha beam, smoother space-stretch falloff).
- Performance: anything that buys back frame budget at 1280×720 at 60 fps.
- GLSL backend (`moderngl`) for the heaviest effects, gated behind a flag.
- Sound effects layered on `ability_release` events.
- A pose calibration mode that records the user's canonical pose as a
  template so the recogniser tunes per-user.
- Cross-platform polish (Linux/macOS audio + camera quirks).

## Reporting bugs

Please include:

- OS, Python version, webcam model
- Output of `pip freeze | grep -E "mediapipe|opencv|pygame|numpy"`
- A short clip or screenshot if it's a visual issue
- The full stack trace if it's a crash

## License

By contributing you agree your changes are licensed under MIT, the same
as the rest of Aether.
