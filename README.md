# Aether

> Real-time gesture-driven anime ability effects. Make a Sasuke seal and
> watch lightning crackle from your palm. Cup your hands together for a
> Kamehameha. Pull space apart between your hands like a stretched
> rubber band. All from a webcam, no controllers, no marker setup.

Aether watches your hands through a regular webcam, classifies what
*pose* you're holding, and plays the matching anime ability with
charging, release motion, and cooldown. Only one ability is active at a
time, so the input is always unambiguous — the system feels like a
fighting-game move list rather than a noise of overlapping VFX.

## Demo

![Aether demo](docs/demo.gif)

See [`docs/DEMO.md`](docs/DEMO.md) for capture instructions and the demo GIF generation script.

```
                          ┌─────────────┐
        webcam  ───────►  │ HandTracker │  21 landmarks/hand, smoothed
                          └──────┬──────┘
                                 ▼
                         ┌──────────────┐
                         │GestureEngine │  continuous signals
                         └──────┬───────┘    (span, expansion, motion, ...)
                                ▼
                     ┌──────────────────┐
                     │  PoseRecognizer  │  discrete poses + confidences
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐  charging → active → cooldown
                     │  AbilityRouter   │  exactly one ability at a time
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │ EffectsRenderer  │  only the active effect runs
                     └────────┬─────────┘
                              ▼
                            pygame
```

## Abilities

| Ability         | Pose                                                                  | Release                  | Visual                                                  |
|-----------------|------------------------------------------------------------------------|--------------------------|---------------------------------------------------------|
| **Chidori**     | Sasuke seal — index + middle extended, others folded                  | Forward thrust           | Cyan-white lightning arcs branching from the palm       |
| **Kamehameha**  | Both hands cupped, palms facing, fingertips touching                  | Hands shoot apart        | Charging sphere → wide cylindrical beam                 |
| **Rasengan**    | Open palm + fist hovering above it                                    | Forward thrust           | Spinning blue sphere with a defined orbital shell       |
| **Space stretch** | Both hands open, spread, palms facing each other                    | (continuous while held)  | Camera frame sheared along the hand-to-hand axis       |
| **Reality tear** | Both hands clawed, held apart                                        | (continuous while held)  | Jagged glowing fracture between the hands               |

## Quick start

### Windows (PowerShell)

```powershell
git clone https://github.com/<you>/aether
cd aether
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe main.py
```

### macOS / Linux

```bash
git clone https://github.com/<you>/aether
cd aether
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/download_model.py
python main.py
```

The model is `~7.5 MB`. The download script is idempotent — re-running
it does nothing if the file is already there.

### Tested with

- Python 3.13.1 (3.10 – 3.13 supported)
- `mediapipe` 0.10.35 (Tasks API; `mp.solutions` is gone in newer wheels)
- `pygame` 2.6.1, `opencv-python` 4.13, `numpy` 2.4
- `pycaw` 20251023 (Windows volume gesture, optional)

## Controls

| Key              | What it does                              |
|------------------|-------------------------------------------|
| `ESC` / `Q`      | Quit                                       |
| `H`              | Toggle the minimal HUD (ability label)    |
| `D`              | Toggle the debug overlay (signals + poses)|
| `S`              | Save a screenshot to `./screenshots/`     |

## How it feels

Every ability has the same loop:

1. **Make the pose.** A confidence indicator appears around the hand.
2. **Hold to charge.** A ring fills around the active hand. Charge time
   varies — Chidori is fast, Kamehameha is dramatic.
3. **Release.** Some abilities (Chidori, Rasengan) fire on a forward
   thrust; some (Kamehameha) on hands shooting apart; continuous ones
   (Space stretch, Reality tear) just stay live while you
   hold the pose.
4. **Cooldown.** Brief pause where you can't immediately start another
   ability — keeps the impact of the release intact.

If you don't see anything happening, press `D` to bring up the debug
overlay. It shows recognised poses with their confidences in real time,
which makes it obvious whether the pose is the problem or the trigger
is. Most "why isn't this working?" issues are the user holding the pose
slightly off the canonical shape — the predicates are tolerant but not
infinite.

## Project layout

```
aether/
├── config.py                     all tunables in one place
├── main.py                       60 Hz pipeline + pygame loop
├── core/
│   ├── state.py                  HandData, FrameState, GestureSignals, AbilityState
│   └── hooks.py                  tiny synchronous pub/sub bus
├── vision/
│   ├── camera.py                 threaded webcam capture
│   └── hand_tracker.py           MediaPipe + One Euro smoothing + mirror fix
├── gestures/
│   ├── smoothing.py              OneEuroFilter, EMA
│   ├── engine.py                 raw hands -> continuous signals
│   ├── poses.py                  discrete pose classifier
│   └── router.py                 single-slot ability state machine
├── effects/
│   ├── base.py                   Effect base class
│   ├── utils.py                  shared draw primitives
│   ├── chidori.py                lightning blade
│   ├── kamehameha.py             sphere + beam
│   ├── rasengan.py               spinning sphere
│   ├── space_stretch.py          elastic membrane (BG warp)
│   ├── reality_tear.py           jagged fracture
│   ├── renderer.py               composes everything
│   └── hud.py                    minimal + debug overlays
├── audio/analyzer.py             threaded mic + 8-band FFT (optional)
├── system/controls.py            Windows volume gesture (optional)
├── models/                       hand_landmarker.task (downloaded)
├── scripts/download_model.py     cross-platform model fetch
└── tests/                        pytest unit tests
```

## How the pipeline stays smooth

Two places to look:

- **`vision/hand_tracker.py`** smooths every landmark with a One Euro
  filter (cutoff frequency adapts to the signal's own velocity, so
  things stay still when still and responsive when fast). The handedness
  fix that makes Left/Right correct under a mirrored selfie view also
  lives here.
- **`gestures/engine.py`** smooths each derived signal again. Two-stage
  smoothing buys you "stable when not moving" *and* "snappy when you
  are" — neither alone is enough.

If the hands jitter: raise `ONE_EURO_BETA` in `config.py` slightly. If
they feel laggy on fast moves: raise it more. If the volume gesture
keeps engaging accidentally: raise `VOLUME_GESTURE_STILLNESS`.

## Adding your own ability

The full guide is in [`CONTRIBUTING.md`](CONTRIBUTING.md), but the short
version is:

1. Add a predicate in `gestures/poses.py` returning `Optional[PoseMatch]`.
2. Wire it into the router in `gestures/router.py::default_abilities()`.
3. Tunables (charge time, cooldown, colours) go in `config.py`.
4. Build the effect in `effects/<name>.py`, subclassing `Effect` and
   setting `ability_name`. Override the lifecycle hooks
   (`on_enter`, `on_release`, ...) you care about.
5. Register the effect in `effects/renderer.py::default_renderer()`.
6. Add tests in `tests/` — fixtures for hand-crafted landmark sets are
   already in `tests/conftest.py`.

## Testing

```bash
pytest                    # all 27 tests, ~1 s
pytest -k poses           # just the pose classifier
pytest --cov=.            # with coverage
```

Pose predicates and the router are deterministic and webcam-free, so
they're covered with synthetic landmark fixtures. Effects are
visual-regression material — eyeball them in `main.py`.

## Troubleshooting

- **"Cannot open camera 0"** — change `CAM_INDEX` in `config.py`.
- **"Hand landmarker model not found"** — run `python scripts/download_model.py`.
- **No audio reactivity** — first run prints the reason. Usually missing
  PortAudio. The system falls back to silent operation; effects that
  read `audio_level` just see `0.0`.
- **Volume keeps engaging when I don't want it to** — raise
  `VOLUME_GESTURE_STILLNESS` or set `SYSTEM_CONTROLS_ENABLED = False`.
- **Effects look blurry on a 4K display** — pygame doesn't auto-scale.
  Bump `WINDOW_W` / `WINDOW_H` to your display resolution. The 1280×720
  default is for headroom.
- **Wrong hand labelled Left/Right** — your camera may already mirror in
  firmware. Set `INVERT_HANDEDNESS_AFTER_MIRROR = False`.

## Roadmap

- Palm-normal estimation so the Kamehameha beam aims where the hands
  actually point, not "screen-up".
- GLSL backend for arc-heavy effects via `moderngl`.
- Sound effects layered on `ability_release` events.
- Per-user pose calibration mode.
- Two-player split-screen.

## Engineering Notes

The effects layer originally dropped frames badly whenever multiple abilities
were active. The root cause was an allocation pattern, not raw math: every
translucent primitive (`additive_polyline`, the Kamehameha beam, the
reality-tear slit, the screen flash) allocated a brand-new full-window
`1280x720` `SRCALPHA` surface and then additively blitted the entire window.
At peak charge, the Chidori effect alone issued roughly 75 of these
full-window allocate-and-blit operations per frame.

The fix replaces that with a single persistent "scratch" surface plus
bounding-box drawing: each primitive clears, draws into, and blits only the
small clipped rectangle it actually touches, so cost scales with the effect's
footprint instead of the whole window. The screen flash was additionally
fixed to fade correctly (it now pre-scales color by alpha and uses
`BLEND_RGB_ADD`, since the opaque display surface ignored source alpha under
`BLEND_RGBA_ADD`).

Measured on a representative peak-Chidori frame, the effects render path went
from about **162 ms/frame (~6 FPS ceiling) to 2.8 ms/frame — roughly 58×
faster** — eliminating the frame drops. Verified with the existing test suite
(27 passing) plus a headless render smoke test covering off-screen,
edge-straddling, and zero-length primitives.

Takeaway: the win came from not doing expensive work that wasn't needed, not
from optimizing the arithmetic.

## License

[MIT](LICENSE). Have fun. If you build something cool with this, please
ping me — I'd love to see it.

## Credits

- Hand tracking by [MediaPipe](https://developers.google.com/mediapipe).
- One Euro filter by Géry Casiez, Nicolas Roussel, Daniel Vogel
  ([1euro paper](https://cristal.univ-lille.fr/~casiez/1euro/)).
- Inspired by the shōnen of my youth.
