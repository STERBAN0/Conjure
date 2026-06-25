# Conjure

[![tests](https://github.com/STERBAN0/conjure/actions/workflows/tests.yml/badge.svg)](https://github.com/STERBAN0/conjure/actions/workflows/tests.yml)

> Real-time gesture-driven anime ability effects. Make a Sasuke seal and
> watch lightning crackle from your palm. Cup your hands together for a
> Kamehameha. Pull space apart between your hands like a stretched
> rubber band. Close your eyes, open them, and fire twin laser beams.
> All from a webcam, no controllers, no marker setup.

<!-- demo video goes here -->
_Demo video coming soon._

Conjure watches your hands (and face) through a regular webcam, classifies what
*pose* you're holding, and plays the matching anime ability with
charging, release motion, and cooldown. Only one ability is active at a
time, so the input is always unambiguous — the system feels like a
fighting-game move list rather than a noise of overlapping VFX.

## Pipeline

```
                          ┌─────────────┐
        webcam  ───────►  │ HandTracker │  21 landmarks/hand, smoothed
                          └──────┬──────┘
                                 │         ┌─────────────┐
                                 │         │ FaceTracker │  eye-closed state (cadenced)
                                 │         └──────┬──────┘
                                 ▼                ▼
                         ┌──────────────────────────┐
                         │      GestureEngine        │  continuous signals
                         └──────────┬────────────────┘  (span, expansion, motion, ...)
                                    ▼
                     ┌──────────────────────┐
                     │   PoseRecognizer     │  discrete poses + hysteresis
                     └────────┬─────────────┘
                              ▼
                     ┌──────────────────┐  charging → active → cooldown
                     │  AbilityRouter   │  exactly one ability at a time
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │ EffectsRenderer  │  only the active effect runs
                     └────────┬─────────┘  (+ ProjectileField for thrown abilities)
                              ▼
                           pygame
```

## Abilities

| Ability | Gesture | Charge | Fire |
|---|---|---|---|
| **Fireball** | 1 hand: only the index finger pointing up | charge once | FLICK the finger to shoot — unlimited shots while the pose is held |
| **Rasengan** | 2 hands stacked: lower cupped palm UP, top hand stirs in a circle | stir to spin | FLICK to throw → slow-drifting sphere, bursts at the frame edge |
| **Chidori** | 1 hand: index + middle extended (V), ring + pinky folded | hold | HOLD — lightning blade stays while the sign is held |
| **Kamehameha** | 2 open hands raised together, palms to camera, index fingertips & thumbs touching to form a triangle/diamond | hold | AIM the cup: face the camera → blast fires at you and floods the screen blue; tilt to a side → the beam shoots that way |
| **Frost Nova** | 2 hands crossed at the wrists (X) | hold | SPREAD (uncross) → frost ring + ice cracks that span the whole screen |
| **Laser Eyes** | close BOTH eyes (face sign, no hands) | eyes shut ~1 s (the whine builds) | OPEN your eyes to fire — twin beams CONVERGE to one point that you aim with your HEAD and EYES together; it starts on your own face (no dead zone — reach anywhere, even yourself) and melts a trail you can draw/write with (a smiley, "HI"); eyes shut again to stop; `R` clears the drawing |
| **Space Stretch** | 2 open palms facing each other, pulled apart | none — just happens | the frame shears/stretches along the hand axis, growing as you separate |
| **Reality Tear** | 2 fists bumped together, then ripped apart | bump to charge | RIP APART → jagged glowing fracture tears open between the hands |
| **Time Freeze** | 1 closed fist, palm facing the camera | hold ~2.5 s | the scene slows to a freeze, then shatters like glass on release |

Press **M** in-app to open the hand-sign manual with cartoon illustrations of each pose. See [`docs/MANUAL.md`](docs/MANUAL.md) for the full written reference.

## Quick start

### System prerequisites

`sounddevice` needs PortAudio on macOS and Linux. Install it before running:

- **macOS:** `brew install portaudio`
- **Linux:** `sudo apt-get install libportaudio2`
- **Windows:** nothing extra needed.

If PortAudio is missing, the app starts anyway — ability sound effects are
silenced, but everything visual works normally.

### End-user install

#### Windows (PowerShell)

```powershell
git clone https://github.com/STERBAN0/conjure
cd conjure
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_model.py
.\.venv\Scripts\python.exe main.py
```

#### macOS / Linux

```bash
git clone https://github.com/STERBAN0/conjure
cd conjure
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py
python main.py
```

The download script fetches both `hand_landmarker.task` (~7.5 MB) and
`face_landmarker.task` (~6 MB). It's idempotent — re-running it skips
files that are already there. The face model is only needed for Laser
Eyes; if it's missing the app logs a hint and continues without it.

Audio SFX (`audio/sfx/`) are committed to the repo, so sound works immediately
after cloning without any extra generation step.

### For contributors

Use the editable dev install instead:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
# macOS / Linux
pip install -e ".[dev]"
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor setup.

### Tested with

- Python 3.13.1 (3.10 – 3.13 supported)
- `mediapipe` 0.10.35 (Tasks API; `mp.solutions` is gone in newer wheels)
- `pygame` 2.6.1, `opencv-python` 4.13, `numpy` 2.4
- `pycaw` 20251023 (Windows volume gesture, optional)

## Controls

| Key | What it does |
|---|---|
| `Q` | Quit |
| `ESC` | Close the manual (when it's open) — does **not** quit |
| `H` | Toggle the minimal HUD (ability label, charge, cooldown) |
| `D` | Toggle the debug overlay (signals + poses + face mask) |
| `M` | Toggle the in-app hand-sign manual (←/→ to page) |
| `L` | Toggle Laser Eyes / face tracking on/off |
| `R` | Clear the Laser Eyes molten "drawing" from the screen |
| `S` | Save a screenshot to `./screenshots/` |

## How it feels

Every ability has the same loop:

1. **Make the pose.** A confidence indicator appears around the hand.
2. **Hold to charge.** A ring fills around the active hand. Charge time
   varies — Chidori is fast, Time Freeze takes a full 2.5 seconds.
3. **Release.** Thrown abilities (Rasengan, Fireball) fire on a wrist
   flick and fly until they reach the frame edge. Melee abilities
   (Chidori) fire on a forward thrust. Spread abilities (Kamehameha,
   Frost Nova) fire when the hands fly apart. Continuous abilities
   (Space Stretch, Reality Tear, Time Freeze) stay live while you hold
   the pose.
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
conjure/
├── config.py                     all tunables in one place
├── main.py                       60 Hz pipeline + pygame loop
├── core/
│   ├── state.py                  HandData, FrameState, GestureSignals, AbilityState
│   └── hooks.py                  tiny synchronous pub/sub bus
├── vision/
│   ├── camera.py                 threaded webcam capture
│   ├── hand_tracker.py           MediaPipe + One Euro smoothing + mirror fix
│   └── face_tracker.py           MediaPipe face landmarker (eye-closed state)
├── gestures/
│   ├── smoothing.py              OneEuroFilter, EMA
│   ├── engine.py                 raw hands -> continuous signals
│   ├── poses.py                  discrete pose classifier (+ hysteresis)
│   └── router.py                 single-slot ability state machine
├── effects/
│   ├── base.py                   Effect base class
│   ├── utils.py                  shared draw primitives
│   ├── chidori.py                lightning blade
│   ├── kamehameha.py             sphere + beam
│   ├── rasengan.py               spinning sphere
│   ├── fireball.py               turbulent ember sphere
│   ├── frost_nova.py             frost ring + ice shards
│   ├── laser_eyes.py             twin eye beams + molten draw trail
│   ├── space_stretch.py          elastic membrane (BG warp)
│   ├── reality_tear.py           jagged fracture
│   ├── time_freeze.py            desaturate + slow-mo
│   ├── time_shatter.py           glass-shatter burst on time-freeze release
│   ├── projectiles.py            flying projectile field (Rasengan, Fireball)
│   ├── renderer.py               composes everything
│   └── hud.py                    ability strip, charge ring, cooldown, debug
├── audio/
│   ├── analyzer.py               threaded mic + 8-band FFT (optional)
│   ├── sounds.py                 hook-driven ability SFX playback
│   └── sfx/                      per-ability charge/ready/cast WAV cues (committed)
├── system/
│   ├── controls.py               Windows volume gesture (optional)
│   └── manual.py                 in-app hand-sign manual overlay
├── models/                       hand_landmarker.task + face_landmarker.task
├── scripts/
│   ├── download_model.py         fetches both model files (idempotent)
│   ├── generate_sfx.py           synthesises the ability SFX into audio/sfx/
│   ├── export_manual.py          renders manual pages to docs/manual_images/
│   └── diagnose_video.py         offline pose/finger diagnostics on a clip
├── docs/
│   ├── MANUAL.md                 full ability reference with hand-sign images
│   └── manual_images/            cartoon hand-sign PNGs (one per ability)
└── tests/                        pytest unit tests
```

## How the pipeline stays smooth

Three places to look:

- **`vision/hand_tracker.py`** smooths every landmark with a One Euro
  filter (cutoff frequency adapts to the signal's own velocity, so
  things stay still when still and responsive when fast). Finger curl is
  derived from PIP joint angles rather than simple tip-to-palm
  distances, and palm-normal orientation is estimated per-frame so poses
  that require the palm to face the camera or face each other are
  reliably separable. The handedness fix that makes Left/Right correct
  under a mirrored selfie view also lives here.
- **`gestures/engine.py`** smooths each derived signal again. Two-stage
  smoothing buys "stable when not moving" and "snappy when you
  are" — neither alone is enough.
- **`gestures/poses.py`** adds temporal hysteresis: a pose only
  becomes active after its raw confidence stays above
  `POSE_ENTER_THRESHOLD` for `POSE_ENTER_FRAMES` consecutive frames and
  only drops when it falls below the lower `POSE_EXIT_THRESHOLD`. This
  kills the flickering and false-fires that happen when you're
  transitioning between gestures.

The face pipeline (`vision/face_tracker.py`) runs at a reduced cadence
controlled by `FACE_DETECT_EVERY_N_FRAMES` in `config.py`, so it never
bottlenecks the main 60 Hz loop. If the face model isn't downloaded, it
degrades gracefully — Laser Eyes is simply unavailable and everything
else runs normally.

Thrown projectiles (Rasengan, Fireball) leave the hand on release and
travel in their throw direction until they reach the frame edge, where
they burst. No depth or object detection — travel-to-edge is the design.

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
6. Add a manual entry in `system/manual.py` and `docs/MANUAL.md`.
7. Add tests in `tests/` — fixtures for hand-crafted landmark sets are
   already in `tests/conftest.py`.

## Using with Claude Code

This repo ships a `CLAUDE.md` so Claude Code understands the architecture
from the first message. Contributors can open the repo with Claude Code
and ask it to add a new ability by following the steps in
[CONTRIBUTING.md](CONTRIBUTING.md) — the architecture context is already
loaded. `CLAUDE.md` also documents the test command and lint setup so
Claude Code can run checks without prompting.

## Testing

```bash
pytest                    # all tests, ~1 s
pytest -k poses           # just the pose classifier
pytest --cov=.            # with coverage
```

Pose predicates and the router are deterministic and webcam-free, so
they're covered with synthetic landmark fixtures. Effects are
visual-regression material — eyeball them in `main.py`.

## Troubleshooting

- **"Cannot open camera 0"** — change `CAM_INDEX` in `config.py`.
- **"Hand landmarker model not found"** — run `python scripts/download_model.py`.
- **Laser Eyes never activates** — the face model isn't downloaded. Run
  `python scripts/download_model.py`; it fetches both models.
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
- **Pose won't activate / activates then immediately drops** — open the
  debug overlay (`D`) and watch the confidence readouts. The hysteresis
  requires `POSE_ENTER_FRAMES` (default 4) consecutive frames above
  `POSE_ENTER_THRESHOLD` (default 0.55). Tweak those in `config.py` if
  your camera or lighting is borderline.

## Roadmap

- GLSL backend for arc-heavy effects via `moderngl`.
- Per-user pose calibration mode.
- Two-player split-screen.

## Engineering notes

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
plus a headless render smoke test covering off-screen, edge-straddling, and
zero-length primitives.

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
