"""Global tunables for Conjure.

Every magic number that shapes how the system feels lives here. Touch this
file first when something doesn't behave the way you want — there is
intentionally no fallback to module-internal defaults.

Sections:
    Display / capture
    Vision (MediaPipe + smoothing)
    Gesture engine (continuous signals)
    Pose recognition (discrete classifier)
    Ability router (state machine)
    Effects (per-ability visual tunables)
    Audio
    System controls (volume gesture)
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Display / capture
# -----------------------------------------------------------------------------
WINDOW_W: int = 1280
WINDOW_H: int = 720
# The window always renders at the WINDOW_W x WINDOW_H logical resolution above;
# pygame's SCALED display stretches that surface to whatever the window is, so
# the title-bar maximize button and edge-resize scale the view while every
# effect/HUD coordinate stays correct.
CAM_INDEX: int = 0
CAM_REQUEST_W: int = 1280
CAM_REQUEST_H: int = 720
CAM_REQUEST_FPS: int = 60
TARGET_FPS: int = 60

# -----------------------------------------------------------------------------
# Vision
# -----------------------------------------------------------------------------
# MediaPipe reports handedness on the *original* frame. We mirror for selfie
# view, so handedness is inverted and we swap it back. Touching either flag
# without the other will silently flip Left/Right.
MIRROR_INPUT: bool = True
INVERT_HANDEDNESS_AFTER_MIRROR: bool = True

MP_MIN_DETECT_CONF: float = 0.72
MP_MIN_TRACK_CONF: float = 0.68
MP_MAX_HANDS: int = 2
# Downscale the frame fed to MediaPipe inference when it is wider than this.
# Landmarks come back normalised (0..1) so downstream geometry is unchanged,
# but the detector preprocesses far fewer pixels — the cheapest reliable FPS win
# when the webcam delivers 720p/1080p. Display still uses the full-res frame.
# Raise toward the capture width for max accuracy, lower (e.g. 480) for max FPS.
MP_INPUT_MAX_WIDTH: int = 640

# One Euro filter: lower mincutoff = more smoothing at low speed,
# higher beta = more responsive at high speed.
ONE_EURO_MIN_CUTOFF: float = 0.9
ONE_EURO_BETA: float = 0.035
ONE_EURO_DCUTOFF: float = 1.0

# Derived hand signals are filtered separately from landmarks. This keeps pose
# geometry steady while release motions still feel snappy.
HAND_VELOCITY_MIN_CUTOFF: float = 2.4
HAND_VELOCITY_BETA: float = 0.08
HAND_PALM_SIZE_VELOCITY_MIN_CUTOFF: float = 2.0
HAND_PALM_SIZE_VELOCITY_BETA: float = 0.06
HAND_STALE_RESET_SECONDS: float = 0.35

# -----------------------------------------------------------------------------
# Gesture engine (continuous signals)
# -----------------------------------------------------------------------------
SPAN_EXPANSION_GAIN: float = 6.0
SPAN_HISTORY_SECONDS: float = 0.5
GRIP_OPEN_THRESHOLD: float = 0.35
MOTION_ENERGY_DECAY: float = 0.85

# -----------------------------------------------------------------------------
# Pose recognition (discrete classifier)
# -----------------------------------------------------------------------------
# Each predicate uses these tolerances. They are intentionally loose so the
# system feels forgiving to natural pose variation; tighten for stricter mode.
POSE_FINGER_EXTENDED: float = 0.55
POSE_FINGER_FOLDED: float = 0.45
POSE_OPEN_PALM_OPENNESS: float = 0.65
POSE_FIST_OPENNESS: float = 0.30
# Kamehameha is the "triangle/diamond" chamber pose: two open hands raised
# together, palms to the camera, with the index fingertips (and thumbs) meeting
# at the apex. The fingertips touching is the decisive signature that tells it
# apart from space_stretch (open palms pulled APART) — see _is_kamehameha_cup.
POSE_KAMEHAMEHA_FINGERTIP_DIST: float = 0.16
POSE_CLAWED_OPENNESS_LO: float = 0.20
POSE_CLAWED_OPENNESS_HI: float = 0.60

# -----------------------------------------------------------------------------
# Ability router (state machine)
# -----------------------------------------------------------------------------
# Per-ability charge time (seconds the pose must hold to reach 100% charge).
ABILITY_CHARGE_TIME: dict[str, float] = {
    "chidori": 0.55,
    "kamehameha": 0.60,     # lowered so the beam charges quicker (was 0.80)
    "rasengan": 0.50,
    "space_stretch": 0.0,   # no charge — the warp just happens when both palms open & face each other
    "reality_tear": 0.30,   # brief hold of fists bumped together before pulling apart
    # New abilities (added in upgrade batch)
    "laser_eyes": 1.0,      # eyes-closed charge build (see LASER_EYES_CHARGE_SECONDS)
    "fireball": 0.5,
    "frost_nova": 0.6,
    # Long, deliberate hold: a raised palm-facing FIST slowly stops time. The
    # 2.5s matches the ticking-clock build in audio/sfx/time_freeze_charge.wav so
    # the screen reaches a full stop exactly as the build culminates.
    "time_freeze": 2.5,
}

# Per-ability cooldown (seconds during which a new ability cannot start).
ABILITY_COOLDOWN: dict[str, float] = {
    "chidori": 0.50,
    "kamehameha": 0.80,
    "rasengan": 0.45,
    "space_stretch": 0.20,
    "reality_tear": 0.30,
    # New abilities (added in upgrade batch)
    "laser_eyes": 0.8,
    "fireball": 0.5,
    "frost_nova": 0.7,
    "time_freeze": 0.6,
}

# Per-ability "active phase" duration after release (seconds). 0 = no active
# phase, ability ends with the release animation.
ABILITY_ACTIVE_DURATION: dict[str, float] = {
    "chidori": 0.0,         # HOLD ability: lightning stays while the V is held
    "kamehameha": 1.50,
    "rasengan": 0.40,
    "space_stretch": 0.0,   # continuous; ends when pose drops
    "reality_tear": 1.2,    # tear stays open ~1.2s after the fists are pulled apart
    # New abilities (added in upgrade batch)
    "laser_eyes": 1.0,
    "fireball": 0.0,        # projectile carries life after release
    "frost_nova": 1.0,      # full-screen ice flash + falling snow needs time to read
    "time_freeze": 0.0,     # continuous while pose held; 0 = no trailing active
}

# Seconds the RELEASING phase lasts for every ability except time_freeze (which
# uses TIME_FREEZE_SHATTER_DELAY). The release animation rides this window before
# the ability transitions to cooldown.
ABILITY_RELEASE_HOLD_SECONDS: float = 0.25

# Abilities temporarily disabled — not recognised at all in the live path.
# classify() drops any match whose name is in this set, so the ability never
# charges or appears. The pure-geometry _raw_matches layer is unaffected
# (geometry tests still assert the shapes are classifiable). Currently empty:
# reality_tear was re-enabled (2026-06-17) now that force_push is removed and
# space_stretch went back to open-palms, so the two-hand poses no longer collide.
DISABLED_ABILITIES: frozenset[str] = frozenset()

# Confidence threshold for the classifier to count as "this pose is held".
POSE_MATCH_THRESHOLD: float = 0.45

# Frames the pose must drop below threshold before we exit the ability.
POSE_LOST_GRACE_FRAMES: int = 8

# Forward-thrust release motion threshold (palm size growth fraction per second).
THRUST_RELEASE_RATE: float = 1.6

# Two-handed forward-spread release (kamehameha): both hands' velocities aimed
# outward + a spike in expansion above this value. Lowered from 1.4 — firing was
# rare because it demanded an unnaturally fast spread; a moderate push now fires.
SPREAD_RELEASE_EXPANSION: float = 0.85
# Once a kamehameha is fully charged it also auto-fires after this many seconds
# (forgiving fire) so a held, charged beam always releases even if the spread
# motion is too gentle to detect.
KAMEHAMEHA_AUTO_FIRE_SECONDS: float = 1.2

# -----------------------------------------------------------------------------
# Effects
# -----------------------------------------------------------------------------
# Chidori (lightning blade)
CHIDORI_ARC_COUNT_BASE: int = 4
CHIDORI_ARC_COUNT_PEAK: int = 30
CHIDORI_ARC_LENGTH_BASE_PX: float = 40.0
CHIDORI_ARC_LENGTH_PEAK_PX: float = 260.0
CHIDORI_BRANCH_PROB: float = 0.45
CHIDORI_SEGMENT_LEN_PX: float = 14.0
CHIDORI_JITTER_PX: float = 30.0
CHIDORI_CORE_COLOR: tuple[int, int, int] = (240, 250, 255)
CHIDORI_GLOW_COLOR: tuple[int, int, int] = (90, 200, 255)
CHIDORI_HIGHLIGHT_COLOR: tuple[int, int, int] = (255, 90, 220)

# Kamehameha (charged beam)
KAMEHAMEHA_SPHERE_RADIUS_BASE: float = 12.0
KAMEHAMEHA_SPHERE_RADIUS_PEAK: float = 80.0
KAMEHAMEHA_BEAM_THICKNESS_PX: float = 110.0
KAMEHAMEHA_BEAM_LENGTH_PX: float = 1600.0
KAMEHAMEHA_CORE_COLOR: tuple[int, int, int] = (255, 255, 255)
KAMEHAMEHA_OUTER_COLOR: tuple[int, int, int] = (90, 200, 255)
KAMEHAMEHA_PARTICLE_COUNT: int = 140
# Aim: the blast now follows where the cupped palms POINT. The averaged palm
# normal is split into a screen-plane lateral component and a toward-camera
# (forward) component. When you face the cup at the camera the beam fires at the
# viewer and the screen engulfs in blue; when you tilt the cup to a side the beam
# (and a far smaller engulf) shoots that way instead. AIM_LATERAL_MIN is how much
# sideways tilt is needed before the beam leaves "straight at the screen".
KAMEHAMEHA_AIM_LATERAL_MIN: float = 0.16
# Screen engulf: when the cup faces the camera the blast comes at the viewer and
# the screen floods with the beam's blue light — an expanding radial bloom from
# the sphere plus a blue veil that builds to a near-total engulf, then fades. The
# engulf strength scales with how squarely the palms face the camera, so a
# sideways-aimed blast barely floods the screen. ENGULF_MAX_ALPHA is the peak veil.
KAMEHAMEHA_ENGULF_MAX_ALPHA: int = 210
KAMEHAMEHA_ENGULF_COLOR: tuple[int, int, int] = (120, 205, 255)

# Rasengan (spinning sphere)
RASENGAN_RADIUS_BASE: float = 18.0
RASENGAN_RADIUS_PEAK: float = 70.0
RASENGAN_PARTICLE_COUNT: int = 170
RASENGAN_CORE_COLOR: tuple[int, int, int] = (180, 230, 255)
RASENGAN_OUTER_COLOR: tuple[int, int, int] = (60, 140, 255)
RASENGAN_SHELL_DENSITY: int = 36

# Space stretch (elastic membrane).
# RESTORED to the original "perfect" version: both open palms facing each other,
# pulled apart. There is NO charge — the warp just happens. Displacement scales
# directly with the live pixel distance between the two palms (see
# effects/space_stretch.py), so the rubber-band grows as the hands separate.
SPACE_STRETCH_MAX_DISPLACEMENT_PX: float = 260.0
SPACE_STRETCH_AXIS_FALLOFF: float = 0.35
SPACE_STRETCH_GRID_SPACING_PX: int = 80
SPACE_STRETCH_GRID_COLOR: tuple[int, int, int] = (90, 200, 255)

# Reality tear (jagged fracture)
REALITY_TEAR_BASE_AMP: float = 22.0
REALITY_TEAR_PEAK_AMP: float = 130.0
REALITY_TEAR_SEGMENTS: int = 42

# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------
AUDIO_ENABLED: bool = True
AUDIO_SAMPLE_RATE: int = 44100
AUDIO_BLOCK: int = 1024
AUDIO_CHANNELS: int = 1

# -----------------------------------------------------------------------------
# System controls
# -----------------------------------------------------------------------------
# Volume gesture is intentionally locked behind the "open palm" pose now that
# the pose system exists. Set False to disable entirely.
SYSTEM_CONTROLS_ENABLED: bool = True
VOLUME_GESTURE_STILLNESS: float = 0.05
VOLUME_GESTURE_OPENNESS: float = 0.7
VOLUME_SMOOTH_ALPHA: float = 0.15
VOLUME_DEAD_ZONE: float = 0.02

# -----------------------------------------------------------------------------
# Face / Laser Eyes
# -----------------------------------------------------------------------------
FACE_ENABLED: bool = True
FACE_DETECT_EVERY_N_FRAMES: int = 2          # cadence so face never bottlenecks 60Hz
FACE_MIN_DETECT_CONF: float = 0.5
FACE_MIN_PRESENCE_CONF: float = 0.5
FACE_MIN_TRACK_CONF: float = 0.5
# Raised 0.45 -> 0.55: a downward glance lowers the eyelids enough to push the
# blink blendshapes past 0.45, which falsely read as "eyes closed" and turned the
# firing laser OFF whenever the user looked down. A real, deliberate close scores
# ~0.9, so 0.55 keeps charge/stop working while ignoring a downcast look.
FACE_EYE_CLOSED_BLENDSHAPE_THRESHOLD: float = 0.55
# Eyes-closed charge build. Close both eyes and the laser charges over this many
# seconds; the laser_eyes_charge.wav build is the SAME length, so when the sound
# finishes that's the cue to OPEN YOUR EYES and fire. Restored from the old
# instant toggle (user wanted the deliberate charge back, now 1s not 2s).
LASER_EYES_CHARGE_SECONDS: float = 1.0

# -----------------------------------------------------------------------------
# Hand detection — joint-angle finger curl
# -----------------------------------------------------------------------------
# Angle at the PIP joint (vectors MCP->PIP and PIP->TIP) that defines the
# fully-extended (open) vs fully-folded (closed) extremes for mapping to 0..1.
HAND_FINGER_EXTENDED_ANGLE_DEG: float = 160.0  # >= this => 1.0 (fully open)
HAND_FINGER_FOLDED_ANGLE_DEG: float = 90.0     # <= this => 0.0 (fully folded)

# Thumb extension is measured differently from the other four fingers. The thumb
# barely curls at its IP joint, so an interior-angle metric — and the old
# cross-product against the across-knuckles axis — read it BACKWARDS: a closed
# fist showed the thumb "extended" (green) while a splayed thumb read "folded"
# (red). Instead we compare how far the thumb TIP is from the opposite palm
# corner (PINKY_MCP) versus the thumb's own IP joint. Extended => the tip reaches
# well past the joint; tucked across the palm => the tip swings back toward the
# pinky side, closer than the joint. Rotation-invariant and curl-correct.
THUMB_OPEN_RATIO_FOLDED: float = 0.98    # d(tip,pinky_mcp)/d(ip,pinky_mcp) ≤ this => folded (0.0)
THUMB_OPEN_RATIO_EXTENDED: float = 1.22  # ≥ this => fully extended (1.0)

# -----------------------------------------------------------------------------
# Pose recognition — temporal hysteresis
# -----------------------------------------------------------------------------
# These sit alongside the existing POSE_MATCH_THRESHOLD / POSE_LOST_GRACE_FRAMES.
POSE_ENTER_THRESHOLD: float = 0.55    # confidence to BEGIN a pose
POSE_EXIT_THRESHOLD: float = 0.35     # confidence to KEEP a held pose
POSE_ENTER_FRAMES: int = 4            # consecutive frames above enter-threshold to activate

# Hand-count debounce. MediaPipe occasionally emits a phantom second hand (or
# drops one) for a frame or two; on real footage the hand count flickered on
# ~4.4% of frames, which whipsawed the classifier between its single-hand and
# two-hand branches. classify() requires the new count to persist this many
# frames before it switches branch, so brief phantoms are ignored. (~50ms at
# 60fps — imperceptible delay for a deliberate two-hand pose.)
HAND_COUNT_DEBOUNCE_FRAMES: int = 3

# -----------------------------------------------------------------------------
# Palm orientation thresholds (kept for reference and vision layer)
# NOTE: gestures/poses.py predicates no longer gate on palm_normal.
# These constants are preserved because vision/hand_tracker.py references them
# in its docstring and may be used by future UI overlays.
# -----------------------------------------------------------------------------
PALM_FACING_CAMERA_DOT: float = 0.5       # |normal·camera_axis| above => palm faces camera
PALM_FACING_EACH_OTHER_DOT: float = 0.4   # for two-hand "palms toward each other" checks

# -----------------------------------------------------------------------------
# Single-hand pose tolerances (2D finger-extension based)
# -----------------------------------------------------------------------------
# Finger extension thresholds used in pose predicates (fingers_open 0..1 values).
HAND_FINGER_EXTENDED_THRESHOLD: float = 0.6  # finger_open >= this => extended
HAND_FINGER_FOLDED_THRESHOLD: float = 0.4    # finger_open <= this => folded

# -----------------------------------------------------------------------------
# Strict single-hand classifier (count-based, dead-zone rejection)
# -----------------------------------------------------------------------------
# The single-hand classifier assigns AT MOST ONE combat pose per frame. A finger
# is "extended" at/above EXT, "folded" at/below FOLD, and AMBIGUOUS in between.
# Any pose that needs a clearly-extended or clearly-folded finger is REJECTED
# when that finger is ambiguous — this is what stops relaxed/half-curled hands
# from misfiring (the dominant real-world bug). Values picked from real footage:
# real fists read every finger ~0.00 with spread ~0.33; relaxed "almost fist"
# transitional hands read 0.25-0.40 with spread 1.1-2.0.
SINGLE_FINGER_EXTENDED: float = 0.58   # finger counts as clearly extended
SINGLE_FINGER_FOLDED: float = 0.33     # finger counts as clearly folded

# Fireball (fist): every finger folded AND a tight fingertip spread. The spread
# gate is the key discriminator — true fists are compact (spread < ~0.5); the
# false positives were spread-out half-curled hands (spread 1.1-2.0).
FIST_FINGER_MAX: float = 0.24          # tightened: relaxed/idle hands read ~0.25-0.35
FIST_SPREAD_MAX: float = 0.62          # reject spread-out "almost fist" hands
FIST_OPENNESS_MAX: float = 0.28        # tightened so a half-curled idle hand isn't a "fist"
# Fireball additionally requires the fist to FACE the camera (orientation
# "palm" or "back", not "edge") so an edge-on resting hand fires nothing.

# Rasengan (index up): index clearly extended, others folded, and the index must
# beat the middle finger by a contrast margin so a sloppy V can't leak in.
RASENGAN_INDEX_MIN: float = 0.52
RASENGAN_OTHERS_MAX: float = 0.42
RASENGAN_INDEX_CONTRAST: float = 0.20  # index_open - middle_open must exceed this

# Chidori (V): index + middle clearly extended, ring + pinky clearly folded.
CHIDORI_EXT_MIN: float = 0.50
CHIDORI_FOLD_MAX: float = 0.40
CHIDORI_PARALLEL_MIN: float = 0.55     # cos-sim of index/middle directions

# Open palm / Time freeze: every finger extended. Time freeze additionally needs
# a wide spread and stillness so a casual resting open hand stops hogging the
# single ability slot (it was firing 97x across a 2-minute clip).
OPEN_FINGER_MIN: float = 0.55
TIME_FREEZE_SPREAD_MIN: float = 0.45

# (The Rasengan pointing-up and Fireball fist openness gates were removed; the
# live predicates use the finger-fold metrics defined above instead.)

# -----------------------------------------------------------------------------
# New two-hand pose tolerances
# -----------------------------------------------------------------------------
# space_stretch wants two genuinely open palms; SPREAD_MIN gates out half-curled
# hands and DIST_MIN is the small separation at which the warp begins to read.
SPACE_STRETCH_SPREAD_MIN: float = 0.4
FROST_NOVA_WRIST_DIST_MAX: float = 0.18   # crossed wrists: wrists close together
# Time freeze is now a raised CLOSED FIST with the palm facing the camera (was an
# open palm, which triggered every time the user lifted a hand). All four
# non-thumb fingers must read folded; the long 2.5s charge makes activation
# deliberate.
TIME_FREEZE_FIST_FINGER_MAX: float = 0.32  # every non-thumb finger at/below this => fist

# Two-hand distance bands (normalised palm-to-palm distance, 0..1 screen space).
KAMEHAMEHA_DIST_MAX: float = 0.32         # loose palm-to-palm bound; the fingertips-touching gate (POSE_KAMEHAMEHA_FINGERTIP_DIST) is what actually separates this from space_stretch
# space_stretch begins as soon as the open palms are this far apart and the warp
# then grows with the live separation (no upper bound — pull as wide as you like).
SPACE_STRETCH_DIST_MIN: float = 0.18
# Palm-to-palm separation span (above DIST_MIN) over which the warp ramps from
# half to full strength; the stretch factor saturates at DIST_MIN + this span.
SPACE_STRETCH_DIST_SPAN: float = 0.5

# -----------------------------------------------------------------------------
# Rasengan: open palm facing UP (cradling the sphere)
# -----------------------------------------------------------------------------
# Separated from Time Freeze (palm toward camera) by the palm-plane normal:
# palm_normal[1] is negative when the palm faces up (MediaPipe y grows downward).
# palm_normal is noisy on webcam feeds, so this is the pose most likely to need
# live tuning with the D-key debug overlay.
RASENGAN_PALM_UP_MIN: float = 0.30        # -palm_normal[1] above this => palm faces up

# -----------------------------------------------------------------------------
# Rasengan: TWO-hand stacked pose (lower cupped palm UP + top hand stirring)
# -----------------------------------------------------------------------------
# Lower hand is an open cupped palm facing the ceiling; the other hand rests on
# top, within ~the size of the sphere, and stirs in a circle to spin it up.
# Charge advances by ACCUMULATED rotation of the top hand about the lower palm
# (robust to low FPS — slow stirring still charges, unlike an instantaneous-speed
# gate that drops out when the hand blurs at speed).
RASENGAN_STACK_MAX_DIST: float = 0.32     # palm-to-palm distance to count as "stacked"
RASENGAN_STACK_MAX_DX: float = 0.22       # horizontal offset of the two palms (vertical stack)
RASENGAN_LOWER_OPEN_MIN: float = 0.45     # lower (cupping) hand must be at least this open
# Accumulated stir (radians) to reach full charge. Dropped to 1.0 (user request)
# so the barest stir of the top hand spins the sphere up almost immediately. The
# bottom cupped hand HOLDS the partial charge in the router even if the top
# (stirring) hand briefly vanishes, so progress never resets on a detection drop.
RASENGAN_SPIN_FULL_RADIANS: float = 1.0
RASENGAN_SPIN_MIN_RADIUS: float = 0.02    # ignore angular jitter when the top hand is ~on centre

# -----------------------------------------------------------------------------
# Fireball: single hand, index finger pointing UP (spawns at the fingertip,
# flies where the index is flicked).
# -----------------------------------------------------------------------------
FIREBALL_INDEX_MIN: float = 0.55          # index clearly extended
FIREBALL_OTHERS_MAX: float = 0.40         # middle/ring/pinky folded
FIREBALL_INDEX_CONTRAST: float = 0.20     # index_open - middle_open must exceed this (reject a V)
# The charging ember sphere (and the launch point) sit this many pixels ABOVE the
# index fingertip, so the fireball appears suspended just over the fingertip
# rather than buried in the centre of the hand.
FIREBALL_FINGERTIP_LIFT_PX: float = 42.0

# -----------------------------------------------------------------------------
# Time freeze: how long the frozen frame is held after movement is detected
# before the glass shatters and the frame un-freezes (both happen together).
# -----------------------------------------------------------------------------
TIME_FREEZE_SHATTER_DELAY: float = 2.0

# Progressive slow-down while CHARGING: the displayed camera frame is held for an
# increasing number of ticks as the charge rises (charge² curve), so motion
# visibly slows and stutters to a complete stop by the time the freeze lands.
# This is the count of ticks a frame is held at (near) full charge.
TIME_FREEZE_SLOWDOWN_MAX_HOLD_FRAMES: int = 24

# -----------------------------------------------------------------------------
# Reality Tear: two fists BUMPED together (charge) -> pulled apart (open)
# -----------------------------------------------------------------------------
REALITY_TEAR_FIST_FINGER_MAX: float = 0.30    # both hands: every finger at/below this
REALITY_TEAR_FIST_OPENNESS_MAX: float = 0.34  # both hands clearly closed fists
REALITY_TEAR_MATCH_MAX_DIST: float = 0.95     # still recognised as the pose when pulled apart
# The two fists must be touching (bumped together) for the charge to advance, so
# it only triggers on the deliberate "knuckles together, then rip apart" motion
# rather than any time both fists are loosely near each other.
REALITY_TEAR_TOGETHER_MAX: float = 0.22       # fists this close (≈ touching) => charge advances
REALITY_TEAR_PULL_APART_DIST: float = 0.50    # charged + separated past this => tear opens

# -----------------------------------------------------------------------------
# Laser eyes — charge-up + blink controls
# -----------------------------------------------------------------------------
# Laser eyes is a CHARGE-then-fire face ability (no hands):
#   1. close both eyes  → after BLINK_GRACE the charge starts (the whine plays),
#      ramping over LASER_EYES_CHARGE_SECONDS.
#   2. the charge sound finishing == fully charged → OPEN YOUR EYES to fire. The
#      beams then fire (and draw) along your gaze; if the face briefly stops being
#      detected it keeps firing at the last known position.
#   3. to turn it OFF, close both eyes again for OFF_BLINK_SECONDS.
# BLINK_GRACE keeps quick natural blinks from ever starting the charge. The off
# threshold is short (0.25s) so a deliberate squint stops it but blinks don't.
LASER_EYES_BLINK_GRACE_SECONDS: float = 0.15  # ignore eye-closes shorter than this
LASER_EYES_OFF_BLINK_SECONDS: float = 0.25    # while firing, a close this long turns it OFF

# -----------------------------------------------------------------------------
# Release motions
# -----------------------------------------------------------------------------
# Lowered from 1.2: the old value demanded a fast flick *while still holding the
# pose*, which is nearly impossible (fast motion blurs detection) — so fireball
# and rasengan never visibly launched. A gentle flick now fires.
THROW_RELEASE_SPEED: float = 0.65    # normalised lateral hand-velocity magnitude/s to throw
# Note: THRUST_RELEASE_RATE and SPREAD_RELEASE_EXPANSION are kept above and reused.

# Fireball repeater: charge ONCE, then fire on every fast finger flick (unlimited
# shots while the index-up pose is held). A flick must clear FIRE_FLICK_SPEED to
# count — high enough that small jitter never misfires, low enough that a firm
# flick of the finger always shoots. REFIRE_COOLDOWN caps the rate so one flick
# can't spam several projectiles across consecutive frames.
# Fire threshold for the loaded fireball. Two OR'd paths fire a shot: the latched
# flick_speed and the live palm velocity. flick_speed only ever reads 0 or
# >= HAND_FLICK_MIN_SPEED (0.35), so this value really bites on the LIVE-velocity
# path — lowered 1.2 -> 0.6 -> 0.3 -> 0.2 so a gentle flick launches a shot
# without outrunning the tracker. Firing is EDGE-TRIGGERED in the router (one
# flick = one shot): flick_speed stays latched for HAND_FLICK_DECAY_SECONDS
# (0.40s), longer than the refire cooldown, so a time-only gate double-fires from
# a single flick. We disarm on fire and re-arm only once the fire signal falls
# below REARM_FRACTION of the threshold.
FIREBALL_FIRE_FLICK_SPEED: float = 0.2   # normalised speed required to shoot
FIREBALL_REFIRE_COOLDOWN: float = 0.28   # min seconds between shots (max-rate cap)
FIREBALL_REARM_FRACTION: float = 0.6     # re-arm once fire level < this * threshold

# Rasengan throw: like the fireball, it launches on a sudden flick/shove of the
# anchor hand and travels along that direction; the projectile itself is slow
# (PROJECTILE_RASENGAN_SPEED_PX) so it drifts. Two OR'd fire paths:
#   * FLICK_SPEED — the latched flick. It only ever reads 0 or
#     >= HAND_FLICK_MIN_SPEED (0.35), so anything below that floor is inert;
#     0.35 means "fire the instant a flick is captured" and gives a clean
#     throw direction.
#   * THROW_VELOCITY — the live palm speed (the responsive path the fireball
#     already had but the rasengan lacked, which used the high 0.65
#     THROW_RELEASE_SPEED). A gentle forward shove crosses this BEFORE the
#     cupped-hand pose breaks, which is why the rasengan now actually throws.
RASENGAN_THROW_FLICK_SPEED: float = 0.35  # captured-flick path (== capture floor)
RASENGAN_THROW_VELOCITY: float = 0.2      # live palm-velocity path (gentle shove)

# Muzzle flash drawn at the launch origin so the throw reads clearly on screen.
PROJECTILE_MUZZLE_RADIUS_PX: float = 90.0
PROJECTILE_MUZZLE_SECONDS: float = 0.22

# -----------------------------------------------------------------------------
# Projectiles
# -----------------------------------------------------------------------------
PROJECTILE_MAX_ACTIVE: int = 12
# Rasengan flies at ~50% of the fireball's speed (user request) so it drifts
# across the screen rather than zipping off like the fireball.
PROJECTILE_FIREBALL_SPEED_PX: float = 1100.0
PROJECTILE_RASENGAN_SPEED_PX: float = 550.0
PROJECTILE_RASENGAN_RADIUS_PX: float = 60.0
PROJECTILE_FIREBALL_RADIUS_PX: float = 50.0
PROJECTILE_BURST_PARTICLES: int = 60
PROJECTILE_EDGE_MARGIN_PX: float = 40.0   # bursts when centre passes this far off-screen

# -----------------------------------------------------------------------------
# Laser eyes effect
# -----------------------------------------------------------------------------
# THIN beams (was 26) so the converging strokes stay legible and the shared
# impact trail reads as a precise molten point rather than a fat blob. The beams
# are drawn straight from each eye to the shared impact (no fixed length).
LASER_EYES_BEAM_THICKNESS_PX: float = 7.0
LASER_EYES_CORE_COLOR: tuple[int, int, int] = (255, 255, 255)
LASER_EYES_OUTER_COLOR: tuple[int, int, int] = (255, 70, 70)
LASER_EYES_CHARGE_GLOW_COLOR: tuple[int, int, int] = (255, 120, 80)
# Aim model — gaze is a MAGNITUDE-CARRYING 2-D offset (NOT a unit direction).
# (0,0) means "looking straight ahead" and the shared impact lands on your own
# face; the FURTHER you look from centre — by turning your HEAD and/or moving
# your EYES — the further the impact slides in that direction. Because the offset
# grows continuously from zero, EVERY pixel (including the area right around your
# own face) is reachable: there is no dead ring. Both inputs contribute so the
# aim follows head movement AND eye movement together (eyes-only felt jittery).
#   impact = eye_midpoint + (gaze - per-activation_baseline) * REACH_PX
# The per-activation baseline (captured the instant you OPEN your eyes to fire)
# makes "wherever you were looking when the beam started" = centre, so the beam
# always begins on your face regardless of head pose. See effects/laser_eyes.py.
# Sensitivity LOWERED after live testing ("too sensitive, very hard to draw"):
# head 2.6->1.5, iris 2.2->1.0, reach 900->650 (≈0.4x as twitchy on the head path,
# ≈0.33x on the iris path) so a small head/eye movement no longer flings the dot
# across the screen; the edges are still reachable with a deliberate look. The
# iris is also the noisiest input (it spikes on blinks and vanishes when looking
# down), so its gain is cut hardest — head pose now leads, iris only fine-tunes.
LASER_EYES_HEAD_GAIN: float = 1.5     # how much head turn/tilt drives the aim (stable, large range)
LASER_EYES_IRIS_GAIN: float = 1.0     # how much eye (iris) movement drives the aim (fine pointing)
LASER_EYES_REACH_PX: float = 650.0    # pixels the impact travels per unit of gaze; raise to fling the dot further for the same head/eye movement
LASER_EYES_GAZE_MAX: float = 1.6      # clamp on gaze magnitude (anti-runaway on a bad landmark frame)
LASER_EYES_GAZE_SMOOTH: float = 0.35  # EMA factor for the gaze offset (anti-jitter; lower = smoother but laggier) — lowered 0.5->0.35 for a steadier line to draw with
LASER_EYES_MELT_RADIUS_PX: float = 26.0     # molten impact pool radius (thin, for writing)
LASER_EYES_MELT_COLOR: tuple[int, int, int] = (255, 140, 40)
LASER_EYES_MELT_CORE_COLOR: tuple[int, int, int] = (255, 240, 180)
LASER_EYES_MELT_DRIP_COUNT: int = 3         # molten drips running down from the impact
# Persistent "drawing": each frame the laser deposits a single scorch blob at the
# shared impact point onto a canvas that survives across frames, so a held beam
# leaves a melted trail you can write words with. The trail PERSISTS even after the
# laser turns off — it is only erased when you press the clear key ('R'); see
# effects/laser_eyes.py.
LASER_EYES_MELT_TRAIL_ALPHA: int = 70       # per-frame scorch deposit onto the trail canvas
LASER_EYES_MELT_TRAIL_RADIUS_PX: float = 8.0   # radius of each deposited scorch blob (thin)
# Consecutive impacts within this many px are joined into a continuous stroke so
# the beam can WRITE; a larger gap is treated as a jump and left unconnected.
LASER_EYES_MELT_TRAIL_MAX_JOIN_PX: float = 220.0

# -----------------------------------------------------------------------------
# Fireball effect
# -----------------------------------------------------------------------------
FIREBALL_RADIUS_BASE: float = 16.0
FIREBALL_RADIUS_PEAK: float = 56.0
FIREBALL_CORE_COLOR: tuple[int, int, int] = (255, 240, 180)
FIREBALL_OUTER_COLOR: tuple[int, int, int] = (255, 110, 30)
FIREBALL_PARTICLE_COUNT: int = 120

# -----------------------------------------------------------------------------
# Frost nova effect
# -----------------------------------------------------------------------------
# Ring sweeps well past the frame so the whole screen reads as frozen, and the
# crack shards (FROST_SHARD_COUNT of them) are grown all the way to the screen
# edges in effects/frost_nova.py so the WHOLE screen cracks, not just the centre.
FROST_NOVA_RING_MAX_RADIUS_PX: float = 900.0
FROST_SHARD_COUNT: int = 36
FROST_CORE_COLOR: tuple[int, int, int] = (225, 250, 255)
FROST_OUTER_COLOR: tuple[int, int, int] = (120, 200, 255)

# -----------------------------------------------------------------------------
# Time freeze effect
# -----------------------------------------------------------------------------
TIME_FREEZE_TIME_SCALE: float = 0.25       # global slow-mo factor while held
TIME_FREEZE_DESATURATION: float = 0.7      # 0..1, how grey the frame goes
TIME_FREEZE_TINT_COLOR: tuple[int, int, int] = (120, 140, 200)
# True "frozen time": once the freeze reaches the active phase, the DISPLAYED
# camera frame is locked to the moment of freezing, so the user literally stops
# moving on screen (the old version only slowed sim dt — motion never stopped).
TIME_FREEZE_FREEZE_FRAME: bool = True
TIME_FREEZE_FROST_VIGNETTE: float = 0.55   # edge darkening on the frozen pane
# Shatter-on-release: when the hand drops, the frozen pane shatters like glass to
# reveal live video again — the visual "time un-freezes" cue the user asked for.
TIME_SHATTER_SECONDS: float = 0.7          # full shatter animation length
TIME_SHATTER_SHARDS: int = 26              # number of glass shards
TIME_SHATTER_CRACK_LINES: int = 14         # radial crack lines drawn first
TIME_SHATTER_GRAVITY_PX: float = 2600.0    # shard fall acceleration (px/s^2)
TIME_SHATTER_CRACK_COLOR: tuple[int, int, int] = (210, 235, 255)

# -----------------------------------------------------------------------------
# HUD / manual
# -----------------------------------------------------------------------------
HUD_SHOW_ROSTER: bool = True
# Length (px) of the gaze arrow drawn from the eye-midpoint in the debug overlay.
HUD_DEBUG_GAZE_ARROW_PX: float = 80.0

# -----------------------------------------------------------------------------
# Hand orientation (palm / back / edge) — live-tuning pass
# -----------------------------------------------------------------------------
# Orientation is derived from the palm-plane normal's z component (MediaPipe
# normalised space: z is negative toward the camera). With -normal_z above
# HAND_ORIENT_FACING_MIN the palm faces the camera; with +normal_z above
# HAND_ORIENT_BACK_MIN the back of the hand faces the camera; in between the
# hand is seen edge-on (the "side" / karate-chop orientation).
# Exposed on HandData.orientation as one of "palm" | "back" | "edge".
HAND_ORIENT_FACING_MIN: float = 0.32   # -normal_z above this => palm toward camera
HAND_ORIENT_BACK_MIN: float = 0.32     # +normal_z above this => back toward camera
# Hand-axis tilt: angle of the wrist->middle-MCP vector from straight up
# (screen -y). 0deg = pointing up, 90deg = horizontal. Exposed on
# HandData.wrist_angle_deg for pose predicates that care about tilt.

# -----------------------------------------------------------------------------
# Flick direction (aim projectiles where the hand actually flicks)
# -----------------------------------------------------------------------------
# The instantaneous velocity at the instant a projectile fires is usually ~0
# (the hand has already stopped) or stale, which made throws shoot in arbitrary
# directions (defaulting to straight up). The tracker keeps a short velocity
# history and exposes the direction of the most recent strong movement as
# HandData.flick (unit 2-vector, x right / y down) with HandData.flick_speed
# (normalised units/sec). The router aims projectiles along this.
HAND_FLICK_HISTORY_SECONDS: float = 0.30   # window scanned for the peak flick
HAND_FLICK_MIN_SPEED: float = 0.35         # min normalised speed to count as a flick
HAND_FLICK_DECAY_SECONDS: float = 0.40     # how long a captured flick stays valid

# -----------------------------------------------------------------------------
# Sound effects (procedural SFX played through pygame.mixer)
# -----------------------------------------------------------------------------
# All SFX are synthesised offline by scripts/generate_sfx.py into SOUND_SFX_DIR
# (no network, no licensing). A SoundManager subscribes to the ability hook bus
# and plays charge / ready / cast cues per ability.
SOUND_ENABLED: bool = True
SOUND_MASTER_VOLUME: float = 0.8           # 0..1 master gain for all SFX
SOUND_SFX_DIR: str = "audio/sfx"           # where generated .wav files live
SOUND_MIXER_FREQUENCY: int = 44100
SOUND_MIXER_CHANNELS: int = 16             # simultaneous SFX voices
# "Charged / ready" cue: played once when an ability's charge first reaches full
# so the user knows exactly when to release (esp. laser eyes — no counting).
SOUND_READY_CUE_ENABLED: bool = True

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
