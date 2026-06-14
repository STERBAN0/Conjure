"""Global tunables for Aether.

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
POSE_KAMEHAMEHA_PALM_DIST_MIN: float = 0.07
POSE_KAMEHAMEHA_PALM_DIST_MAX: float = 0.32
POSE_KAMEHAMEHA_FINGERTIP_DIST: float = 0.12
POSE_CLAWED_OPENNESS_LO: float = 0.20
POSE_CLAWED_OPENNESS_HI: float = 0.60

# -----------------------------------------------------------------------------
# Ability router (state machine)
# -----------------------------------------------------------------------------
# Per-ability charge time (seconds the pose must hold to reach 100% charge).
ABILITY_CHARGE_TIME: dict[str, float] = {
    "chidori": 0.55,
    "kamehameha": 0.80,
    "rasengan": 0.50,
    "space_stretch": 0.10,
    "reality_tear": 0.10,
}

# Per-ability cooldown (seconds during which a new ability cannot start).
ABILITY_COOLDOWN: dict[str, float] = {
    "chidori": 0.50,
    "kamehameha": 0.80,
    "rasengan": 0.45,
    "space_stretch": 0.20,
    "reality_tear": 0.30,
}

# Per-ability "active phase" duration after release (seconds). 0 = no active
# phase, ability ends with the release animation.
ABILITY_ACTIVE_DURATION: dict[str, float] = {
    "chidori": 0.45,
    "kamehameha": 1.50,
    "rasengan": 0.40,
    "space_stretch": 0.0,   # continuous; ends when pose drops
    "reality_tear": 0.0,
}

# Confidence threshold for the classifier to count as "this pose is held".
POSE_MATCH_THRESHOLD: float = 0.45

# Frames the pose must drop below threshold before we exit the ability.
POSE_LOST_GRACE_FRAMES: int = 8

# Forward-thrust release motion threshold (palm size growth fraction per second).
THRUST_RELEASE_RATE: float = 1.6

# Two-handed forward-spread release (kamehameha): both hands' velocities aimed
# outward + a sudden spike in expansion above this value.
SPREAD_RELEASE_EXPANSION: float = 1.4

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

# Rasengan (spinning sphere)
RASENGAN_RADIUS_BASE: float = 18.0
RASENGAN_RADIUS_PEAK: float = 70.0
RASENGAN_PARTICLE_COUNT: int = 170
RASENGAN_CORE_COLOR: tuple[int, int, int] = (180, 230, 255)
RASENGAN_OUTER_COLOR: tuple[int, int, int] = (60, 140, 255)
RASENGAN_SHELL_DENSITY: int = 36

# Space stretch (elastic membrane)
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
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
