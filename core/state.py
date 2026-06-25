"""Shared, immutable-by-convention data types passed between modules.

All vectors are numpy arrays in normalised image coordinates (0..1) unless
explicitly named *_px (pixels). Keeping data normalised lets every layer
above the tracker stay resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class HandData:
    """One hand at one instant. All values are smoothed by the tracker."""

    label: str                       # "Left" or "Right" (post mirror-fix)
    palm: np.ndarray                 # (x, y) in 0..1
    palm_px: tuple[int, int]         # pixel coordinates
    velocity: np.ndarray             # (dx, dy) per second, normalised
    fingers_open: np.ndarray         # 5-vector in 0..1, thumb..pinky
    openness: float                  # mean of fingers_open
    spread: float                    # average inter-fingertip distance
    pinch: float                     # thumb-tip to index-tip, 0..1
    landmarks: np.ndarray            # (21, 3) normalised landmarks
    palm_size: float = 0.0           # wrist -> middle MCP, normalised
    palm_size_velocity: float = 0.0  # d(palm_size)/dt; positive = approaching camera
    tracking_confidence: float = 1.0 # MediaPipe handedness score, when available
    # Palm-plane unit normal in normalised 3-D landmark space.
    # Computed as cross(INDEX_MCP - WRIST, PINKY_MCP - WRIST), then normalised.
    # Sign convention: +z in MediaPipe normalised coords points *away* from the
    # camera (MediaPipe's z is negative when a point is closer to the camera),
    # so palm_normal[2] > 0 means the palm faces *away* from the camera and
    # palm_normal[2] < 0 means the palm faces *toward* the camera (selfie view,
    # i.e. the typical "palm facing you" pose). Compare against
    # config.PALM_FACING_CAMERA_DOT using -palm_normal[2] or abs() as needed.
    palm_normal: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    # Hand orientation derived from palm_normal's z component.
    # "palm"  → palm faces the camera (-palm_normal[2] >= config.HAND_ORIENT_FACING_MIN)
    # "back"  → back of hand faces the camera (+palm_normal[2] >= config.HAND_ORIENT_BACK_MIN)
    # "edge"  → hand seen edge-on (neither threshold met)
    orientation: str = "edge"
    # Angle of the wrist→MIDDLE_MCP vector from straight up (screen −y).
    # 0 ≈ pointing up, 90 ≈ horizontal, 180 ≈ pointing down.
    wrist_angle_deg: float = 0.0
    # Unit 2-vector (x right, y down, normalised image space) of the most recent
    # strong hand movement. Stays populated for config.HAND_FLICK_DECAY_SECONDS
    # after the peak so the router can aim projectiles even after the hand stills.
    flick: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    # Normalised-units/sec magnitude of the captured flick. 0.0 when no flick.
    flick_speed: float = 0.0
    # Live (x, y) velocity of the INDEX FINGERTIP in normalised units/sec.
    # Separate from `velocity`, which is the palm centroid (wrist + MCPs): a
    # fireball is thrown by flicking the index finger, which barely moves the
    # palm, so the fireball trigger reads this signal. 0 when not tracked.
    index_tip_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )


@dataclass(frozen=True)
class FaceData:
    """Face state for one frame, produced by FaceTracker."""

    present: bool = False
    both_eyes_closed: bool = False
    eyes_closed_duration: float = 0.0   # seconds both eyes have been continuously closed
    left_eye_px: tuple[int, int] = (0, 0)
    right_eye_px: tuple[int, int] = (0, 0)
    face_center: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5], dtype=np.float32)
    )
    # Magnitude-carrying 2-D gaze OFFSET in screen space (x right, y down).
    # (0, 0) = looking straight ahead; the vector grows the further the user
    # looks from centre (head turn + iris), so the laser maps it directly to a
    # pixel offset with no unreachable ring. Blends head pose and iris offset,
    # scaled by the inter-eye width. See FaceTracker._compute_gaze.
    gaze: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    # Full (N, 3) normalised face-mesh landmarks for this frame, or None when
    # unavailable. Carried purely so the debug overlay can draw the face "mask"
    # (the runtime laser-eyes logic only needs the eye points, centre, and gaze).
    landmarks: np.ndarray | None = None


@dataclass(frozen=True)
class ProjectileSpawn:
    """Describes a projectile that should be spawned this frame.

    Produced by the ability router and consumed by ProjectileField
    (effects package). All pixel values are in the mirrored frame's
    coordinate system.
    """

    kind: str                   # e.g. "rasengan", "fireball"
    origin_px: tuple[float, float]  # launch point in pixels
    direction: np.ndarray       # unit 2-vector (dx, dy) in pixel space
    speed_px: float             # pixels per second
    intensity: float            # 0..1 release strength
    radius_px: float            # initial visual radius in pixels


@dataclass(frozen=True)
class FrameState:
    """Everything the rest of the system needs to know about one frame."""

    frame_bgr: np.ndarray            # raw (already-mirrored) webcam frame
    timestamp: float                 # monotonic seconds
    dt: float                        # seconds since previous frame
    hands: list[HandData] = field(default_factory=list)
    face: FaceData | None = None  # None when face tracking is disabled or skipped

    def hand(self, label: str) -> HandData | None:
        for h in self.hands:
            if h.label == label:
                return h
        return None


@dataclass
class GestureSignals:
    """Continuous expressive signals. All in 0..1 unless stated otherwise.

    NOT frozen: GestureEngine mutates fields in-place each frame (motion_energy,
    grip, midpoint, span, expansion, rotation, time_scale), and main.py appends
    audio_level / audio_bands after engine.update() returns. Constructing a fresh
    instance every frame would cost a full re-allocation of the numpy fields; the
    mutable pattern is intentional for performance.
    """

    span: float = 0.0                # current normalised inter-hand distance
    expansion: float = 0.0           # signed rate of change of span (~ -1..1)
    rotation: float = 0.0            # signed angular velocity of hand axis
    grip: float = 0.0                # average grip across visible hands
    motion_energy: float = 0.0       # smoothed kinetic energy of hands
    midpoint: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    midpoint_px: tuple[int, int] = (0, 0)
    axis_angle: float = 0.0          # radians, hand-to-hand vector angle
    time_scale: float = 1.0          # global slow-mo factor (<=1)
    audio_level: float = 0.0         # 0..1 broadband audio energy
    audio_bands: np.ndarray = field(default_factory=lambda: np.zeros(8))


# -- Ability lifecycle -------------------------------------------------------

# Phases of an ability's life. Effects use these to decide what to draw.
PHASE_IDLE = "idle"
PHASE_CHARGING = "charging"      # pose held, charge ramping
PHASE_ACTIVE = "active"          # post-release "live" phase (e.g. Kamehameha beam)
PHASE_RELEASING = "releasing"    # one-shot release animation, decays out
PHASE_COOLDOWN = "cooldown"      # blocking new abilities briefly


@dataclass
class AbilityState:
    """Snapshot of the router's state for the current frame.

    Effects read this rather than the router directly, so they stay loosely
    coupled and can be unit-tested with a hand-crafted state.

    NOT frozen: AbilityRouter holds a single long-lived AbilityState instance
    and mutates it in-place on every frame tick (phase, charge, age, phase_age,
    intensity, primary_hand, secondary_hand, name). This is intentional — the
    router's state machine transitions by attribute assignment rather than
    constructing new instances, which avoids churn for effect subscribers that
    hold a reference to router.state.
    """

    name: str = ""                   # active ability id, "" when IDLE
    phase: str = PHASE_IDLE
    charge: float = 0.0              # 0..1
    age: float = 0.0                 # seconds since enter
    phase_age: float = 0.0           # seconds since last phase change
    intensity: float = 0.0           # release strength 0..1 (set on release)
    primary_hand: HandData | None = None
    secondary_hand: HandData | None = None

    @property
    def active(self) -> bool:
        return self.phase != PHASE_IDLE
