"""Discrete pose classifier with hysteresis.

The gesture engine produces continuous signals. The pose classifier sits
*beside* it and turns each frame's hands into a list of recognised poses
with confidence scores. The router then chooses one ability per frame.

Pose IDs (9 abilities + neutral):
    chidori       - 1 hand: index+middle extended, ring+pinky folded
    fireball      - 1 hand: index finger pointing UP (others folded); spawns at the fingertip
    rasengan      - 2 hands: lower cupped palm facing UP + the other hand stacked on top (stirred to spin)
    kamehameha    - 2 hands: raised together, palms to camera, index fingertips + thumbs touching to form a triangle/diamond apex
    space_stretch - 2 hands: both open palms, pulled apart; the warp grows with the separation
    reality_tear  - 2 hands: both fists bumped together, then pulled apart
    frost_nova    - 2 hands: crossed wrists (palms swapped sides)
    time_freeze   - 1 hand: raised closed fist, palm facing the camera (held ~2.5s)
    laser_eyes    - face: both eyes closed long enough (no hands needed)
    open_palm     - neutral single open palm

(force_push was removed entirely — it kept colliding with space_stretch.)

Architecture:
    ``_raw_matches`` is the pure-geometry, stateless layer. Tests call it
    directly to assert geometric properties.
    ``classify`` wraps it with per-pose hysteresis: a pose only becomes
    *active* after raw confidence >= POSE_ENTER_THRESHOLD for POSE_ENTER_FRAMES
    consecutive frames and stays active while >= POSE_EXIT_THRESHOLD.

NOTE: Most predicates use only robust 2D signals (fingers_open, openness,
spread, velocity, normalised palm distance, and landmark 2D positions). The
orientation-sensitive poses (rasengan = palm up, time_freeze = palm toward
camera, and fireball's edge-on reject) additionally read ``HandData.orientation``
/ ``palm_normal``, which is derived from MediaPipe's noisy z-coordinate and is
the part most likely to need live tuning via the D-key debug overlay.

NOTE: ``classify`` suppresses any pose listed in ``config.DISABLED_ABILITIES``
(currently empty). The geometry in ``_raw_matches`` still recognises every pose;
only the live, stateful path would drop a disabled one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from core.state import FaceData, FrameState, HandData
from vision.landmarks import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, WRIST

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class PoseMatch:
    """A recognised pose for one frame.

    ``primary`` is the dominant hand for single-hand poses or the "anchor"
    hand for two-hand poses. ``secondary`` is the supporting hand if any.
    ``extra`` is a free-form bag for predicate output the router may want.
    """

    name: str
    confidence: float
    primary: HandData | None = None
    secondary: HandData | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hysteresis state per pose
# ---------------------------------------------------------------------------

@dataclass
class _PoseHysteresis:
    """Tracks enter/exit state for one pose id."""

    active: bool = False
    enter_frames: int = 0   # consecutive frames raw conf >= ENTER_THRESHOLD


class PoseRecognizer:
    """Stateful classifier with per-pose hysteresis.

    Call ``classify`` each frame for stabilised output. Call ``_raw_matches``
    (accessible for tests) for the raw geometry layer only.
    """

    def __init__(self) -> None:
        self._hysteresis: dict[str, _PoseHysteresis] = {}
        # Debounced single/dual hand mode (see _update_hand_mode).
        self._hand_mode: str = "single"
        self._two_streak: int = 0
        self._one_streak: int = 0

    def _get_hysteresis(self, name: str) -> _PoseHysteresis:
        if name not in self._hysteresis:
            self._hysteresis[name] = _PoseHysteresis()
        return self._hysteresis[name]

    def _update_hand_mode(self, hands: list[HandData]) -> None:
        """Debounce the single↔dual branch so a phantom hand can't whipsaw it.

        The new count must persist HAND_COUNT_DEBOUNCE_FRAMES frames before the
        mode flips. Called once per frame by ``classify`` (the live path).
        """
        if len(hands) >= 2:
            self._two_streak += 1
            self._one_streak = 0
        else:
            self._one_streak += 1
            self._two_streak = 0
        k = config.HAND_COUNT_DEBOUNCE_FRAMES
        if self._hand_mode != "dual" and self._two_streak >= k:
            self._hand_mode = "dual"
        elif self._hand_mode != "single" and self._one_streak >= k:
            self._hand_mode = "single"

    def _raw_matches(
        self, frame: FrameState, hand_mode: str | None = None
    ) -> list[PoseMatch]:
        """Pure geometry pass — stateless.

        Returns all poses whose geometry conditions are met, with raw
        confidence scores. No hysteresis applied. Tests call this directly.

        Hand-count splitting prevents cross-talk between single- and two-hand
        poses. With ``hand_mode=None`` (the default, used by the geometry tests)
        the split is the strict by-count rule. ``classify`` passes a *debounced*
        mode so a brief phantom hand can't flip the branch mid-pose.
        """
        out: list[PoseMatch] = []
        left = frame.hand("Left")
        right = frame.hand("Right")
        hands = frame.hands

        # --- Face-based pose (no hands needed) ---
        if frame.face is not None:
            m = _is_laser_eyes(frame.face)
            if m is not None:
                out.append(m)

        if hand_mode is None:
            use_single = len(hands) == 1
            use_dual = len(hands) == 2
        else:
            use_single = hand_mode == "single"
            use_dual = hand_mode == "dual"

        # --- Single-hand poses ---
        # A strict, mutually-exclusive classifier returns AT MOST ONE combat
        # pose (plus the neutral open_palm). When two hands are present but the
        # mode is single (a transient phantom), classify the dominant hand.
        if use_single and hands:
            hand = (
                hands[0]
                if len(hands) == 1
                else max(hands, key=lambda h: (h.tracking_confidence, h.palm_size))
            )
            out.extend(_single_hand_matches(hand))

        # --- Two-hand poses (need both a Left and a Right hand) ---
        if use_dual and left is not None and right is not None:
            for pred in (
                _is_rasengan,
                _is_kamehameha_cup,
                _is_space_stretch,
                _is_reality_tear,
                _is_frost_nova,
            ):
                m = pred(left, right)
                if m is not None:
                    out.append(m)

        return out

    def classify(self, frame: FrameState) -> list[PoseMatch]:
        """Geometry + hysteresis.

        A pose becomes active after POSE_ENTER_FRAMES consecutive frames
        above POSE_ENTER_THRESHOLD and drops when below POSE_EXIT_THRESHOLD.
        Only active poses are returned.
        """
        self._update_hand_mode(frame.hands)
        raw = self._raw_matches(frame, hand_mode=self._hand_mode)
        raw_by_name: dict[str, PoseMatch] = {}
        for m in raw:
            # Drop abilities turned off in config.DISABLED_ABILITIES before they
            # can accumulate hysteresis or become active (the live disable).
            if m.name in config.DISABLED_ABILITIES:
                continue
            cur = raw_by_name.get(m.name)
            if cur is None or m.confidence > cur.confidence:
                raw_by_name[m.name] = m

        # Collect all pose names we need to consider: seen this frame + any
        # currently active ones that might need to drop.
        all_names = set(raw_by_name) | {
            name for name, h in self._hysteresis.items() if h.active
        }

        out: list[PoseMatch] = []
        for name in all_names:
            h = self._get_hysteresis(name)
            match = raw_by_name.get(name)
            raw_conf = match.confidence if match else 0.0

            if h.active:
                if raw_conf < config.POSE_EXIT_THRESHOLD:
                    h.active = False
                    h.enter_frames = 0
                    log.debug("pose hysteresis: DROP %s (conf=%.2f)", name, raw_conf)
                else:
                    # Stay active — return the match with its raw confidence.
                    if match is not None:
                        out.append(match)
            else:
                if raw_conf >= config.POSE_ENTER_THRESHOLD:
                    h.enter_frames += 1
                    if h.enter_frames >= config.POSE_ENTER_FRAMES:
                        h.active = True
                        h.enter_frames = 0
                        log.debug(
                            "pose hysteresis: ENTER %s (conf=%.2f)", name, raw_conf
                        )
                        if match is not None:
                            out.append(match)
                else:
                    h.enter_frames = 0

        return out


# ---------------------------------------------------------------------------
# Finger-state helpers (strict three-way classification with a dead-zone)
# ---------------------------------------------------------------------------

_EXT = "ext"
_FOLD = "fold"
_AMB = "amb"


def _finger_state(value: float) -> str:
    """Three-way finger state: clearly extended, clearly folded, or ambiguous.

    The ambiguous band between SINGLE_FINGER_FOLDED and SINGLE_FINGER_EXTENDED is
    what makes the classifier reject relaxed/half-curled transitional hands
    instead of misfiring an ability on them.
    """
    if value >= config.SINGLE_FINGER_EXTENDED:
        return _EXT
    if value <= config.SINGLE_FINGER_FOLDED:
        return _FOLD
    return _AMB


# ---------------------------------------------------------------------------
# Single-hand classifier — returns AT MOST ONE combat pose (+ neutral open_palm)
# ---------------------------------------------------------------------------

def _single_hand_matches(h: HandData) -> list[PoseMatch]:
    """Classify one hand into at most one combat pose, plus neutral open_palm.

    Poses are checked most-specific-first and the first match wins, so a single
    frame can never report e.g. both fireball and chidori. Any pose whose
    required fingers are ambiguous (in the dead-zone) is skipped, so a hand
    that is mid-transition fires nothing.
    """
    f = h.fingers_open
    idx, mid, rng_f, pky = float(f[1]), float(f[2]), float(f[3]), float(f[4])
    states = [_finger_state(v) for v in (idx, mid, rng_f, pky)]

    # Chidori (index+middle V) is checked before fireball (index only): a clean
    # V has the middle extended and must win; fireball requires it folded.
    m = _match_chidori(h, idx, mid, rng_f, pky)
    if m is not None:
        return [m]

    m = _match_fireball(h, idx, mid, rng_f, pky)
    if m is not None:
        return [m]

    # Closed fist with the palm toward the camera, held → time_freeze. Checked
    # before the open-palm branch; a fist can't also be all-extended so order is
    # not strictly required, but keeping it here documents the intent.
    m = _match_time_freeze(h, idx, mid, rng_f, pky)
    if m is not None:
        return [m]

    # Fully open hand: neutral open_palm only (no ability). Rasengan is a TWO-hand
    # pose (see _is_rasengan); time_freeze is now a fist (above).
    if all(s == _EXT for s in states):
        return [_make_open_palm(h)]

    # Anything else (ambiguous / transitional) deliberately fires nothing.
    return []


def _match_fireball(
    h: HandData, idx: float, mid: float, rng_f: float, pky: float
) -> PoseMatch | None:
    """Index finger pointing UP, the other three fingers folded.

    The fireball spawns at the index fingertip and flies where the index is
    flicked (both handled in the router). A clean V (chidori) is rejected because
    the middle finger must be folded and the index must beat the middle by
    FIREBALL_INDEX_CONTRAST.
    """
    if idx < config.FIREBALL_INDEX_MIN:
        return None
    if max(mid, rng_f, pky) > config.FIREBALL_OTHERS_MAX:
        return None
    if idx - mid < config.FIREBALL_INDEX_CONTRAST:
        return None
    # Index must point generally UP: fingertip above the knuckle on screen
    # (MediaPipe y grows downward, so "above" means a smaller y).
    tip = h.landmarks[INDEX_TIP, :2]
    mcp = h.landmarks[INDEX_MCP, :2]
    if float(tip[1]) >= float(mcp[1]):
        return None
    contrast = idx - max(mid, rng_f, pky)
    conf = float(np.clip(0.55 + 0.45 * contrast, 0.0, 1.0))
    return PoseMatch("fireball", conf, primary=h, extra={"fingertip": tip.copy()})


def _match_chidori(
    h: HandData, idx: float, mid: float, rng_f: float, pky: float
) -> PoseMatch | None:
    """Index + middle extended (V sign), ring + pinky folded."""
    if idx < config.CHIDORI_EXT_MIN or mid < config.CHIDORI_EXT_MIN:
        return None
    if rng_f > config.CHIDORI_FOLD_MAX or pky > config.CHIDORI_FOLD_MAX:
        return None

    idx_dir = h.landmarks[INDEX_TIP, :2] - h.landmarks[INDEX_MCP, :2]
    mid_dir = h.landmarks[MIDDLE_TIP, :2] - h.landmarks[MIDDLE_MCP, :2]
    parallel = _cos(idx_dir, mid_dir)
    if parallel < config.CHIDORI_PARALLEL_MIN:
        return None

    contrast = min(idx, mid) - max(rng_f, pky)
    conf = float(np.clip(
        0.5 + 0.5 * contrast + (parallel - config.CHIDORI_PARALLEL_MIN) * 0.4,
        0.0, 1.0,
    ))
    return PoseMatch("chidori", conf, primary=h, extra={"fingertip_dir": idx_dir})


def _match_time_freeze(
    h: HandData, idx: float, mid: float, rng_f: float, pky: float
) -> PoseMatch | None:
    """Raised CLOSED FIST with the palm facing the CAMERA — slowly stops time.

    All four non-thumb fingers must read folded and the palm must face the
    camera. This replaces the old open-palm gesture, which fired whenever the
    user simply lifted an open hand. The long 2.5s charge (config) makes
    activation deliberate, so a brief fist no longer freezes the screen.

    Palm orientation comes from the (noisy) palm_normal; HAND_ORIENT_FACING_MIN
    may want live tuning with the D-overlay if a palm-facing fist doesn't read.
    """
    if max(idx, mid, rng_f, pky) > config.TIME_FREEZE_FIST_FINGER_MAX:
        return None
    if h.orientation != "palm":
        return None
    fold = 1.0 - max(idx, mid, rng_f, pky) / max(
        config.TIME_FREEZE_FIST_FINGER_MAX, 1e-6
    )
    face = float(np.clip(-float(h.palm_normal[2]), 0.0, 1.0))
    conf = float(np.clip(0.55 + 0.25 * fold + 0.2 * face, 0.0, 1.0))
    return PoseMatch("time_freeze", conf, primary=h)


def _make_open_palm(h: HandData) -> PoseMatch:
    """Neutral fully-open palm marker (not an ability)."""
    conf = float(np.clip((h.openness - config.POSE_OPEN_PALM_OPENNESS) * 3.5, 0.3, 1.0))
    return PoseMatch("open_palm", conf, primary=h)


# ---------------------------------------------------------------------------
# Two-hand predicates
# ---------------------------------------------------------------------------

def _is_rasengan(left: HandData, right: HandData) -> PoseMatch | None:
    """Two hands stacked vertically: the LOWER hand is an open cupped palm
    facing UP (cradling the sphere); the OTHER hand rests on top within roughly
    a sphere's width and stirs in a circle to spin it up.

    Charge is driven by ACCUMULATED rotation of the top hand about the lower
    palm (in the router), so slow stirring works and it doesn't drop out when a
    fast hand motion-blurs at low FPS.

    Distinguished from kamehameha (palms toward the camera) by the lower hand's
    palm-up orientation. palm_normal is noisy on webcam feeds, so this and the
    kamehameha split are the most likely to need live D-overlay tuning.
    """
    # Lower hand = the one further down the screen (larger y).
    lower, upper = (left, right) if left.palm[1] >= right.palm[1] else (right, left)

    # Lower hand: reasonably open/cupped and palm facing the ceiling.
    if lower.openness < config.RASENGAN_LOWER_OPEN_MIN:
        return None
    if -float(lower.palm_normal[1]) < config.RASENGAN_PALM_UP_MIN:
        return None

    # Stacked: top hand above the lower one, horizontally aligned and close.
    if upper.palm[1] >= lower.palm[1]:
        return None
    if abs(float(upper.palm[0]) - float(lower.palm[0])) > config.RASENGAN_STACK_MAX_DX:
        return None
    palm_dist = float(np.linalg.norm(upper.palm - lower.palm))
    if palm_dist > config.RASENGAN_STACK_MAX_DIST:
        return None

    up_amt = float(np.clip(-float(lower.palm_normal[1]), 0.0, 1.0))
    conf = float(np.clip(0.55 + 0.45 * up_amt, 0.0, 1.0))
    return PoseMatch(
        "rasengan", conf,
        primary=lower, secondary=upper,
        extra={"sphere_anchor": lower.palm.copy(), "spin_hand": upper},
    )


def _is_kamehameha_cup(left: HandData, right: HandData) -> PoseMatch | None:
    """The Kamehameha "triangle/diamond" chamber pose: two open hands raised
    together, palms toward the camera, with the index fingertips (and thumbs)
    meeting at the apex to frame a triangular window.

    The decisive signature — and the fix for kamehameha being confused with
    space_stretch — is that the two hands' INDEX FINGERTIPS touch (distance
    <= POSE_KAMEHAMEHA_FINGERTIP_DIST). space_stretch is open palms pulled APART,
    so its fingertips are always far apart and can never satisfy this gate, even
    in the palm-distance band where the two poses used to overlap. Also rejects
    the rasengan stack (lower hand palm UP) and requires a palm toward the camera.
    """
    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if palm_dist > config.KAMEHAMEHA_DIST_MAX:
        return None

    # Apex gate: the index fingertips must be touching (forming the triangle).
    # This is what space_stretch (fingertips pulled apart) can never satisfy.
    tip_dist = float(
        np.linalg.norm(left.landmarks[INDEX_TIP, :2] - right.landmarks[INDEX_TIP, :2])
    )
    if tip_dist > config.POSE_KAMEHAMEHA_FINGERTIP_DIST:
        return None

    # Reject the rasengan stack (lower hand palm UP) so the two don't collide.
    lower = left if left.palm[1] >= right.palm[1] else right
    if -float(lower.palm_normal[1]) >= config.RASENGAN_PALM_UP_MIN:
        return None

    # At least one palm faces the camera (the "gathering energy" cup).
    if left.orientation != "palm" and right.orientation != "palm":
        return None

    apex = 1.0 - tip_dist / max(config.POSE_KAMEHAMEHA_FINGERTIP_DIST, 1e-6)
    conf = float(np.clip(0.55 + 0.45 * apex, 0.0, 1.0))
    axis = np.array([0.0, -1.0])
    return PoseMatch(
        "kamehameha", conf,
        primary=left, secondary=right,
        extra={"beam_axis": axis},
    )


def _is_space_stretch(left: HandData, right: HandData) -> PoseMatch | None:
    """Both palms OPEN and spread, held apart — the original space-stretch pose.

    This is the restored "perfect" version the user asked for: two open palms
    (facing each other) pulled apart. There is no charge and no edge-on
    requirement — open both hands and separate them and the membrane warps,
    growing with how far apart the hands are. Confidence ramps with the
    separation so the visual reacts to the stretch.
    """
    if left.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if right.openness < config.POSE_OPEN_PALM_OPENNESS:
        return None
    if left.spread < config.SPACE_STRETCH_SPREAD_MIN:
        return None
    if right.spread < config.SPACE_STRETCH_SPREAD_MIN:
        return None
    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if palm_dist < config.SPACE_STRETCH_DIST_MIN:
        return None
    stretch = float(np.clip((palm_dist - config.SPACE_STRETCH_DIST_MIN) / config.SPACE_STRETCH_DIST_SPAN, 0.0, 1.0))
    conf = float(np.clip(0.5 + 0.5 * stretch, 0.0, 1.0))
    return PoseMatch(
        "space_stretch", conf,
        primary=left, secondary=right,
        extra={"stretch": stretch, "palm_dist": palm_dist},
    )


def _is_reality_tear(left: HandData, right: HandData) -> PoseMatch | None:
    """Both hands closed fists. Charge while together, tear opens on pull-apart.

    This predicate only recognises "two fists" (at any distance up to
    REALITY_TEAR_MATCH_MAX_DIST) and reports the current palm distance in
    ``extra``. The router gates charge on the fists being together and fires the
    release when they are pulled apart, so the pose must keep matching while the
    hands separate — hence the generous distance ceiling.
    """
    for h in (left, right):
        f = h.fingers_open
        if max(float(f[1]), float(f[2]), float(f[3]), float(f[4])) > (
            config.REALITY_TEAR_FIST_FINGER_MAX
        ):
            return None
        if h.openness > config.REALITY_TEAR_FIST_OPENNESS_MAX:
            return None

    palm_dist = float(np.linalg.norm(left.palm - right.palm))
    if palm_dist > config.REALITY_TEAR_MATCH_MAX_DIST:
        return None

    tightness = 1.0 - max(left.openness, right.openness) / max(
        config.REALITY_TEAR_FIST_OPENNESS_MAX, 1e-6
    )
    conf = float(np.clip(0.6 + 0.4 * tightness, 0.0, 1.0))
    return PoseMatch(
        "reality_tear", conf,
        primary=left, secondary=right,
        extra={"palm_dist": palm_dist},
    )


def _is_frost_nova(left: HandData, right: HandData) -> PoseMatch | None:
    """Crossed wrists: wrists close together AND hands swapped sides.

    Geometry: wrist landmarks within FROST_NOVA_WRIST_DIST_MAX AND
    the left-labelled hand's palm x is to the RIGHT of the right-labelled
    hand's palm x (they swapped sides). Very distinct from every other pose.
    """
    left_wrist = left.landmarks[WRIST, :2]
    right_wrist = right.landmarks[WRIST, :2]
    wrist_dist = float(np.linalg.norm(left_wrist - right_wrist))

    if wrist_dist > config.FROST_NOVA_WRIST_DIST_MAX:
        return None

    # Crossed: left hand palm is to the right of right hand palm.
    if left.palm[0] <= right.palm[0]:
        return None

    dist_score = float(np.clip(
        1.0 - wrist_dist / config.FROST_NOVA_WRIST_DIST_MAX, 0.0, 1.0
    ))
    cross_score = float(np.clip((left.palm[0] - right.palm[0]) * 4.0, 0.0, 1.0))
    conf = float(np.clip(0.5 * dist_score + 0.5 * cross_score, 0.0, 1.0))
    return PoseMatch("frost_nova", conf, primary=left, secondary=right)


# ---------------------------------------------------------------------------
# Face predicate
# ---------------------------------------------------------------------------

def _is_laser_eyes(face: FaceData) -> PoseMatch | None:
    """Eyes closed past the blink grace — kept as a recognised pose for the HUD.

    NOTE: the router no longer charges laser_eyes from this pose match (it drops
    it and runs its own face-driven charge→fire machine in
    AbilityRouter._update_laser). This match is retained only so the diagnostics
    layer / HUD roster can show that eyes-closed is being recognised. The user
    must still hold their eyes closed for LASER_EYES_BLINK_GRACE_SECONDS to
    distinguish intentional activation from natural blinks.
    """
    if not face.present:
        return None
    if not face.both_eyes_closed:
        return None
    if face.eyes_closed_duration < config.LASER_EYES_BLINK_GRACE_SECONDS:
        return None
    return PoseMatch(
        "laser_eyes",
        confidence=1.0,
        primary=None,
        extra={"face": face},
    )


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with epsilon to avoid divide-by-zero."""
    na = float(np.linalg.norm(a)) + 1e-6
    nb = float(np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / (na * nb))
