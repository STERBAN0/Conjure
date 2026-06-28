"""Ability router: the single-slot state machine.

The router consumes pose matches each frame and decides which ability — at
most one — is currently in flight. It transitions through a small state
machine and emits hook events that effects subscribe to.

Phases (see core.state):
    IDLE       - no ability active.
    CHARGING   - matching pose held; charge ramps 0 -> 1 over charge_time.
    ACTIVE     - post-release sustain (e.g. Kamehameha beam) or, for
                 continuous abilities (space stretch), the whole "live" phase.
    RELEASING  - one-shot release animation; auto-decays into COOLDOWN.
    COOLDOWN   - blocks new abilities for cooldown_time seconds.

Hook events (synchronous):
    ability_enter(name, frame, signals)          -> on first transition into CHARGING
    ability_charge(name, charge, frame, signals) -> per-frame while CHARGING
    ability_release(name, intensity, frame)      -> on transition into ACTIVE/RELEASING
    ability_active(name, frame, signals)         -> per-frame while ACTIVE
    ability_exit(name)                            -> on return to IDLE/COOLDOWN
    projectile_spawn(ProjectileSpawn)            -> when a projectile is launched

Effects can subscribe to any of these. They can also read `router.state`
each frame for a richer snapshot (charge, age, primary_hand, etc.).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

import config
from core.hooks import HookBus
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_IDLE,
    PHASE_RELEASING,
    AbilityState,
    FrameState,
    GestureSignals,
    HandData,
)
from gestures.poses import PoseMatch
from gestures.router_laser import _LaserMixin
from gestures.router_projectile import _ProjectileMixin

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbilityDef:
    """Static configuration for a single ability slot."""

    name: str
    pose_id: str
    charge_time: float
    cooldown: float
    active_duration: float
    release_motion: str | None     # "thrust"|"spread"|"throw"|"shove"|"pose_release"|"pull_apart"|None
    projectile_kind: str | None = None  # "rasengan"|"fireball"|None

    @classmethod
    def from_config(
        cls,
        name: str,
        pose_id: str,
        release_motion: str | None,
        projectile_kind: str | None = None,
    ) -> AbilityDef:
        return cls(
            name=name,
            pose_id=pose_id,
            charge_time=config.ABILITY_CHARGE_TIME[name],
            cooldown=config.ABILITY_COOLDOWN[name],
            active_duration=config.ABILITY_ACTIVE_DURATION[name],
            release_motion=release_motion,
            projectile_kind=projectile_kind,
        )


def default_abilities() -> dict[str, AbilityDef]:
    """The canonical ability roster Conjure ships with."""
    return {
        a.name: a for a in (
            AbilityDef.from_config("chidori",      "chidori",      None,           None),
            AbilityDef.from_config("kamehameha",   "kamehameha",   "spread",       None),
            AbilityDef.from_config("rasengan",     "rasengan",     "throw",        "rasengan"),
            AbilityDef.from_config("fireball",     "fireball",     "throw",        "fireball"),
            AbilityDef.from_config("frost_nova",   "frost_nova",   "pose_release", None),
            AbilityDef.from_config("laser_eyes",   "laser_eyes",   "pose_release", None),
            AbilityDef.from_config("space_stretch","space_stretch", None,          None),
            AbilityDef.from_config("reality_tear", "reality_tear", "pull_apart",   None),
            AbilityDef.from_config("time_freeze",  "time_freeze",  None,           None),
        )
    }


class AbilityRouter(_LaserMixin, _ProjectileMixin):
    """Single-active-ability state machine.

    Lifetime is per-process; keep one instance.
    """

    def __init__(
        self,
        hooks: HookBus,
        abilities: dict[str, AbilityDef] | None = None,
    ) -> None:
        self.hooks = hooks
        self.abilities = abilities if abilities is not None else default_abilities()
        self.state = AbilityState()
        self._lost_frames = 0
        # ability_exit must fire exactly once per ability lifecycle; both
        # _goto_cooldown and _goto_idle used to emit it, double-triggering
        # one-shot subscribers (e.g. the time-freeze glass shatter fired twice).
        self._exit_emitted = False
        # Rasengan charges by ACCUMULATED rotation of the top hand about the
        # lower palm (robust to low FPS). Tracked here across charging frames.
        self._rasengan_accum = 0.0
        self._rasengan_prev_angle: float | None = None
        # Wall-age (s.age) at which the current ability first reached full charge,
        # so projectile auto-fire measures time-since-full regardless of whether
        # charge advanced by dt or by rotation.
        self._charge_full_since: float | None = None
        # Fireball repeater: once charged, every fast finger flick fires a shot.
        # _refire_timer is a max-rate cap; _armed edge-triggers so one flick
        # (whose flick_speed stays latched ~0.40s) only ever fires one shot.
        self._fireball_refire_timer: float = 0.0
        self._fireball_armed: bool = True
        # Laser eyes is a face-driven CHARGE→fire ability handled outside the
        # normal single-slot machine (see _update_laser). State: "off" (idle),
        # "charging" (eyes closed, ramping over LASER_EYES_CHARGE_SECONDS), "on"
        # (firing). _laser_armed gates the charge so it only begins after the eyes
        # have been open (quick blinks don't start it); _laser_off_armed gates the
        # off-close so turning on doesn't instantly turn off while the eyes are
        # still shut from charging.
        self._laser_state: str = "off"
        self._laser_armed: bool = False
        self._laser_off_armed: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: Iterable[PoseMatch],
    ) -> AbilityState:
        match_by_pose = self._best_per_pose(matches)

        # Laser eyes is a face-driven charge→fire ability (eyes closed to charge,
        # open to fire, close again to stop). It owns the slot while charging or
        # firing, so handle it first and let it short-circuit the normal machine.
        # Any stray laser_eyes pose match is dropped so the normal machine never
        # tries to charge it the old (hand-pose) way. Return self.state (not the
        # `s` captured above): beginning a laser charge replaces self.state with a
        # fresh AbilityState, so `s` would be stale on the transition frame.
        # NOTE: age/phase_age are incremented AFTER the laser check so that a
        # frame where the laser path replaces self.state doesn't lose the dt
        # that was applied to the now-discarded old state object.
        if self._update_laser(frame, signals):
            self.state.age += frame.dt
            self.state.phase_age += frame.dt
            return self.state
        match_by_pose.pop("laser_eyes", None)  # HUD-only match; the face path drives it, never hand-charged

        s = self.state
        s.age += frame.dt
        s.phase_age += frame.dt

        if s.phase == PHASE_IDLE:
            self._tick_idle(frame, signals, match_by_pose)
        elif s.phase == PHASE_CHARGING:
            self._tick_charging(frame, signals, match_by_pose)
        elif s.phase == PHASE_ACTIVE:
            self._tick_active(frame, signals, match_by_pose)
        elif s.phase == PHASE_RELEASING:
            self._tick_releasing(frame, signals)
        elif s.phase == PHASE_COOLDOWN:
            self._tick_cooldown(frame, signals)
        else:  # pragma: no cover — defensive
            log.warning("router: unknown phase %r, resetting", s.phase)
            self._goto_idle()

        return s

    def is_active(self, ability_name: str) -> bool:
        return self.state.name == ability_name and self.state.phase != PHASE_IDLE

    # ------------------------------------------------------------------
    # Phase ticks
    # ------------------------------------------------------------------

    def _tick_idle(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: dict[str, PoseMatch],
    ) -> None:
        candidate = self._pick_candidate(matches)
        if candidate is None:
            return
        ability = self.abilities[candidate.name]
        s = self.state
        s.name = ability.name
        s.phase = PHASE_CHARGING
        s.charge = 0.0
        s.age = 0.0
        s.phase_age = 0.0
        s.intensity = 0.0
        s.primary_hand = candidate.primary
        s.secondary_hand = candidate.secondary
        self._lost_frames = 0
        self._exit_emitted = False
        self._rasengan_accum = 0.0
        self._rasengan_prev_angle = None
        self._charge_full_since = None
        self._fireball_refire_timer = 0.0
        self._fireball_armed = True
        self.hooks.emit("ability_enter", ability.name, frame, signals)
        log.info("router: ENTER %s (conf=%.2f)", ability.name, candidate.confidence)

    def _tick_charging(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: dict[str, PoseMatch],
    ) -> None:
        s = self.state
        ability = self.abilities[s.name]
        match = matches.get(ability.pose_id)

        # pull_apart abilities (reality_tear): charge while the two fists are
        # together; open the tear when they are pulled apart after full charge.
        if ability.release_motion == "pull_apart":
            self._tick_reality_tear(ability, match, frame, signals)
            return

        # rasengan: charge by stirring; HOLD the partial charge in the lower
        # cupped hand even when the top (stirring) hand briefly drops; fire only
        # on a throw of the lower hand. Custom tick, so it bypasses the generic
        # forgiving-fire (no auto-timeout / pose-drop launch).
        if s.name == "rasengan":
            self._tick_rasengan(ability, match, frame, signals)
            return

        # fireball: charge ONCE, then fire on every fast finger flick for as long
        # as the index-up pose is held (unlimited shots, no re-charge). Custom
        # tick so it bypasses the one-shot forgiving-fire below.
        if s.name == "fireball":
            self._tick_fireball(ability, match, frame, signals)
            return

        # pose_release abilities: fire on pose *disappearing* when fully charged.
        # Must check this BEFORE the "match lost → cooldown" path below.
        if ability.release_motion == "pose_release":
            pose_present = (
                match is not None and match.confidence >= config.POSE_MATCH_THRESHOLD
            )
            if s.charge >= 1.0 and not pose_present:
                # Pose just dropped while fully charged → fire!
                s.intensity = 1.0
                self.hooks.emit("ability_release", s.name, s.intensity, frame)
                self._spawn_projectile_if_needed(ability, frame)
                self._goto_active(frame, signals)
                return
            if s.charge < 1.0 and not pose_present:
                # Pose dropped before full charge → cancel
                self._goto_cooldown(reason="pose_lost_early", frame=frame)
                return

        # (Projectile abilities — fireball, rasengan — have their own custom
        # ticks above and never reach here.)

        # Forgiving fire for spread abilities (kamehameha): once fully charged,
        # release on a moderate spread OR on the pose dropping OR after a short
        # timeout, so the beam reliably fires instead of staying stuck charged.
        if ability.release_motion == "spread" and s.charge >= 1.0:
            pose_present = (
                match is not None and match.confidence >= config.POSE_MATCH_THRESHOLD
            )
            spread = signals.expansion > config.SPREAD_RELEASE_EXPANSION
            time_at_full = (
                s.age - self._charge_full_since
                if self._charge_full_since is not None
                else 0.0
            )
            timed_out = time_at_full >= config.KAMEHAMEHA_AUTO_FIRE_SECONDS
            if spread or not pose_present or timed_out:
                s.intensity = float(np.clip(s.charge, 0.0, 1.0))
                self.hooks.emit("ability_release", s.name, s.intensity, frame)
                self._spawn_projectile_if_needed(ability, frame)
                self._goto_active(frame, signals)
                return

        if match is None or match.confidence < config.POSE_MATCH_THRESHOLD:
            self._lost_frames += 1
            if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                self._goto_cooldown(reason="pose_lost", frame=frame)
            return
        self._lost_frames = 0
        s.primary_hand = match.primary
        s.secondary_hand = match.secondary

        # (rasengan is handled by _tick_rasengan above and never reaches here.)
        if ability.charge_time <= 0.0:
            s.charge = 1.0
        else:
            s.charge = float(min(1.0, s.charge + frame.dt / ability.charge_time))

        if s.charge >= 1.0 and self._charge_full_since is None:
            self._charge_full_since = s.age

        self.hooks.emit("ability_charge", s.name, s.charge, frame, signals)

        if s.charge >= 1.0:
            self._on_charge_complete(ability, frame, signals)

    def _advance_rasengan_charge(self, frame: FrameState) -> None:
        """Charge rasengan by ACCUMULATED rotation of the top hand about the
        lower (cupping) palm. Integrating angular travel means slow stirring
        still charges and a fast, motion-blurred hand never resets progress."""
        s = self.state
        center = s.primary_hand   # lower cupped palm (sphere anchor)
        spin = s.secondary_hand   # top hand stirring
        if center is not None and spin is not None:
            vec = np.asarray(spin.palm, dtype=np.float32) - np.asarray(
                center.palm, dtype=np.float32
            )
            r = float(np.linalg.norm(vec))
            if r >= config.RASENGAN_SPIN_MIN_RADIUS:
                angle = float(np.arctan2(float(vec[1]), float(vec[0])))
                if self._rasengan_prev_angle is not None:
                    d = angle - self._rasengan_prev_angle
                    while d > np.pi:
                        d -= 2.0 * np.pi
                    while d < -np.pi:
                        d += 2.0 * np.pi
                    self._rasengan_accum += abs(d)
                self._rasengan_prev_angle = angle
        s.charge = float(min(
            1.0, self._rasengan_accum / max(config.RASENGAN_SPIN_FULL_RADIANS, 1e-6)
        ))

    def _tick_rasengan(
        self,
        ability: AbilityDef,
        match: PoseMatch | None,
        frame: FrameState,
        signals: GestureSignals,
    ) -> None:
        """Rasengan charging with sticky persistence.

        - Full two-hand match present → advance charge by accumulated stirring.
        - Match lost but the lower cupped palm-up hand still visible → HOLD the
          charge in that hand (no reset, no cancel); the spin resumes when the
          top hand returns. This is what keeps the sphere "in the bottom hand"
          when the top hand momentarily drops out of tracking.
        - Neither present for the grace window → cancel to cooldown.
        - Once fully charged, a throw (fast move of the lower hand) launches it.
        """
        s = self.state
        pose_present = (
            match is not None and match.confidence >= config.POSE_MATCH_THRESHOLD
        )

        # 1. Keep primary_hand current BEFORE the throw check.
        if pose_present:
            self._lost_frames = 0
            s.primary_hand = match.primary
            s.secondary_hand = match.secondary
        else:
            # Top hand gone (or pose otherwise broke): hold the charge in the
            # lower cupped hand if it's still visible; only cancel once that hand
            # is gone for the grace window.
            lower = self._rasengan_lower_hand(frame)
            if lower is None:
                self._lost_frames += 1
                if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                    self._goto_cooldown(reason="pose_lost", frame=frame)
                return
            self._lost_frames = 0
            s.primary_hand = lower
            s.secondary_hand = None
            # Drop the previous-angle reference so a returning top hand doesn't
            # inject one giant angular delta on its first re-detected frame.
            self._rasengan_prev_angle = None

        # 2. Fire on a sudden flick/throw of the lower/anchor hand once fully
        #    charged. Like the fireball, the throw goes along the flick direction
        #    (the way the palm shoves); the projectile itself is slow (config).
        if s.charge >= 1.0 and s.primary_hand is not None:
            primary = s.primary_hand
            # Two paths: a captured flick (>= the 0.35 capture floor, gives a
            # clean throw direction) OR the live palm velocity crossing the low
            # RASENGAN_THROW_VELOCITY — the responsive path that lets a gentle
            # shove fire before the cupped-hand pose breaks.
            threw = (
                primary.flick_speed >= config.RASENGAN_THROW_FLICK_SPEED
                or float(np.linalg.norm(primary.velocity)) > config.RASENGAN_THROW_VELOCITY
            )
            if threw:
                s.intensity = float(np.clip(s.charge, 0.0, 1.0))
                self.hooks.emit("ability_release", s.name, s.intensity, frame)
                self._spawn_projectile_if_needed(ability, frame)
                self._goto_active(frame, signals)
                return

        # 3. Stir to charge — only while the full two-hand pose is present.
        if pose_present:
            self._advance_rasengan_charge(frame)
            if s.charge >= 1.0 and self._charge_full_since is None:
                self._charge_full_since = s.age
        self.hooks.emit("ability_charge", s.name, s.charge, frame, signals)

    @staticmethod
    def _rasengan_lower_hand(frame: FrameState) -> HandData | None:
        """The still-present hand that satisfies the rasengan lower-cup gate
        (open-ish and palm facing up). Prefers the one lower on screen."""
        best: HandData | None = None
        for h in frame.hands:
            if h.openness < config.RASENGAN_LOWER_OPEN_MIN:
                continue
            if -float(h.palm_normal[1]) < config.RASENGAN_PALM_UP_MIN:
                continue
            if best is None or float(h.palm[1]) > float(best.palm[1]):
                best = h
        return best

    def _tick_fireball(
        self,
        ability: AbilityDef,
        match: PoseMatch | None,
        frame: FrameState,
        signals: GestureSignals,
    ) -> None:
        """Fireball repeater: charge ONCE, then fire on every fast finger flick.

        - Pose lost for the grace window → cancel to cooldown.
        - Still charging → ramp the charge (the ember grows at the fingertip).
        - Loaded (charge == 1.0) → each time the index flick clears
          FIREBALL_FIRE_FLICK_SPEED (and the per-shot refire cooldown elapsed) a
          new projectile launches along the flick. The pose stays loaded, so you
          get unlimited shots without changing and re-forming the gesture.
        """
        s = self.state
        pose_present = (
            match is not None and match.confidence >= config.POSE_MATCH_THRESHOLD
        )
        if not pose_present:
            self._lost_frames += 1
            if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                self._goto_cooldown(reason="pose_lost", frame=frame)
            return
        self._lost_frames = 0
        s.primary_hand = match.primary
        s.secondary_hand = match.secondary

        if self._fireball_refire_timer > 0.0:
            self._fireball_refire_timer = max(0.0, self._fireball_refire_timer - frame.dt)

        # Still winding up the first charge.
        if s.charge < 1.0:
            if ability.charge_time <= 0.0:
                s.charge = 1.0
            else:
                s.charge = float(min(1.0, s.charge + frame.dt / ability.charge_time))
            if s.charge >= 1.0 and self._charge_full_since is None:
                self._charge_full_since = s.age
            # Emitting ability_charge drives the growing-ember visual + the ready
            # cue (fired once at 1.0). We deliberately stop emitting it once loaded
            # so the ready cue never replays per shot.
            self.hooks.emit("ability_charge", s.name, s.charge, frame, signals)
            return

        # Loaded: fire on a fast flick of the index finger. The PRIMARY signal is
        # the index-fingertip velocity — the fireball is thrown by flicking the
        # finger, which barely moves the palm, so palm velocity alone needed a
        # whole-hand shove. We still OR in the latched flick and palm velocity so
        # a whole-hand throw works too. EDGE-TRIGGERED so one flick = one shot:
        # flick_speed stays latched for HAND_FLICK_DECAY_SECONDS (0.40s) — longer
        # than the refire cooldown — so a time-only gate would fire twice from a
        # single flick. We disarm on fire and re-arm only once the level falls
        # back below REARM_FRACTION of the threshold (the flick has subsided).
        hand = s.primary_hand
        fire_level = 0.0
        if hand is not None:
            fire_level = max(
                float(np.linalg.norm(hand.index_tip_velocity)),
                float(hand.flick_speed),
                float(np.linalg.norm(hand.velocity)),
            )
        if fire_level < config.FIREBALL_FIRE_FLICK_SPEED * config.FIREBALL_REARM_FRACTION:
            self._fireball_armed = True
        can_fire = (
            fire_level >= config.FIREBALL_FIRE_FLICK_SPEED
            and self._fireball_armed
            and self._fireball_refire_timer <= 0.0
        )
        if can_fire:
            s.intensity = 1.0
            # ability_release plays the cast SFX + the muzzle puff for this shot
            # without leaving the loaded state (we stay in CHARGING).
            self.hooks.emit("ability_release", s.name, s.intensity, frame)
            self._spawn_projectile_if_needed(ability, frame)
            self._fireball_refire_timer = config.FIREBALL_REFIRE_COOLDOWN
            self._fireball_armed = False

    def _tick_reality_tear(
        self,
        ability: AbilityDef,
        match: PoseMatch | None,
        frame: FrameState,
        signals: GestureSignals,
    ) -> None:
        """reality_tear charging: advance only while the fists are together; open
        the tear when they're pulled apart past the threshold after full charge.
        A dead zone between TOGETHER_MAX and PULL_APART_DIST prevents instant fire.
        """
        s = self.state
        pose_present = (
            match is not None and match.confidence >= config.POSE_MATCH_THRESHOLD
        )
        if not pose_present:
            self._lost_frames += 1
            if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                self._goto_cooldown(reason="pose_lost", frame=frame)
            return
        self._lost_frames = 0
        s.primary_hand = match.primary
        s.secondary_hand = match.secondary
        palm_dist = float(match.extra.get("palm_dist", 0.0))

        if s.charge >= 1.0 and palm_dist >= config.REALITY_TEAR_PULL_APART_DIST:
            s.intensity = 1.0
            self.hooks.emit("ability_release", s.name, s.intensity, frame)
            self._spawn_projectile_if_needed(ability, frame)
            self._goto_active(frame, signals)
            return

        if palm_dist <= config.REALITY_TEAR_TOGETHER_MAX:
            if ability.charge_time <= 0.0:
                s.charge = 1.0
            else:
                s.charge = float(min(1.0, s.charge + frame.dt / ability.charge_time))
            self.hooks.emit("ability_charge", s.name, s.charge, frame, signals)
        # dead zone between together and pull-apart: hold the charge and wait.

    def _tick_active(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: dict[str, PoseMatch],
    ) -> None:
        s = self.state
        ability = self.abilities[s.name]

        if ability.active_duration <= 0.0:
            match = matches.get(ability.pose_id)
            if match is None or match.confidence < config.POSE_MATCH_THRESHOLD:
                self._lost_frames += 1
                if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                    self._goto_releasing(intensity=0.4, frame=frame)
                return
            self._lost_frames = 0
            s.primary_hand = match.primary
            s.secondary_hand = match.secondary
            self.hooks.emit("ability_active", s.name, frame, signals)
            return

        self.hooks.emit("ability_active", s.name, frame, signals)
        if s.phase_age >= ability.active_duration:
            self._goto_releasing(intensity=s.intensity * 0.5, frame=frame)

    def _tick_releasing(
        self,
        frame: FrameState,
        signals: GestureSignals,
    ) -> None:
        s = self.state
        # Time Freeze holds the frozen frame for TIME_FREEZE_SHATTER_DELAY after
        # movement is detected; then the glass shatters and the frame un-freezes
        # together (both ride the single ability_exit emitted on cooldown entry).
        # Every other ability releases quickly.
        hold = (
            config.TIME_FREEZE_SHATTER_DELAY
            if s.name == "time_freeze"
            else config.ABILITY_RELEASE_HOLD_SECONDS
        )
        if s.phase_age > hold:
            self._goto_cooldown(reason="release_done", frame=frame)

    def _tick_cooldown(self, frame: FrameState, signals: GestureSignals) -> None:
        s = self.state
        ability = self.abilities.get(s.name) if s.name else None
        cooldown = ability.cooldown if ability else 0.2
        if s.phase_age >= cooldown:
            self._goto_idle()

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def _on_charge_complete(
        self, ability: AbilityDef, frame: FrameState, signals: GestureSignals
    ) -> None:
        """Charge has reached 1.0. Continuous → ACTIVE; motion-release → wait;
        pose_release → wait for pose disappearance (handled in _tick_charging)."""
        s = self.state

        if ability.release_motion is None:
            s.intensity = 1.0
            self._goto_active(frame, signals)
            return

        if ability.release_motion == "pose_release":
            # Do NOT transition yet; _tick_charging checks for pose disappearance.
            return

        if self._release_triggered(ability, frame, signals):
            s.intensity = float(np.clip(s.charge, 0.0, 1.0))
            self.hooks.emit("ability_release", s.name, s.intensity, frame)
            self._spawn_projectile_if_needed(ability, frame)
            self._goto_active(frame, signals)

    def _release_triggered(
        self, ability: AbilityDef, frame: FrameState, signals: GestureSignals
    ) -> bool:
        """Return True when the ability's release motion has been detected."""
        if ability.release_motion == "thrust":
            hand = self.state.primary_hand
            if hand is None:
                return False
            return hand.palm_size_velocity > config.THRUST_RELEASE_RATE

        if ability.release_motion == "spread":
            return signals.expansion > config.SPREAD_RELEASE_EXPANSION

        if ability.release_motion == "throw":
            hand = self.state.primary_hand
            if hand is None:
                return False
            return float(np.linalg.norm(hand.velocity)) > config.THROW_RELEASE_SPEED

        return False

    def _goto_idle(self) -> None:
        prev = self.state.name
        already_exited = self._exit_emitted
        self.state = AbilityState()
        self._exit_emitted = False
        self._charge_full_since = None
        self._rasengan_accum = 0.0
        self._rasengan_prev_angle = None
        # Only emit if cooldown didn't already (the normal path always passes
        # through _goto_cooldown first; this covers the defensive direct-to-idle).
        if prev and not already_exited:
            self.hooks.emit("ability_exit", prev)
            log.info("router: EXIT %s", prev)

    def _goto_active(self, frame: FrameState, signals: GestureSignals) -> None:
        s = self.state
        s.phase = PHASE_ACTIVE
        s.phase_age = 0.0
        if s.intensity <= 0.0:
            s.intensity = float(np.clip(s.charge, 0.0, 1.0))
        log.info("router: ACTIVE %s (intensity=%.2f)", s.name, s.intensity)

    def _goto_releasing(self, intensity: float, frame: FrameState) -> None:
        s = self.state
        s.phase = PHASE_RELEASING
        s.phase_age = 0.0
        s.intensity = float(np.clip(intensity, 0.0, 1.0))

    def _goto_cooldown(self, reason: str, frame: FrameState) -> None:
        s = self.state
        prev = s.name
        s.phase = PHASE_COOLDOWN
        s.phase_age = 0.0
        s.charge = 0.0
        s.intensity = 0.0
        self._charge_full_since = None
        if prev and not self._exit_emitted:
            self._exit_emitted = True
            self.hooks.emit("ability_exit", prev)
            log.info("router: %s -> COOLDOWN (%s)", prev, reason)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_per_pose(matches: Iterable[PoseMatch]) -> dict[str, PoseMatch]:
        out: dict[str, PoseMatch] = {}
        for m in matches:
            cur = out.get(m.name)
            if cur is None or m.confidence > cur.confidence:
                out[m.name] = m
        return out

    def _pick_candidate(self, matches: dict[str, PoseMatch]) -> PoseMatch | None:
        """Pick the highest-confidence match that is mapped to a known ability.

        Generic neutral poses (open_palm) are *not* abilities; they still count
        as "tracked" for diagnostics but won't trigger CHARGING.
        """
        best: PoseMatch | None = None
        for ability in self.abilities.values():
            m = matches.get(ability.pose_id)
            if m is None:
                continue
            if m.confidence < config.POSE_MATCH_THRESHOLD:
                continue
            if best is None or m.confidence > best.confidence:
                best = m
        return best
