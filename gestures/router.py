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

Effects can subscribe to any of these. They can also read `router.state`
each frame for a richer snapshot (charge, age, primary_hand, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

import config
from core.hooks import HookBus
from core.state import (
    AbilityState,
    FrameState,
    GestureSignals,
    HandData,
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_COOLDOWN,
    PHASE_IDLE,
    PHASE_RELEASING,
)
from gestures.poses import PoseMatch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AbilityDef:
    """Static configuration for a single ability slot."""

    name: str
    pose_id: str
    charge_time: float
    cooldown: float
    active_duration: float
    release_motion: Optional[str]   # "thrust" | "spread" | None for continuous

    @classmethod
    def from_config(cls, name: str, pose_id: str, release_motion: Optional[str]) -> "AbilityDef":
        return cls(
            name=name,
            pose_id=pose_id,
            charge_time=config.ABILITY_CHARGE_TIME[name],
            cooldown=config.ABILITY_COOLDOWN[name],
            active_duration=config.ABILITY_ACTIVE_DURATION[name],
            release_motion=release_motion,
        )


def default_abilities() -> dict[str, AbilityDef]:
    """The canonical ability roster Aether ships with."""
    return {
        a.name: a for a in (
            AbilityDef.from_config("chidori", "chidori", "thrust"),
            AbilityDef.from_config("kamehameha", "kamehameha", "spread"),
            AbilityDef.from_config("rasengan", "rasengan", "thrust"),
            AbilityDef.from_config("space_stretch", "space_stretch", None),
            AbilityDef.from_config("reality_tear", "reality_tear", None),
        )
    }


@dataclass
class _ReleaseDetector:
    """Tracks the motion required to transition CHARGING -> ACTIVE."""

    primary_palm_size_velocity: float = 0.0
    expansion: float = 0.0


class AbilityRouter:
    """Single-active-ability state machine.

    Lifetime is per-process; keep one instance.
    """

    def __init__(
        self,
        hooks: HookBus,
        abilities: Optional[dict[str, AbilityDef]] = None,
    ) -> None:
        self.hooks = hooks
        self.abilities = abilities if abilities is not None else default_abilities()
        self.state = AbilityState()
        self._lost_frames = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: Iterable[PoseMatch],
    ) -> AbilityState:
        s = self.state
        s.age += frame.dt
        s.phase_age += frame.dt

        match_by_pose = self._best_per_pose(matches)

        if s.phase == PHASE_IDLE:
            self._tick_idle(frame, signals, match_by_pose)
        elif s.phase == PHASE_CHARGING:
            self._tick_charging(frame, signals, match_by_pose)
        elif s.phase == PHASE_ACTIVE:
            self._tick_active(frame, signals, match_by_pose)
        elif s.phase == PHASE_RELEASING:
            self._tick_releasing(frame, signals, match_by_pose)
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
        # Find the highest-confidence match that's both above threshold
        # *and* maps to a known ability.
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

        if match is None or match.confidence < config.POSE_MATCH_THRESHOLD:
            self._lost_frames += 1
            if self._lost_frames > config.POSE_LOST_GRACE_FRAMES:
                self._goto_cooldown(reason="pose_lost", frame=frame)
            return
        self._lost_frames = 0
        s.primary_hand = match.primary
        s.secondary_hand = match.secondary

        # Continuous charge: ramp 0 -> 1 across charge_time. If charge_time
        # is zero we go straight to fully charged on the first frame.
        if ability.charge_time <= 0.0:
            s.charge = 1.0
        else:
            s.charge = float(min(1.0, s.charge + frame.dt / ability.charge_time))

        self.hooks.emit("ability_charge", s.name, s.charge, frame, signals)

        if s.charge >= 1.0:
            self._on_charge_complete(ability, frame, signals)

    def _tick_active(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: dict[str, PoseMatch],
    ) -> None:
        s = self.state
        ability = self.abilities[s.name]

        # Continuous abilities: stay alive while pose holds; otherwise exit.
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

        # Timed active phase (e.g. Kamehameha beam holds for 1.5s).
        self.hooks.emit("ability_active", s.name, frame, signals)
        if s.phase_age >= ability.active_duration:
            self._goto_releasing(intensity=s.intensity * 0.5, frame=frame)

    def _tick_releasing(
        self,
        frame: FrameState,
        signals: GestureSignals,
        matches: dict[str, PoseMatch],
    ) -> None:
        s = self.state
        # Releasing is short — 0.25s — then drop into cooldown. Effects
        # use the phase to play their fade-out animation.
        if s.phase_age > 0.25:
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
        """Charge has reached 1.0. Either wait for the release motion or
        transition straight into ACTIVE for continuous abilities."""
        s = self.state
        if ability.release_motion is None:
            # Continuous ability — go straight to ACTIVE.
            s.intensity = 1.0
            self._goto_active(frame, signals)
            return
        # Wait for the release motion. We *stay* in CHARGING with charge=1.0
        # until the motion fires or the pose drops.
        if self._release_triggered(ability, frame, signals):
            s.intensity = float(np.clip(s.charge, 0.0, 1.0))
            self.hooks.emit("ability_release", s.name, s.intensity, frame)
            self._goto_active(frame, signals)

    def _release_triggered(
        self, ability: AbilityDef, frame: FrameState, signals: GestureSignals
    ) -> bool:
        if ability.release_motion == "thrust":
            hand = self.state.primary_hand
            if hand is None:
                return False
            return hand.palm_size_velocity > config.THRUST_RELEASE_RATE
        if ability.release_motion == "spread":
            return signals.expansion > config.SPREAD_RELEASE_EXPANSION
        return False

    def _goto_idle(self) -> None:
        prev = self.state.name
        self.state = AbilityState()
        if prev:
            self.hooks.emit("ability_exit", prev)
            log.info("router: EXIT %s", prev)

    def _goto_active(self, frame: FrameState, signals: GestureSignals) -> None:
        s = self.state
        s.phase = PHASE_ACTIVE
        s.phase_age = 0.0
        # If release_motion was None we already set intensity above. Otherwise
        # set it here so the active phase can use it.
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
        if prev:
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

    def _pick_candidate(self, matches: dict[str, PoseMatch]) -> Optional[PoseMatch]:
        """Pick the highest-confidence match that is mapped to an ability
        we know about. Generic neutral poses (open_palm) are *not* abilities;
        they still count as "tracked" for diagnostics but won't enter."""
        best: Optional[PoseMatch] = None
        for ability in self.abilities.values():
            m = matches.get(ability.pose_id)
            if m is None:
                continue
            if m.confidence < config.POSE_MATCH_THRESHOLD:
                continue
            if best is None or m.confidence > best.confidence:
                best = m
        return best
