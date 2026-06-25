"""Laser-eyes sub-machine mixin for AbilityRouter.

Contains the face-driven charge→fire state machine for the laser_eyes ability.
Pulled out of router.py to keep that file under the 800-line ceiling.
All methods reference ``self.*`` and are mixed into ``AbilityRouter``.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    AbilityState,
    FrameState,
    GestureSignals,
)

log = logging.getLogger(__name__)


class _LaserMixin:
    """Mixin providing laser-eyes charge→fire logic for AbilityRouter."""

    def _update_laser(
        self, frame: FrameState, signals: GestureSignals
    ) -> bool:
        """Face-driven laser-eyes charge→fire state machine.

        off → close both eyes (past the blink grace) to begin charging; the
        charge ramps over LASER_EYES_CHARGE_SECONDS while the eyes stay shut and
        the charge whine plays. Reopening before full cancels. At full charge the
        laser turns on; OPEN YOUR EYES to aim/fire (it keeps firing through brief
        face dropouts at the last known position). Close both eyes again for
        LASER_EYES_OFF_BLINK_SECONDS to turn it off.

        Returns True when the laser owns the ability slot this frame, so the
        caller skips the normal single-slot machine.
        """
        face = frame.face
        present = face is not None and getattr(face, "present", False)
        closed = bool(present and face.both_eyes_closed)
        dur = float(face.eyes_closed_duration) if present else 0.0
        open_now = present and not closed

        state = self._laser_state

        if state == "off":
            # Arm only once the eyes are open, so a close that's already in
            # progress (e.g. right after turning off) doesn't immediately recharge.
            if open_now:
                self._laser_armed = True
            if (
                closed
                and self._laser_armed
                and dur >= config.LASER_EYES_BLINK_GRACE_SECONDS
            ):
                self._begin_laser_charge(frame, signals)
                state = self._laser_state = "charging"
            else:
                return False

        if state == "charging":
            if not closed:
                # Eyes reopened before full charge → cancel (cuts the whine).
                self._cancel_laser_charge(frame)
                self._laser_state = "off"
                self._laser_armed = True       # eyes are open now
                return True
            charge = self._laser_charge_fraction(dur)
            s = self.state
            s.name = "laser_eyes"
            s.phase = PHASE_CHARGING
            s.charge = charge
            s.intensity = 0.0
            self.hooks.emit("ability_charge", "laser_eyes", charge, frame, signals)
            if charge >= 1.0:
                self._activate_laser(frame)
                self._laser_state = "on"
            return True

        if state == "on":
            # The off-close only counts once the eyes have reopened after firing,
            # so the eyes you held shut to charge don't instantly switch it off.
            if open_now:
                self._laser_off_armed = True
            if (
                self._laser_off_armed
                and closed
                and dur >= config.LASER_EYES_OFF_BLINK_SECONDS
            ):
                self._stop_laser(frame)
                self._laser_state = "off"
                self._laser_armed = False       # require a fresh open before recharge
                return True
            s = self.state
            s.name = "laser_eyes"
            s.phase = PHASE_ACTIVE
            s.charge = 1.0
            s.intensity = 1.0
            # Re-cache the (possibly absent) face on the effect each frame; when
            # the face is gone the effect keeps firing at the last known position.
            self.hooks.emit("ability_active", "laser_eyes", frame, signals)
            return True

        return False

    @staticmethod
    def _laser_charge_fraction(dur: float) -> float:
        """Charge 0..1 from the eyes-closed duration, ignoring the blink grace so
        the ramp (and the matching charge sound) starts the moment charging does."""
        grace = config.LASER_EYES_BLINK_GRACE_SECONDS
        span = max(config.LASER_EYES_CHARGE_SECONDS, 1e-6)
        return float(np.clip((dur - grace) / span, 0.0, 1.0))

    def _begin_laser_charge(
        self, frame: FrameState, signals: GestureSignals
    ) -> None:
        self.state = AbilityState()
        s = self.state
        s.name = "laser_eyes"
        s.phase = PHASE_CHARGING
        s.charge = 0.0
        self._exit_emitted = False
        self._charge_full_since = None
        self._laser_off_armed = False
        # enter starts the charge whine (a play-once 1s build); the effect caches
        # the face and renders the eye charge-glow.
        self.hooks.emit("ability_enter", "laser_eyes", frame, signals)
        log.info("router: LASER EYES charging")

    def _cancel_laser_charge(self, frame: FrameState) -> None:
        self._goto_cooldown(reason="laser_charge_canceled", frame=frame)
        log.info("router: LASER EYES charge canceled")

    def _activate_laser(self, frame: FrameState) -> None:
        s = self.state
        s.phase = PHASE_ACTIVE
        s.phase_age = 0.0
        s.charge = 1.0
        s.intensity = 1.0
        self._laser_off_armed = False
        # release stops the charge whine and plays the cast zap.
        self.hooks.emit("ability_release", "laser_eyes", 1.0, frame)
        log.info("router: LASER EYES on")

    def _stop_laser(self, frame: FrameState) -> None:
        self._goto_cooldown(reason="laser_off", frame=frame)
        log.info("router: LASER EYES off")
