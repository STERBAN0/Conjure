"""System-level controls driven by gestures.

Currently: master volume on Windows via pycaw. Engagement gating is
strict on purpose — accidental volume changes are deeply annoying:

    A single open hand, palm centred-ish horizontally, motion energy
    near zero, no Aether ability currently in flight, and the
    *other* hand absent (or also stationary) for a sustained moment.

When engaged, hand vertical position maps to volume:
  - hand near top of frame   -> max volume
  - hand near bottom of frame -> min volume

Volume target is EMA-smoothed and pushed to the OS only when it crosses
a dead-zone, which avoids spamming the audio service.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

import config
from core.state import AbilityState, FrameState, GestureSignals, PHASE_IDLE
from gestures.smoothing import EMA

log = logging.getLogger(__name__)

try:
    # pycaw 2025+ exposes EndpointVolume directly off the device.
    from pycaw.pycaw import AudioUtilities

    _PYCAW_OK = True
    _PYCAW_ERR = ""
except Exception as _e:
    _PYCAW_OK = False
    _PYCAW_ERR = repr(_e)


class SystemControls:
    def __init__(self) -> None:
        self._volume_iface = None
        self._volume_ema = EMA(alpha=config.VOLUME_SMOOTH_ALPHA, init=0.5)
        self._last_pushed: Optional[float] = None
        self._engaged_since: Optional[float] = None
        self._engagement_grace = 0.4
        self.engaged: bool = False
        self.current_volume: float = 0.5

        if not config.SYSTEM_CONTROLS_ENABLED:
            log.info("system controls disabled in config")
            return
        if not _PYCAW_OK:
            log.info("system controls disabled: pycaw unavailable (%s)", _PYCAW_ERR)
            return

        try:
            self._volume_iface = AudioUtilities.GetSpeakers().EndpointVolume
            current = float(self._volume_iface.GetMasterVolumeLevelScalar())
            self._volume_ema.reset(current)
            self.current_volume = current
        except Exception as e:
            log.warning("could not acquire audio endpoint: %r", e)
            self._volume_iface = None

    def update(
        self,
        frame: FrameState,
        signals: GestureSignals,
        ability: Optional[AbilityState] = None,
    ) -> None:
        if self._volume_iface is None:
            return

        # Suspend volume gesture whenever an ability is in flight.
        if ability is not None and ability.phase != PHASE_IDLE:
            self._engaged_since = None
            self.engaged = False
            return

        hands = frame.hands
        candidate = None
        if len(hands) == 1:
            candidate = hands[0]
        elif len(hands) == 2:
            opens = sorted(hands, key=lambda h: h.openness, reverse=True)
            if opens[0].openness > config.VOLUME_GESTURE_OPENNESS \
               and opens[1].openness < 0.3:
                candidate = opens[0]

        engaging = (
            candidate is not None
            and candidate.openness >= config.VOLUME_GESTURE_OPENNESS
            and signals.motion_energy < config.VOLUME_GESTURE_STILLNESS * 8
        )

        now = frame.timestamp
        if engaging:
            if self._engaged_since is None:
                self._engaged_since = now
            self.engaged = (now - self._engaged_since) >= self._engagement_grace
        else:
            self._engaged_since = None
            self.engaged = False

        if not self.engaged or candidate is None:
            return

        target = float(np.clip(1.0 - candidate.palm[1], 0.0, 1.0))
        smoothed = self._volume_ema(target)
        self.current_volume = smoothed

        if (
            self._last_pushed is None
            or abs(smoothed - self._last_pushed) >= config.VOLUME_DEAD_ZONE
        ):
            try:
                self._volume_iface.SetMasterVolumeLevelScalar(float(smoothed), None)
                self._last_pushed = smoothed
            except Exception as e:
                log.debug("volume push failed: %r", e)
