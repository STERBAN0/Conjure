"""SoundManager — hook-driven ability SFX playback.

Subscribes to the HookBus events emitted by gestures/router.py and plays
per-ability sounds using pygame.mixer.  All playback is fire-and-forget; the
class never raises into the event loop.

Lifecycle:
    ability_enter  → start looping the charge sound on a dedicated channel
    ability_charge → once charge reaches 1.0, play the ready cue exactly once
    ability_release→ play the cast sound, stop the charge loop
    ability_exit   → stop the charge loop and reset per-activation state

WAV files are loaded from ``config.SOUND_SFX_DIR`` at construction time.
Missing files are silently skipped (prints a warning, continues).

Graceful degradation mirrors audio/analyzer.py:
  - If pygame.mixer is not initialised (no display server, headless CI, …)
    the SoundManager sets ``_enabled = False`` and becomes a no-op.
  - Every public method is wrapped with a try/except so a runtime glitch can
    never propagate out through the HookBus.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import pygame  # type: ignore[import-untyped]
    import pygame.mixer  # type: ignore[import-untyped]

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

import config

if TYPE_CHECKING:
    from core.hooks import HookBus

log = logging.getLogger(__name__)

# Ability names that can have sound cues
_ABILITY_NAMES: tuple[str, ...] = (
    "fireball",
    "rasengan",
    "chidori",
    "time_freeze",
    "laser_eyes",
    "kamehameha",
    "space_stretch",
    "reality_tear",
    "frost_nova",
)

# Channel reserved for charge loops (always the last allocated channel)
_CHARGE_CHANNEL_INDEX = config.SOUND_MIXER_CHANNELS - 1
# Channel reserved for the chidori voice clip, so it can be stopped on exit
# independently of the looping electric-crackle charge sound.
_CHIDORI_VOICE_CHANNEL_INDEX = config.SOUND_MIXER_CHANNELS - 2

# Abilities whose "charge" cue is a one-shot build (not a seamless loop) and that
# culminate on their own — so we play it exactly once and suppress the separate
# "ready" stinger. time_freeze uses the JoJo-style ticking-clock build that ends
# in silence right as the screen freezes; laser_eyes uses a rising whine the same
# length as LASER_EYES_CHARGE_SECONDS, so the sound finishing == "open your eyes".
_PLAY_ONCE_CHARGE: frozenset[str] = frozenset({"time_freeze", "laser_eyes"})


def _resolve_sfx_dir(sfx_dir: str) -> Path:
    """Resolve the configured SFX directory to an absolute path.

    ``config.SOUND_SFX_DIR`` is a repo-relative path ("audio/sfx"). Resolving it
    against the *current working directory* silently breaks whenever the app is
    launched from anywhere other than the repo root — an editable/console-script
    install (``conjure`` run from the home folder), an IDE "Run" button, etc.:
    every WAV then fails to load and the app runs with no sound at all.

    Anchor it to the package root instead — the same ``__file__``-based approach
    ``vision/hand_tracker.py`` uses to locate the model files — so audio works
    regardless of the working directory. Absolute overrides are honoured as-is.
    """
    path = Path(sfx_dir)
    if path.is_absolute():
        return path
    # this file is <repo_root>/audio/sounds.py → parents[1] is <repo_root>
    return Path(__file__).resolve().parents[1] / path


def _clamp01(value: float) -> float:
    """Clamp a volume level to the inclusive 0.0–1.0 range."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _ability_gain(ability_name: str) -> float:
    """Per-ability baseline gain from config (1.0 when unconfigured)."""
    gains = getattr(config, "SOUND_ABILITY_GAIN", {})
    return float(gains.get(ability_name, 1.0))


class SoundManager:
    """Plays ability sound cues driven by HookBus events.

    Parameters
    ----------
    hooks:
        The application-wide HookBus instance.  Subscriptions are registered
        during ``__init__``; no cleanup is needed on shutdown.
    """

    def __init__(self, hooks: HookBus) -> None:
        self._enabled: bool = False
        self._sounds: dict[str, pygame.mixer.Sound | None] = {}
        self._charge_channel: pygame.mixer.Channel | None = None
        # Dedicated channel for the chidori voice clip (the 8s "Chidori!" sound).
        self._chidori_voice_channel: pygame.mixer.Channel | None = None
        self._oneshots: dict[str, pygame.mixer.Sound | None] = {}

        # Live-adjustable master volume + mute, driven by the in-app Options
        # panel (O key). Kept separate from the per-Sound volumes so unmuting
        # restores the exact previous level. Safe to read/set even when the
        # mixer is disabled — the setters become no-ops with no loaded sounds.
        self._master_volume: float = _clamp01(
            float(getattr(config, "SOUND_MASTER_VOLUME", 0.8))
        )
        self._muted: bool = False
        # Per-sound baseline gain, keyed exactly like _sounds / _oneshots, so the
        # applied volume is master × gain. Populated by _load_sounds.
        self._gains: dict[str, float] = {}

        # Track per-activation ready-cue state keyed by ability name.
        # Reset on ability_enter and ability_exit.
        self._ready_played: dict[str, bool] = {}

        if not getattr(config, "SOUND_ENABLED", True):
            log.info("SoundManager: SOUND_ENABLED=False — sound disabled by config")
            return

        if not _PYGAME_AVAILABLE:
            log.warning("SoundManager: pygame not available — running silent")
            return

        try:
            if not pygame.mixer.get_init():
                log.warning("SoundManager: mixer not initialised — running silent")
                return
            self._charge_channel = pygame.mixer.Channel(_CHARGE_CHANNEL_INDEX)
            self._chidori_voice_channel = pygame.mixer.Channel(
                _CHIDORI_VOICE_CHANNEL_INDEX
            )
            self._load_sounds()
            self._enabled = True
            log.info("SoundManager: initialised OK (%d cues loaded)", len(self._sounds))
        except Exception as exc:  # noqa: BLE001
            log.warning("SoundManager: init error (%s) — running silent", exc)

        hooks.on("ability_enter", self._on_enter)
        hooks.on("ability_charge", self._on_charge)
        hooks.on("ability_release", self._on_release)
        hooks.on("ability_exit", self._on_exit)

    # ------------------------------------------------------------------
    # Public volume / mute API (driven by the Options panel)
    # ------------------------------------------------------------------

    @property
    def master_volume(self) -> float:
        """Current master SFX volume, 0.0–1.0 (independent of mute state)."""
        return self._master_volume

    @property
    def is_muted(self) -> bool:
        return self._muted

    def set_master_volume(self, volume: float) -> None:
        """Set the master SFX volume (clamped 0–1) and apply it live."""
        self._master_volume = _clamp01(float(volume))
        self._apply_volume()

    def set_muted(self, muted: bool) -> None:
        """Mute or unmute all SFX without losing the chosen volume level."""
        self._muted = bool(muted)
        self._apply_volume()

    def toggle_muted(self) -> bool:
        """Flip the mute state; return the new state."""
        self.set_muted(not self._muted)
        return self._muted

    def _effective_volume(self) -> float:
        """The volume actually applied to Sounds: 0 while muted, else master."""
        return 0.0 if self._muted else self._master_volume

    def _apply_volume(self) -> None:
        """Push each Sound's effective volume (master × per-ability gain).

        Safe to call with no loaded sounds — it simply iterates nothing.
        """
        master = self._effective_volume()
        for key, snd in (*self._sounds.items(), *self._oneshots.items()):
            if snd is None:
                continue
            try:
                snd.set_volume(_clamp01(master * self._gains.get(key, 1.0)))
            except Exception as exc:  # noqa: BLE001
                log.debug("SoundManager: set_volume failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_sounds(self) -> None:
        """Load all WAV files from the SFX directory into pygame Sound objects.

        Each sound's initial volume is the effective master scaled by its
        ability's baseline gain (config.SOUND_ABILITY_GAIN); the gain is recorded
        in ``_gains`` so _apply_volume can re-scale live when the slider moves.
        """
        sfx_dir = _resolve_sfx_dir(config.SOUND_SFX_DIR)
        master = self._effective_volume()

        for ability in _ABILITY_NAMES:
            gain = _ability_gain(ability)
            for cue in ("charge", "ready", "cast"):
                key = f"{ability}_{cue}"
                wav_path = sfx_dir / f"{key}.wav"
                self._gains[key] = gain
                # None if missing — that's fine
                self._sounds[key] = self._load_wav(wav_path, _clamp01(master * gain))

        # Load one-shot event cues, each tied to its parent ability's gain.
        # ``chidori_voice`` is the real 8s "Chidori!" clip (converted from
        # chidori_sound.mp3); the looping electric crackle is the generated
        # chidori_charge cue, which keeps playing after it ends.
        oneshot_ability = {"time_shatter": "time_freeze", "chidori_voice": "chidori"}
        for name, ability in oneshot_ability.items():
            gain = _ability_gain(ability)
            wav_path = sfx_dir / f"{name}.wav"
            self._gains[name] = gain
            self._oneshots[name] = self._load_wav(wav_path, _clamp01(master * gain))

        # If nothing loaded, the app would run completely silent — surface why and
        # where we looked, rather than leaving the user guessing.
        if not any(self._sounds.values()) and not any(self._oneshots.values()):
            log.warning(
                "SoundManager: no SFX loaded from %s — the app will run silent. "
                "Expected the committed audio/sfx/*.wav files there.",
                sfx_dir,
            )

    def _load_wav(self, path: Path, volume: float) -> pygame.mixer.Sound | None:
        """Load a single WAV; return None (with warning) on failure."""
        if not path.exists():
            log.warning("SoundManager: WAV not found: %s", path)
            return None
        try:
            snd = pygame.mixer.Sound(str(path))
            snd.set_volume(volume)
            return snd
        except Exception as exc:  # noqa: BLE001
            log.warning("SoundManager: failed to load %s: %s", path, exc)
            return None

    def _get(self, ability: str, cue: str) -> pygame.mixer.Sound | None:
        return self._sounds.get(f"{ability}_{cue}")

    def _play_oneshot(self, sound: pygame.mixer.Sound | None) -> None:
        """Play a one-shot sound on any free channel."""
        if sound is None:
            return
        try:
            sound.play()
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager: play error: %s", exc)

    def _start_charge_loop(self, ability: str) -> None:
        """Start the charge sound on the dedicated charge channel.

        Most abilities loop the charge cue seamlessly (-1); play-once abilities
        (see _PLAY_ONCE_CHARGE) play their build a single time so it culminates
        on its own rather than restarting.
        """
        if self._charge_channel is None:
            return
        sound = self._get(ability, "charge")
        if sound is None:
            return
        loops = 0 if ability in _PLAY_ONCE_CHARGE else -1
        try:
            self._charge_channel.play(sound, loops=loops)
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager: charge loop error: %s", exc)

    def _stop_charge_loop(self) -> None:
        """Stop the charge channel immediately."""
        if self._charge_channel is None:
            return
        try:
            self._charge_channel.stop()
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager: stop error: %s", exc)

    def _start_chidori_voice(self) -> None:
        """Play the 8s chidori voice clip once on its dedicated channel."""
        if self._chidori_voice_channel is None:
            return
        sound = self._oneshots.get("chidori_voice")
        if sound is None:
            return
        try:
            self._chidori_voice_channel.play(sound, loops=0)
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager: chidori voice error: %s", exc)

    def _stop_chidori_voice(self) -> None:
        """Stop the chidori voice clip (e.g. when the sign is dropped early)."""
        if self._chidori_voice_channel is None:
            return
        try:
            self._chidori_voice_channel.stop()
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager: chidori voice stop error: %s", exc)

    # ------------------------------------------------------------------
    # HookBus event handlers
    # ------------------------------------------------------------------

    def _on_enter(self, ability_name: str, frame: object, signals: object) -> None:
        """ability_enter(ability_name, frame, signals)"""
        if not self._enabled:
            return
        try:
            self._ready_played[ability_name] = False
            self._stop_charge_loop()
            self._start_charge_loop(ability_name)
            # Chidori: play the real 8s "Chidori!" clip once over the looping
            # electric crackle (the charge cue). The crackle keeps going after the
            # clip ends, so the lightning still sounds alive while the V is held.
            if ability_name == "chidori":
                self._start_chidori_voice()
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager._on_enter error: %s", exc)

    def _on_charge(
        self,
        ability_name: str,
        charge: float,
        frame: object,
        signals: object,
    ) -> None:
        """ability_charge(ability_name, charge, frame, signals)

        Fire the ready cue exactly once when charge reaches 1.0.
        """
        if not self._enabled:
            return
        if not getattr(config, "SOUND_READY_CUE_ENABLED", True):
            return
        if ability_name in _PLAY_ONCE_CHARGE:
            # The one-shot build is the cue; no separate ready stinger.
            return
        try:
            if charge >= 1.0 and not self._ready_played.get(ability_name, False):
                self._ready_played[ability_name] = True
                self._play_oneshot(self._get(ability_name, "ready"))
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager._on_charge error: %s", exc)

    def _on_release(self, ability_name: str, intensity: float, frame: object) -> None:
        """ability_release(ability_name, intensity, frame)"""
        if not self._enabled:
            return
        try:
            self._stop_charge_loop()
            self._play_oneshot(self._get(ability_name, "cast"))
            self._ready_played.pop(ability_name, None)
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager._on_release error: %s", exc)

    def _on_exit(self, ability_name: str) -> None:
        """ability_exit(ability_name)"""
        if not self._enabled:
            return
        try:
            self._stop_charge_loop()
            self._ready_played.pop(ability_name, None)
            # Play glass shatter sound when time_freeze ends
            if ability_name == "time_freeze":
                self._play_oneshot(self._oneshots.get("time_shatter"))
            # Cut the chidori voice clip when the sign is dropped.
            if ability_name == "chidori":
                self._stop_chidori_voice()
        except Exception as exc:  # noqa: BLE001
            log.debug("SoundManager._on_exit error: %s", exc)
