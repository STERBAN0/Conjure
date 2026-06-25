"""Tests for audio/sounds.py — SoundManager hook integration.

Strategy:
- Patch pygame.mixer at import time so tests are headless-safe.
- Use the real HookBus to exercise the full pub/sub path.
- Assert channel and sound interactions via Mock call counts / call args.
- The ready cue must fire exactly once when charge >= 1.0, even if the
  ability_charge event fires repeatedly above 1.0.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pygame_mixer() -> types.ModuleType:
    """Return a minimal fake pygame.mixer module.

    All Sound objects and Channel objects are Mocks so we can introspect them.
    """
    mixer_mod = MagicMock()
    # get_init() → truthy so SoundManager thinks the mixer is ready
    mixer_mod.get_init.return_value = (44100, -16, 2)

    # Channel() returns a fresh Mock each time; we give it a stable reference
    channel_mock = MagicMock()
    mixer_mod.Channel.return_value = channel_mock

    # Sound() returns a fresh Mock
    mixer_mod.Sound.return_value = MagicMock()

    return mixer_mod, channel_mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_mixer():
    """Patch pygame and pygame.mixer before importing SoundManager."""
    mixer_mod, channel_mock = _make_mock_pygame_mixer()

    # We need to patch at the level that audio.sounds imports from.
    with (
        patch.dict("sys.modules", {"pygame": MagicMock(), "pygame.mixer": mixer_mod}),
        patch("audio.sounds.pygame.mixer", mixer_mod),
        patch("audio.sounds._PYGAME_AVAILABLE", True),
    ):
        yield mixer_mod, channel_mock


@pytest.fixture()
def hooks():
    from core.hooks import HookBus
    return HookBus()


@pytest.fixture()
def sound_manager(hooks, mock_mixer):
    """Construct a SoundManager wired to a real HookBus; mixer is mocked."""
    # Re-import sounds module inside the patch context established by mock_mixer
    from audio.sounds import SoundManager

    mixer_mod, channel_mock = mock_mixer
    # Make Sound load always succeed (file existence check bypassed via mock)
    with patch("audio.sounds.Path.exists", return_value=True):
        sm = SoundManager(hooks)

    # Verify it actually initialised (not in silent mode)
    assert sm._enabled, "SoundManager should be enabled with mocked mixer"
    return sm, hooks, channel_mock


# ---------------------------------------------------------------------------
# Tests: charge loop lifecycle
# ---------------------------------------------------------------------------

class TestChargeLoop:
    def test_enter_starts_charge_loop(self, sound_manager):
        sm, hooks, channel = sound_manager
        # rasengan is a plain looping-charge ability (chidori is special — it also
        # plays the 8s voice clip, so it would call play() twice).
        hooks.emit("ability_enter", "rasengan", object(), object())
        channel.play.assert_called_once()
        # loops=-1 for infinite looping
        _, kwargs = channel.play.call_args
        assert kwargs.get("loops", channel.play.call_args[0][1] if len(channel.play.call_args[0]) > 1 else None) == -1 or \
               channel.play.call_args == call(sm._get("rasengan", "charge"), loops=-1)

    def test_chidori_plays_voice_clip_and_crackle_on_enter(self, sound_manager):
        """Chidori enter plays the looping electric crackle AND the 8s voice clip."""
        sm, hooks, channel = sound_manager
        assert sm._oneshots.get("chidori_voice") is not None
        channel.reset_mock()
        hooks.emit("ability_enter", "chidori", object(), object())
        # Both reserved channels (charge crackle + voice) play; the mocked mixer
        # hands back the same channel object, so that's two play() calls.
        assert channel.play.call_count == 2

    def test_chidori_voice_stops_on_exit(self, sound_manager):
        sm, hooks, channel = sound_manager
        hooks.emit("ability_enter", "chidori", object(), object())
        channel.reset_mock()
        hooks.emit("ability_exit", "chidori")
        # Both the crackle loop and the voice clip are stopped.
        assert channel.stop.call_count == 2

    def test_release_stops_charge_loop(self, sound_manager):
        sm, hooks, channel = sound_manager
        hooks.emit("ability_enter", "rasengan", object(), object())
        channel.reset_mock()
        hooks.emit("ability_release", "rasengan", 1.0, object())
        channel.stop.assert_called_once()

    def test_exit_stops_charge_loop(self, sound_manager):
        sm, hooks, channel = sound_manager
        hooks.emit("ability_enter", "fireball", object(), object())
        channel.reset_mock()
        hooks.emit("ability_exit", "fireball")
        channel.stop.assert_called_once()

    def test_second_enter_restarts_charge_loop(self, sound_manager):
        sm, hooks, channel = sound_manager
        hooks.emit("ability_enter", "kamehameha", object(), object())
        assert channel.play.call_count == 1
        # Exit then re-enter
        hooks.emit("ability_exit", "kamehameha")
        hooks.emit("ability_enter", "kamehameha", object(), object())
        assert channel.play.call_count == 2


# ---------------------------------------------------------------------------
# Tests: ready cue fires exactly once
# ---------------------------------------------------------------------------

class TestReadyCue:
    def test_ready_cue_fires_once_at_full_charge(self, sound_manager):
        sm, hooks, channel = sound_manager
        # frost_nova is a plain looping-charge ability with a ready stinger.
        # (laser_eyes no longer has one — its charge build is a play-once cue.)
        ready_sound = sm._get("frost_nova", "ready")
        assert ready_sound is not None

        hooks.emit("ability_enter", "frost_nova", object(), object())
        ready_sound.reset_mock()

        # Charge below threshold → should not fire
        hooks.emit("ability_charge", "frost_nova", 0.5, object(), object())
        ready_sound.play.assert_not_called()

        hooks.emit("ability_charge", "frost_nova", 0.99, object(), object())
        ready_sound.play.assert_not_called()

        # Exactly at threshold → should fire
        hooks.emit("ability_charge", "frost_nova", 1.0, object(), object())
        assert ready_sound.play.call_count == 1

    def test_ready_cue_fires_only_once_above_threshold(self, sound_manager):
        sm, hooks, channel = sound_manager
        ready_sound = sm._get("frost_nova", "ready")

        hooks.emit("ability_enter", "frost_nova", object(), object())
        ready_sound.reset_mock()

        # Multiple charge events above 1.0 — ready cue must still be exactly once
        for charge_val in (1.0, 1.0, 1.0, 1.0):
            hooks.emit("ability_charge", "frost_nova", charge_val, object(), object())

        assert ready_sound.play.call_count == 1, (
            "ready cue must fire exactly once per activation, not multiple times"
        )

    def test_ready_cue_resets_after_release(self, sound_manager):
        sm, hooks, channel = sound_manager
        ready_sound = sm._get("chidori", "ready")

        # First activation
        hooks.emit("ability_enter", "chidori", object(), object())
        ready_sound.reset_mock()
        hooks.emit("ability_charge", "chidori", 1.0, object(), object())
        assert ready_sound.play.call_count == 1

        # Release resets the gate
        hooks.emit("ability_release", "chidori", 1.0, object())

        # Second activation — ready cue should fire again
        hooks.emit("ability_enter", "chidori", object(), object())
        ready_sound.reset_mock()
        hooks.emit("ability_charge", "chidori", 1.0, object(), object())
        assert ready_sound.play.call_count == 1

    def test_ready_cue_resets_after_exit(self, sound_manager):
        sm, hooks, channel = sound_manager
        ready_sound = sm._get("rasengan", "ready")

        hooks.emit("ability_enter", "rasengan", object(), object())
        ready_sound.reset_mock()
        hooks.emit("ability_charge", "rasengan", 1.0, object(), object())
        assert ready_sound.play.call_count == 1

        # Exit (without release) also resets the gate
        hooks.emit("ability_exit", "rasengan")
        hooks.emit("ability_enter", "rasengan", object(), object())
        ready_sound.reset_mock()
        hooks.emit("ability_charge", "rasengan", 1.0, object(), object())
        assert ready_sound.play.call_count == 1

    def test_ready_cue_does_not_fire_below_one(self, sound_manager):
        sm, hooks, channel = sound_manager
        ready_sound = sm._get("kamehameha", "ready")

        hooks.emit("ability_enter", "kamehameha", object(), object())
        ready_sound.reset_mock()

        for val in (0.0, 0.25, 0.5, 0.75, 0.99):
            hooks.emit("ability_charge", "kamehameha", val, object(), object())

        ready_sound.play.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: cast sound plays on release
# ---------------------------------------------------------------------------

class TestCastSound:
    def test_cast_sound_plays_on_release(self, sound_manager):
        sm, hooks, channel = sound_manager
        cast_sound = sm._get("fireball", "cast")

        hooks.emit("ability_enter", "fireball", object(), object())
        cast_sound.reset_mock()
        hooks.emit("ability_release", "fireball", 1.0, object())
        cast_sound.play.assert_called_once()

    def test_cast_sound_not_played_on_exit_without_release(self, sound_manager):
        sm, hooks, channel = sound_manager
        cast_sound = sm._get("fireball", "cast")

        hooks.emit("ability_enter", "fireball", object(), object())
        cast_sound.reset_mock()
        hooks.emit("ability_exit", "fireball")
        cast_sound.play.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: graceful degradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_disabled_when_sound_enabled_false(self, hooks):
        import config as cfg
        original = cfg.SOUND_ENABLED
        cfg.SOUND_ENABLED = False
        try:
            with patch("audio.sounds._PYGAME_AVAILABLE", True):
                from audio.sounds import SoundManager
                sm = SoundManager(hooks)
            assert not sm._enabled
        finally:
            cfg.SOUND_ENABLED = original

    def test_disabled_when_pygame_unavailable(self, hooks):
        with patch("audio.sounds._PYGAME_AVAILABLE", False):
            from audio.sounds import SoundManager
            sm = SoundManager(hooks)
        assert not sm._enabled

    def test_disabled_when_mixer_not_initialised(self, hooks):
        mixer_mod = MagicMock()
        mixer_mod.get_init.return_value = None  # falsy → not initialised
        with (
            patch("audio.sounds._PYGAME_AVAILABLE", True),
            patch("audio.sounds.pygame.mixer", mixer_mod),
        ):
            from audio.sounds import SoundManager
            sm = SoundManager(hooks)
        assert not sm._enabled

    def test_missing_wav_does_not_raise(self, sound_manager):
        """A None sound (wav not found) must not cause any exception."""
        sm, hooks, channel = sound_manager
        # Force the charge sound for one ability to None
        sm._sounds["frost_nova_charge"] = None
        sm._sounds["frost_nova_ready"] = None
        sm._sounds["frost_nova_cast"] = None

        # These must not raise
        hooks.emit("ability_enter", "frost_nova", object(), object())
        hooks.emit("ability_charge", "frost_nova", 1.0, object(), object())
        hooks.emit("ability_release", "frost_nova", 1.0, object())
        hooks.emit("ability_exit", "frost_nova")


# ---------------------------------------------------------------------------
# Tests: time_shatter event cue
# ---------------------------------------------------------------------------

class TestTimeShatter:
    def test_time_shatter_plays_on_time_freeze_exit(self, sound_manager):
        sm, hooks, channel = sound_manager
        shatter_sound = sm._oneshots.get("time_shatter")
        assert shatter_sound is not None

        hooks.emit("ability_enter", "time_freeze", object(), object())
        shatter_sound.reset_mock()

        # Exit the ability — shatter sound should play
        hooks.emit("ability_exit", "time_freeze")
        shatter_sound.play.assert_called_once()

    def test_time_shatter_not_played_on_other_ability_exit(self, sound_manager):
        sm, hooks, channel = sound_manager
        shatter_sound = sm._oneshots.get("time_shatter")

        hooks.emit("ability_enter", "chidori", object(), object())
        shatter_sound.reset_mock()

        # Exit a different ability — shatter sound should NOT play
        hooks.emit("ability_exit", "chidori")
        shatter_sound.play.assert_not_called()

    def test_time_shatter_missing_wav_does_not_raise(self, sound_manager):
        """Missing time_shatter.wav must not cause an exception."""
        sm, hooks, channel = sound_manager
        sm._oneshots["time_shatter"] = None

        # This must not raise
        hooks.emit("ability_enter", "time_freeze", object(), object())
        hooks.emit("ability_exit", "time_freeze")
