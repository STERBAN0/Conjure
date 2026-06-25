"""Generate all Conjure ability sound-effect WAV files.

Writes 16-bit PCM stereo WAVs into audio/sfx/ using only stdlib (wave) and
numpy.  No network access, no licensed content, no extra dependencies.

Run once to populate the asset directory:
    python scripts/generate_sfx.py

The script is idempotent — it always overwrites the output files.

Each ability gets three cues:
    <ability>_charge.wav   — loops while the pose is held (charge phase)
    <ability>_ready.wav    — one-shot "fully charged, release now" cue
    <ability>_cast.wav     — one-shot release / cast sound

Additionally, some effects have one-shot event cues:
    time_shatter.wav       — glass-shattering sound when Time Freeze ends

Sound design notes (one line per ability at the bottom of this module).
"""

from __future__ import annotations

import math
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS = 2           # stereo
SAMPLE_WIDTH = 2       # 16-bit
PEAK = 0.85            # peak amplitude (headroom to avoid clipping after mix)

# Canonical output directory (relative to repo root, where the script is run)
SFX_DIR = Path("audio/sfx")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a numpy float32 array as a 16-bit PCM stereo WAV file.

    ``samples`` shape: (N,) mono  →  duplicated to stereo.
                       (N, 2)     →  written as-is.
    """
    if samples.ndim == 1:
        stereo = np.stack([samples, samples], axis=1)
    else:
        stereo = samples

    # Clip and convert to int16
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _t(duration: float) -> np.ndarray:
    """Time axis from 0 to duration (exclusive)."""
    n = int(SAMPLE_RATE * duration)
    return np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)


def _env(t: np.ndarray, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    """ADSR envelope over a time axis (all values relative to t.max())."""
    dur = float(t[-1]) if len(t) > 1 else 1.0
    env = np.ones_like(t)
    a_end = attack * dur
    d_end = (attack + decay) * dur
    r_start = (1.0 - release) * dur

    mask_a = t < a_end
    if np.any(mask_a):
        env[mask_a] = t[mask_a] / (a_end + 1e-12)

    mask_d = (t >= a_end) & (t < d_end)
    if np.any(mask_d):
        frac = (t[mask_d] - a_end) / (d_end - a_end + 1e-12)
        env[mask_d] = 1.0 - frac * (1.0 - sustain)

    mask_s = (t >= d_end) & (t < r_start)
    env[mask_s] = sustain

    mask_r = t >= r_start
    if np.any(mask_r):
        frac = (t[mask_r] - r_start) / (dur - r_start + 1e-12)
        env[mask_r] = sustain * (1.0 - frac)

    return env.astype(np.float32)


def _noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _sine(t: np.ndarray, freq: float) -> np.ndarray:
    return np.sin(2 * math.pi * freq * t).astype(np.float32)


def _saw(t: np.ndarray, freq: float) -> np.ndarray:
    phase = (freq * t) % 1.0
    return (2.0 * phase - 1.0).astype(np.float32)


def _square(t: np.ndarray, freq: float) -> np.ndarray:
    return np.sign(_sine(t, freq)).astype(np.float32)


def _lowpass(sig: np.ndarray, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Simple biquad-like IIR low-pass (Butterworth approximation)."""
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / SAMPLE_RATE
    alpha = dt / (rc + dt)
    out = sig.copy()
    for _ in range(order):
        prev = 0.0
        for i in range(len(out)):
            out[i] = prev + alpha * (out[i] - prev)
            prev = out[i]
    return out


def _highpass(sig: np.ndarray, cutoff_hz: float, order: int = 2) -> np.ndarray:
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / SAMPLE_RATE
    alpha = rc / (rc + dt)
    out = np.zeros_like(sig)
    prev_in = 0.0
    prev_out = 0.0
    for i in range(len(sig)):
        out[i] = alpha * (prev_out + sig[i] - prev_in)
        prev_in = sig[i]
        prev_out = out[i]
    return out


def _bandpass(sig: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _highpass(_lowpass(sig, hi), lo)


def _fade(sig: np.ndarray, fade_in: float = 0.02, fade_out: float = 0.05) -> np.ndarray:
    """Apply a short linear fade-in / fade-out to avoid clicks."""
    n = len(sig)
    fi = min(int(SAMPLE_RATE * fade_in), n // 4)
    fo = min(int(SAMPLE_RATE * fade_out), n // 4)
    out = sig.copy()
    if fi > 0:
        out[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    if fo > 0:
        out[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return out


def _norm(sig: np.ndarray, peak: float = PEAK) -> np.ndarray:
    mx = float(np.max(np.abs(sig))) + 1e-12
    return (sig / mx * peak).astype(np.float32)


# ---------------------------------------------------------------------------
# Ability sound synthesis
# ---------------------------------------------------------------------------

# ---------- CHIDORI ---------------------------------------------------------
# Characteristic: electric crackle, "chirping birds", high-frequency sparks.
# Ref: the Chidori sound is a rapid high-frequency crackling + bird-like
# harmonic chirp (hence the name "one thousand birds").

def _chidori_charge(dur: float = 1.8) -> np.ndarray:
    """Electric crackle that intensifies — high-band noise + rapid sine bursts."""
    t = _t(dur)
    # Rising high-frequency noise (filtered white noise)
    raw = _noise(len(t), seed=1)
    hp = _highpass(raw, 3000.0)
    # Rapid amplitude modulation → crackle texture
    mod_freq = 80.0  # 80 Hz modulation = crackle rate
    mod = 0.5 + 0.5 * np.abs(_sine(t, mod_freq))
    crackle = hp * mod

    # Add faint bird-chirp overtones (quick sine sweeps)
    chirp = np.zeros(len(t), dtype=np.float32)
    rng = np.random.default_rng(42)
    for _ in range(6):
        onset = float(rng.uniform(0.1, dur - 0.2))
        width = float(rng.uniform(0.03, 0.08))
        f0 = float(rng.uniform(2000.0, 5000.0))
        sweep = rng.uniform(1.2, 2.5)
        mask = (t >= onset) & (t < onset + width)
        t_local = t[mask] - onset
        freq_sweep = f0 * (1.0 + sweep * t_local / width)
        chirp[mask] += np.sin(2 * math.pi * freq_sweep * t_local).astype(np.float32) * 0.3

    # Rising intensity envelope
    intensity = (t / dur) ** 1.5
    sig = (crackle * 0.7 + chirp) * intensity
    return _fade(_norm(sig))


def _chidori_ready(dur: float = 0.4) -> np.ndarray:
    """Bright electric zing — a sharp chirp burst signalling 'ready'."""
    t = _t(dur)
    # Descending chirp from 6kHz to 3kHz
    freq = 6000.0 - 3000.0 * (t / dur)
    sig = np.sin(2 * math.pi * np.cumsum(freq / SAMPLE_RATE)).astype(np.float32)
    env = _env(t, 0.05, 0.15, 0.5, 0.4)
    # Add a crackle burst
    crackle = _highpass(_noise(len(t), seed=7), 3500.0) * 0.4
    return _fade(_norm((sig + crackle) * env))


def _chidori_cast(dur: float = 0.7) -> np.ndarray:
    """Sharp electric discharge — rapid crackle + impact thud."""
    t = _t(dur)
    # High-frequency crackle burst
    crackle = _highpass(_noise(len(t), seed=3), 2000.0)
    burst_env = np.exp(-t * 8.0).astype(np.float32)
    # Low thud for the thrust
    thud = _sine(t, 90.0) * np.exp(-t * 14.0).astype(np.float32)
    sig = crackle * burst_env * 0.8 + thud * 0.5
    return _fade(_norm(sig))


# ---------- KAMEHAMEHA -------------------------------------------------------
# Characteristic: rising hum/whine building from low to high, then a massive
# deep blast / beam roar with a pronounced low-frequency rumble.

def _kamehameha_charge(dur: float = 2.0) -> np.ndarray:
    """Rising ki hum: sine wave sweeping up from 60 Hz to 400 Hz + harmonic shimmer."""
    t = _t(dur)
    # Logarithmic frequency sweep
    f_start, f_end = 60.0, 400.0
    freq = f_start * (f_end / f_start) ** (t / dur)
    phase = np.cumsum(freq / SAMPLE_RATE)
    sig = np.sin(2 * math.pi * phase).astype(np.float32)

    # Add harmonics that fade in over time
    h2 = np.sin(2 * math.pi * phase * 2.0).astype(np.float32) * 0.3
    h3 = np.sin(2 * math.pi * phase * 3.0).astype(np.float32) * 0.15
    shimmer = _highpass(_noise(len(t), seed=5), 1000.0) * 0.08 * (t / dur)

    combined = sig + h2 + h3 + shimmer
    intensity = (t / dur) ** 0.8
    return _fade(_norm(combined * intensity))


def _kamehameha_ready(dur: float = 0.5) -> np.ndarray:
    """Ki orb locks in — tight harmonic pulse + resonant tone."""
    t = _t(dur)
    sig = _sine(t, 300.0) * 0.6 + _sine(t, 600.0) * 0.3 + _sine(t, 900.0) * 0.1
    env = _env(t, 0.03, 0.1, 0.7, 0.3)
    # High shimmer
    shimmer = _highpass(_noise(len(t), seed=9), 2000.0) * 0.2 * env
    return _fade(_norm((sig * env) + shimmer))


def _kamehameha_cast(dur: float = 1.0) -> np.ndarray:
    """The HAAAAAA blast — massive low rumble + beam whine + air displacement."""
    t = _t(dur)
    # Deep bass rumble
    bass = _sine(t, 55.0) * 0.5 + _sine(t, 80.0) * 0.4
    # Beam whine (mid-high tone that fades)
    beam_freq = 350.0 * np.exp(-t * 1.5)
    beam_phase = np.cumsum(beam_freq / SAMPLE_RATE)
    beam = np.sin(2 * math.pi * beam_phase).astype(np.float32) * 0.4
    # Air displacement (broadband filtered noise)
    whoosh = _bandpass(_noise(len(t), seed=11), 200.0, 3000.0) * 0.5
    env = np.exp(-t * 1.2).astype(np.float32)
    sig = (bass + beam + whoosh) * env * 1.2 + bass * 0.3  # sustain bass
    return _fade(_norm(sig))


# ---------- RASENGAN ---------------------------------------------------------
# Characteristic: swirling wind, a tight spinning whoosh, rising whirlwind hum.

def _rasengan_charge(dur: float = 1.5) -> np.ndarray:
    """Whirlwind building — filtered noise with rotational amplitude modulation."""
    t = _t(dur)
    # Band-passed noise (wind-like)
    raw = _bandpass(_noise(len(t), seed=13), 200.0, 4000.0)
    # Rotational modulation: ~8 Hz spin rate, accelerating slightly
    spin_rate = 6.0 + 4.0 * (t / dur)
    spin_phase = np.cumsum(spin_rate / SAMPLE_RATE)
    mod = 0.5 + 0.5 * np.sin(2 * math.pi * spin_phase)
    # Rising tone (the chakra sphere)
    tone_freq = 180.0 + 120.0 * (t / dur)
    tone_phase = np.cumsum(tone_freq / SAMPLE_RATE)
    tone = np.sin(2 * math.pi * tone_phase).astype(np.float32) * 0.3
    intensity = (t / dur) ** 0.7
    sig = raw * mod * intensity + tone * intensity
    return _fade(_norm(sig))


def _rasengan_ready(dur: float = 0.35) -> np.ndarray:
    """Sphere snaps into focus — sharp rotational whip + resonant tone."""
    t = _t(dur)
    # Quick whip sound
    whip = _bandpass(_noise(len(t), seed=15), 500.0, 6000.0)
    env = np.exp(-t * 10.0).astype(np.float32)
    tone = _sine(t, 280.0) * 0.5
    return _fade(_norm((whip * env + tone * env) * 1.0))


def _rasengan_cast(dur: float = 0.6) -> np.ndarray:
    """Launch whoosh — rapid expanding spiral + impact."""
    t = _t(dur)
    whoosh = _bandpass(_noise(len(t), seed=17), 300.0, 5000.0)
    spin_phase = np.cumsum((8.0 + 20.0 * t / dur) / SAMPLE_RATE)
    mod = 0.5 + 0.5 * np.sin(2 * math.pi * spin_phase)
    impact = _sine(t, 160.0) * np.exp(-t * 10.0).astype(np.float32) * 0.4
    env = np.exp(-t * 3.0).astype(np.float32)
    sig = whoosh * mod * env + impact
    return _fade(_norm(sig))


# ---------- FIREBALL ---------------------------------------------------------
# Characteristic: brief whoosh during ignition, then a roaring fire crackle.

def _fireball_charge(dur: float = 1.2) -> np.ndarray:
    """Fire kindling — low rumble + crackling ignition noise."""
    t = _t(dur)
    # Low fire rumble (very low-pass noise)
    rumble = _lowpass(_noise(len(t), seed=19), 400.0)
    # Crackle: amplitude-modulated broadband noise
    crackle = _bandpass(_noise(len(t), seed=21), 500.0, 8000.0)
    crackle_env = (0.3 + 0.7 * np.random.default_rng(21).random(len(t)).astype(np.float32))
    intensity = (t / dur) ** 1.2
    sig = (rumble * 0.7 + crackle * crackle_env * 0.4) * intensity
    return _fade(_norm(sig))


def _fireball_ready(dur: float = 0.4) -> np.ndarray:
    """Fist ignites — sharp ignition whoosh."""
    t = _t(dur)
    whoosh = _bandpass(_noise(len(t), seed=23), 200.0, 5000.0)
    env = np.exp(-t * 7.0).astype(np.float32)
    pop = _sine(t, 120.0) * np.exp(-t * 20.0).astype(np.float32) * 0.4
    return _fade(_norm(whoosh * env + pop))


def _fireball_cast(dur: float = 0.8) -> np.ndarray:
    """Launch + roar — forward whoosh + fire roar tail."""
    t = _t(dur)
    whoosh = _bandpass(_noise(len(t), seed=25), 150.0, 4000.0)
    roar_env = np.exp(-t * 3.0).astype(np.float32)
    roar = _lowpass(_noise(len(t), seed=27), 600.0) * roar_env * 0.6
    # Rising then falling whoosh amplitude
    whoosh_env = np.exp(-t * 4.0).astype(np.float32)
    sig = whoosh * whoosh_env + roar
    return _fade(_norm(sig))


# ---------- FROST NOVA -------------------------------------------------------
# Characteristic: icy shimmer, crystalline tones, shattering glass burst.

def _frost_nova_charge(dur: float = 1.5) -> np.ndarray:
    """Ice forming — high-frequency shimmer, crystalline overtones."""
    t = _t(dur)
    # High shimmer (filtered noise)
    shimmer = _highpass(_noise(len(t), seed=29), 3000.0)
    # Crystalline bell tones
    freqs = [3200.0, 4500.0, 6000.0, 8000.0]
    bells = np.zeros(len(t), dtype=np.float32)
    for f in freqs:
        decay = np.exp(-t * 3.0).astype(np.float32)
        bells += _sine(t, f) * decay * 0.15
    intensity = (t / dur) ** 0.8
    sig = (shimmer * 0.5 + bells) * intensity
    return _fade(_norm(sig))


def _frost_nova_ready(dur: float = 0.4) -> np.ndarray:
    """Ice lock — crisp crystalline chord."""
    t = _t(dur)
    chord = _sine(t, 2400.0) + _sine(t, 3600.0) * 0.7 + _sine(t, 4800.0) * 0.4
    shimmer = _highpass(_noise(len(t), seed=31), 4000.0) * 0.3
    env = _env(t, 0.02, 0.1, 0.6, 0.4)
    return _fade(_norm((chord + shimmer) * env))


def _frost_nova_cast(dur: float = 0.8) -> np.ndarray:
    """Nova burst — glass shatter + icy explosion."""
    t = _t(dur)
    # Glass shatter: sharp broadband noise burst
    shatter = _highpass(_noise(len(t), seed=33), 2000.0)
    burst_env = np.exp(-t * 6.0).astype(np.float32)
    # Low icy whoosh (cold air displacement)
    cold_whoosh = _bandpass(_noise(len(t), seed=35), 80.0, 1200.0)
    whoosh_env = np.exp(-t * 4.0).astype(np.float32)
    # Ice crack tones
    crack = _sine(t, 2800.0) * np.exp(-t * 12.0).astype(np.float32) * 0.3
    sig = shatter * burst_env * 0.8 + cold_whoosh * whoosh_env * 0.6 + crack
    return _fade(_norm(sig))


# ---------- SPACE STRETCH ----------------------------------------------------
# Characteristic: elastic groan, a warping/wobbling spatial distortion,
# like reality being stretched like rubber or taffy.

def _space_stretch_charge(dur: float = 2.0) -> np.ndarray:
    """Fabric of space creaking — elastic wobble, slowly deepening."""
    t = _t(dur)
    # Slow LFO wobble of a low tone (elastic rubber sound)
    lfo_rate = 2.5  # Hz
    lfo = np.sin(2 * math.pi * lfo_rate * t).astype(np.float32)
    base_freq = 120.0
    freq_mod = base_freq + 60.0 * lfo  # ±60 Hz wobble
    phase = np.cumsum(freq_mod / SAMPLE_RATE)
    elastic = np.sin(2 * math.pi * phase).astype(np.float32)
    # Groan texture
    groan = _lowpass(_noise(len(t), seed=43), 500.0) * 0.3
    intensity = 0.3 + 0.7 * (t / dur)
    sig = (elastic * 0.8 + groan) * intensity
    return _fade(_norm(sig))


def _space_stretch_ready(dur: float = 0.5) -> np.ndarray:
    """Space fully warped — deep resonant groan."""
    t = _t(dur)
    lfo = np.sin(2 * math.pi * 4.0 * t).astype(np.float32)
    base_phase = np.cumsum((80.0 + 40.0 * lfo) / SAMPLE_RATE)
    groaner = np.sin(2 * math.pi * base_phase).astype(np.float32)
    env = _env(t, 0.1, 0.2, 0.6, 0.3)
    return _fade(_norm(groaner * env))


def _space_stretch_cast(dur: float = 0.8) -> np.ndarray:
    """Snap-back — rubber band snap + low resonant decay."""
    t = _t(dur)
    # Pitch ramp up then crash down
    freq_sweep = 400.0 * np.exp(-t * 5.0) + 60.0
    phase = np.cumsum(freq_sweep / SAMPLE_RATE)
    snap = np.sin(2 * math.pi * phase).astype(np.float32)
    env = np.exp(-t * 4.0).astype(np.float32)
    # Low resonant tail
    tail = _lowpass(_noise(len(t), seed=45), 300.0) * 0.4 * env
    return _fade(_norm(snap * env + tail))


# ---------- REALITY TEAR -----------------------------------------------------
# Characteristic: glitchy digital rip, distortion artifacts, a tearing fabric
# sound combined with digital glitch noise.

def _reality_tear_charge(dur: float = 1.5) -> np.ndarray:
    """Reality destabilising — glitchy noise bursts + rising dissonance."""
    t = _t(dur)
    rng = np.random.default_rng(47)
    # Square wave detuned pair (harsh, dissonant)
    s1 = _square(t, 220.0) * 0.4
    s2 = _square(t, 233.0) * 0.35  # slight detune → harsh beating
    # Glitch bursts: random amplitude spikes
    glitch = rng.standard_normal(len(t)).astype(np.float32)
    glitch_mask = rng.random(len(t)) > 0.92  # 8% of samples spike
    glitch *= glitch_mask.astype(np.float32) * 0.6
    # Digital noise
    digital = _highpass(_noise(len(t), seed=49), 4000.0) * 0.2
    intensity = (t / dur) ** 0.6
    sig = (s1 + s2 + glitch + digital) * intensity
    return _fade(_norm(sig))


def _reality_tear_ready(dur: float = 0.35) -> np.ndarray:
    """Tear opens — sharp digital glitch crack."""
    t = _t(dur)
    rng = np.random.default_rng(51)
    glitch = rng.standard_normal(len(t)).astype(np.float32)
    glitch_mask = rng.random(len(t)) > 0.70
    glitch *= glitch_mask.astype(np.float32)
    env = np.exp(-t * 8.0).astype(np.float32)
    tone = _square(t, 440.0) * 0.3
    return _fade(_norm((glitch * env + tone) * 0.8))


def _reality_tear_cast(dur: float = 0.7) -> np.ndarray:
    """The rip — tearing fabric sound with digital artifacts."""
    t = _t(dur)
    # Fabric tear: broadband noise with descending pitch
    tear = _bandpass(_noise(len(t), seed=53), 500.0, 6000.0)
    tear_env = np.exp(-t * 4.0).astype(np.float32)
    # Low rift rumble
    rift = _lowpass(_noise(len(t), seed=55), 300.0) * 0.5
    rift_env = np.exp(-t * 3.0).astype(np.float32)
    # High glitch artifacts
    glitch_hp = _highpass(_noise(len(t), seed=57), 5000.0) * 0.3
    sig = tear * tear_env + rift * rift_env + glitch_hp * tear_env
    return _fade(_norm(sig))


# ---------- TIME FREEZE ------------------------------------------------------
# Characteristic: descending/slowing pitch → deep sub-bass freeze, a
# clock-stopping resonance — the world grinds to a halt.

def _time_freeze_charge(dur: float = 1.0) -> np.ndarray:
    """Time slowing — pitched-down tone + sub resonance."""
    t = _t(dur)
    # Descending pitch
    freq = 400.0 * np.exp(-t * 1.8) + 50.0  # falls from ~400 to ~50 Hz
    phase = np.cumsum(freq / SAMPLE_RATE)
    descend = np.sin(2 * math.pi * phase).astype(np.float32)
    # Sub hum underneath
    sub = _sine(t, 40.0) * 0.4
    shimmer = _highpass(_noise(len(t), seed=59), 2000.0) * 0.1
    intensity = 0.4 + 0.6 * (t / dur)
    sig = (descend * 0.8 + sub + shimmer) * intensity
    return _fade(_norm(sig))


def _time_freeze_ready(dur: float = 0.4) -> np.ndarray:
    """Time stopped — deep resonant boom + stillness shimmer."""
    t = _t(dur)
    boom = _sine(t, 45.0) + _sine(t, 90.0) * 0.5
    boom_env = np.exp(-t * 6.0).astype(np.float32)
    shimmer = _highpass(_noise(len(t), seed=61), 3000.0) * 0.2
    env = _env(t, 0.02, 0.08, 0.5, 0.5)
    return _fade(_norm(boom * boom_env + shimmer * env))


def _time_freeze_cast(dur: float = 0.8) -> np.ndarray:
    """Freeze pulse — concussive sub-bass + ghostly reverb tail."""
    t = _t(dur)
    sub_burst = _sine(t, 35.0) + _sine(t, 55.0) * 0.6
    burst_env = np.exp(-t * 5.0).astype(np.float32)
    # Ghostly shimmer fade-in (reverberation)
    reverb = _lowpass(_noise(len(t), seed=63), 600.0) * 0.3
    reverb_env = (1.0 - np.exp(-t * 8.0)) * np.exp(-t * 2.0)
    reverb_env = reverb_env.astype(np.float32)
    sig = sub_burst * burst_env + reverb * reverb_env
    return _fade(_norm(sig))


# ---------- LASER EYES -------------------------------------------------------
# Characteristic: building high-pitched electric whine → sustained beam zap.
# The charge build is exactly LASER_EYES_CHARGE_SECONDS long (1.0s) and is played
# ONCE (not looped), so the moment the whine finishes is the cue to OPEN YOUR EYES
# and fire. Keep this duration in sync with config.LASER_EYES_CHARGE_SECONDS.

def _laser_eyes_charge(dur: float = 1.0) -> np.ndarray:
    """Rising electric whine — pitch sweeps 200→1800 Hz over the 1s charge."""
    t = _t(dur)
    freq = 200.0 + 1600.0 * (t / dur) ** 1.5
    phase = np.cumsum(freq / SAMPLE_RATE)
    whine = np.sin(2 * math.pi * phase).astype(np.float32)
    # Add harmonics for richness
    h2 = np.sin(2 * math.pi * phase * 2.0).astype(np.float32) * 0.25
    h3 = np.sin(2 * math.pi * phase * 3.0).astype(np.float32) * 0.1
    # Subtle crackle from electric buildup
    crackle = _highpass(_noise(len(t), seed=65), 5000.0) * 0.15 * (t / dur)
    intensity = 0.3 + 0.7 * (t / dur) ** 0.8
    sig = (whine + h2 + h3 + crackle) * intensity
    return _fade(_norm(sig))


def _laser_eyes_ready(dur: float = 0.5) -> np.ndarray:
    """Eyes CHARGED — unmistakable rising-pitch alarm ping + zap crackle.

    This is the most important ready cue: tells user to open eyes NOW.
    Bright, piercing, distinct from the charge whine.
    """
    t = _t(dur)
    # Fast upward sweep: 1800→3500 Hz (continues from where charge left off)
    freq = 1800.0 + 1700.0 * (t / dur)
    phase = np.cumsum(freq / SAMPLE_RATE)
    ping = np.sin(2 * math.pi * phase).astype(np.float32)
    # Sharp electric crackle
    crackle = _highpass(_noise(len(t), seed=67), 3000.0) * 0.5
    env = _env(t, 0.03, 0.05, 0.8, 0.2)
    # Two-tone alarm effect
    beep = _sine(t, 2400.0) * 0.3 + _sine(t, 3000.0) * 0.2
    return _fade(_norm((ping + beep + crackle) * env))


def _laser_eyes_cast(dur: float = 0.8) -> np.ndarray:
    """Beam fires — sustained high-energy zap with power drop-off."""
    t = _t(dur)
    # Core beam: high-frequency sustained tone
    beam = _sine(t, 2800.0) * 0.4 + _sine(t, 1400.0) * 0.3
    env = np.exp(-t * 2.5).astype(np.float32)
    # Sizzle
    sizzle = _highpass(_noise(len(t), seed=69), 4000.0) * 0.4 * np.exp(-t * 3.0).astype(np.float32)
    # Low impact rumble
    impact = _sine(t, 120.0) * 0.3 * np.exp(-t * 8.0).astype(np.float32)
    sig = beam * env + sizzle + impact
    return _fade(_norm(sig))


# ---------- TIME SHATTER -------------------------------------------------
# Characteristic: bright glass-shattering burst with fast decay and crystalline
# tinkle pings. Plays when Time Freeze ends and the frozen pane breaks.

def _time_shatter(dur: float = 0.7) -> np.ndarray:
    """Glass shatter — bright noise burst + high-pitched tinkle pings.

    Main layer: high-pass filtered white noise with exponential decay,
    creating the initial sharp shattering sound. Overlay: random short
    sine-wave pings at crystalline frequencies to simulate glass fragments.
    """
    t = _t(dur)
    rng = np.random.default_rng(71)

    # Main shatter: bright broadband noise burst with fast decay
    shatter = _highpass(_noise(len(t), seed=71), 4000.0)
    burst_env = np.exp(-t * 6.5).astype(np.float32)
    main_layer = shatter * burst_env * 0.9

    # Crystalline tinkle pings: randomized high-frequency sine bursts
    tinkle = np.zeros(len(t), dtype=np.float32)
    for _ in range(8):  # 8 random "glass piece" pings
        onset = float(rng.uniform(0.0, dur - 0.15))
        duration = float(rng.uniform(0.08, 0.18))
        freq = float(rng.uniform(3000.0, 8000.0))

        mask = (t >= onset) & (t < onset + duration)
        t_local = t[mask] - onset
        ping_env = np.exp(-t_local * 12.0).astype(np.float32)
        tinkle[mask] += np.sin(2 * math.pi * freq * t_local).astype(np.float32) * ping_env * 0.25

    # Combine: main shatter + tinkle overlay
    sig = main_layer + tinkle
    return _fade(_norm(sig))


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_SynthFn = Callable[[], np.ndarray]

_GENERATORS: dict[str, tuple[_SynthFn, _SynthFn, _SynthFn]] = {
    "chidori":       (_chidori_charge,       _chidori_ready,       _chidori_cast),
    "kamehameha":    (_kamehameha_charge,     _kamehameha_ready,    _kamehameha_cast),
    "rasengan":      (_rasengan_charge,       _rasengan_ready,      _rasengan_cast),
    "fireball":      (_fireball_charge,       _fireball_ready,      _fireball_cast),
    "frost_nova":    (_frost_nova_charge,     _frost_nova_ready,    _frost_nova_cast),
    "space_stretch": (_space_stretch_charge,  _space_stretch_ready, _space_stretch_cast),
    "reality_tear":  (_reality_tear_charge,   _reality_tear_ready,  _reality_tear_cast),
    "time_freeze":   (_time_freeze_charge,    _time_freeze_ready,   _time_freeze_cast),
    "laser_eyes":    (_laser_eyes_charge,     _laser_eyes_ready,    _laser_eyes_cast),
}

# One-shot sounds (not charge/ready/cast triplets) are generated separately
_ONE_SHOT_GENERATORS: dict[str, _SynthFn] = {
    "time_shatter": _time_shatter,
}

# Cues supplied by real external audio (converted from mp3/m4a via ffmpeg), NOT
# synthesised. generate_all() skips these so a regen never clobbers the user's
# real clips. Re-run the converts after regenerating:
#   kamehameha_cast  <- kamehameha-wave-sound-effect.mp3
#   time_freeze_charge <- ticking_clock.m4a
#   chidori_voice    <- chidori_sound.mp3   (loaded as a one-shot, not in this table)
_EXTERNAL_CUES: frozenset[str] = frozenset({"kamehameha_cast", "time_freeze_charge"})


def generate_all(out_dir: Path = SFX_DIR) -> None:
    """Generate all SFX files and write them to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    skipped = 0
    for name, (charge_fn, ready_fn, cast_fn) in _GENERATORS.items():
        for suffix, fn in (("charge", charge_fn), ("ready", ready_fn), ("cast", cast_fn)):
            if f"{name}_{suffix}" in _EXTERNAL_CUES:
                print(f"  skip  {name}_{suffix}.wav  (external clip — keep the real audio)")
                skipped += 1
                continue
            path = out_dir / f"{name}_{suffix}.wav"
            samples = fn()
            _write_wav(path, samples)
            size = path.stat().st_size
            total_bytes += size
            print(f"  wrote {path}  ({size // 1024} KB)")

    for name, fn in _ONE_SHOT_GENERATORS.items():
        path = out_dir / f"{name}.wav"
        samples = fn()
        _write_wav(path, samples)
        size = path.stat().st_size
        total_bytes += size
        print(f"  wrote {path}  ({size // 1024} KB)")

    total_count = len(_GENERATORS) * 3 + len(_ONE_SHOT_GENERATORS) - skipped
    print(f"\nDone — {total_count} files, {total_bytes // 1024} KB total")


if __name__ == "__main__":
    print(f"Generating SFX -> {SFX_DIR.resolve()}")
    generate_all()
