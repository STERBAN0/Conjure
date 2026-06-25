"""Threaded audio capture + FFT.

Reads from the default input device (mic) using sounddevice. Computes:
- broadband RMS level (0..1, perceptual scaling)
- 8-band log-spaced spectrum (0..1)

Exposes the latest values via thread-safe getters. Optional — if
sounddevice fails to initialise (no mic, missing PortAudio binary),
the system continues with zeroed audio signals.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

import config

log = logging.getLogger(__name__)

try:
    import sounddevice as sd
    _SD_OK = True
except Exception as _e:  # PortAudio missing / no device
    sd = None
    _SD_OK = False
    _SD_ERR = repr(_e)


# 8 log-spaced band edges (Hz). Perceptually meaningful for music + voice.
_BAND_EDGES = np.array([20, 80, 200, 500, 1000, 2000, 4000, 8000, 16000])


class AudioAnalyzer:
    def __init__(self) -> None:
        self._level = 0.0
        self._bands = np.zeros(8, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None
        self._enabled = False

        if not config.AUDIO_ENABLED or not _SD_OK:
            if not _SD_OK:
                log.warning("audio disabled: sounddevice unavailable (%s)", _SD_ERR)
            return

        try:
            self._stream = sd.InputStream(
                samplerate=config.AUDIO_SAMPLE_RATE,
                blocksize=config.AUDIO_BLOCK,
                channels=config.AUDIO_CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._enabled = True
        except Exception as e:
            log.warning("audio disabled: stream start failed (%r)", e)
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        # `status` may report under/overflows; log at debug level to avoid
        # flooding the console while keeping the audio path real-time.
        if status:
            log.debug("audio stream status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata

        # RMS with light perceptual curve.
        rms = float(np.sqrt(np.mean(mono * mono) + 1e-12))
        level = float(np.clip(np.tanh(rms * 6.0), 0.0, 1.0))

        # FFT magnitude, log frequency bands.
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        freqs = np.fft.rfftfreq(len(mono), 1.0 / config.AUDIO_SAMPLE_RATE)

        bands = np.zeros(8, dtype=np.float32)
        for i in range(8):
            lo, hi = _BAND_EDGES[i], _BAND_EDGES[i + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if np.any(mask):
                bands[i] = float(np.mean(spectrum[mask]))
        # Normalise: log scale + soft clip.
        bands = np.tanh(np.log1p(bands * 50.0) * 0.4)

        with self._lock:
            self._level = level
            self._bands = bands

    def get(self) -> tuple[float, np.ndarray]:
        with self._lock:
            return self._level, self._bands.copy()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
