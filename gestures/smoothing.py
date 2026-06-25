"""Signal smoothing primitives used everywhere noisy data crosses a boundary.

OneEuroFilter is the workhorse — it adapts cutoff frequency based on the
signal's own velocity, so it stays smooth when you're still and stays
responsive when you move fast. Standard reference:
http://cristal.univ-lille.fr/~casiez/1euro/
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, freq: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    te = 1.0 / freq
    return 1.0 / (1.0 + tau / te)


class OneEuroFilter:
    """Adaptive low-pass filter. Works on scalars *or* numpy arrays."""

    def __init__(
        self,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
        freq: float = 60.0,
    ) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.freq = freq
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def __call__(self, x, t: float | None = None):
        x = np.asarray(x, dtype=np.float32)

        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt > 1e-6:
                self.freq = 1.0 / dt
        if t is not None:
            self._t_prev = t

        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = np.zeros_like(x)
            return x

        dx = (x - self._x_prev) * self.freq
        a_d = _alpha(self.dcutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        # `_alpha` is monotonic in cutoff; vectorise via numpy ufuncs.
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        a = 1.0 / (1.0 + tau / te)

        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class EMA:
    """Exponential moving average. Cheap, biased, perfectly fine for energy-like signals."""

    def __init__(self, alpha: float, init: float = 0.0) -> None:
        self.alpha = alpha
        self.value = float(init)

    def __call__(self, x: float) -> float:
        self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, init: float = 0.0) -> None:
        self.value = float(init)
