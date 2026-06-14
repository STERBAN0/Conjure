"""Unit tests for smoothing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from gestures.smoothing import EMA, OneEuroFilter


@pytest.mark.unit
def test_ema_initial_value_returns_immediately():
    # Arrange
    ema = EMA(alpha=0.5, init=0.0)

    # Act
    result = ema(1.0)

    # Assert — single sample mixes init and target
    assert 0.4 <= result <= 0.6


@pytest.mark.unit
def test_ema_converges_toward_constant_input():
    # Arrange
    ema = EMA(alpha=0.4)
    target = 5.0

    # Act
    for _ in range(50):
        last = ema(target)

    # Assert
    assert abs(last - target) < 0.01


@pytest.mark.unit
def test_one_euro_first_call_returns_input_unchanged():
    # Arrange
    f = OneEuroFilter(mincutoff=1.0, beta=0.0)

    # Act
    out = f(np.array([1.0, 2.0, 3.0]), t=0.0)

    # Assert
    assert np.allclose(out, [1.0, 2.0, 3.0])


@pytest.mark.unit
def test_one_euro_smooths_noisy_signal_around_zero():
    # Arrange
    f = OneEuroFilter(mincutoff=1.0, beta=0.0)
    rng = np.random.default_rng(42)

    # Act
    samples = []
    t = 0.0
    for _ in range(200):
        x = float(rng.normal(0.0, 1.0))
        samples.append(float(f(x, t=t)))
        t += 1.0 / 60.0

    # Assert — smoothed standard deviation should be much smaller than raw
    smoothed_std = float(np.std(samples[100:]))
    assert smoothed_std < 0.5  # raw std is ~1.0


@pytest.mark.unit
def test_one_euro_reset_clears_state():
    # Arrange
    f = OneEuroFilter()
    f(1.0, t=0.0)
    f(2.0, t=1.0 / 60.0)

    # Act
    f.reset()
    out = f(7.0, t=0.0)

    # Assert
    assert np.isclose(float(out), 7.0)
