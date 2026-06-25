"""Tests for the synchronous hook bus."""

from __future__ import annotations

import pytest

from core.hooks import HookBus


@pytest.mark.unit
def test_hookbus_emits_to_all_subscribers():
    # Arrange
    bus = HookBus()
    received: list[str] = []
    bus.on("ping", lambda x: received.append(x))
    bus.on("ping", lambda x: received.append(x.upper()))

    # Act
    bus.emit("ping", "hello")

    # Assert
    assert received == ["hello", "HELLO"]


@pytest.mark.unit
def test_hookbus_isolates_subscribers_from_each_others_errors(capsys):
    # Arrange
    bus = HookBus()
    after = []

    def boom(_):
        raise RuntimeError("boom")

    def good(x):
        after.append(x)

    bus.on("evt", boom)
    bus.on("evt", good)

    # Act
    bus.emit("evt", "ok")

    # Assert — second subscriber still ran despite the first raising
    assert after == ["ok"]


@pytest.mark.unit
def test_hookbus_off_removes_subscriber():
    # Arrange
    bus = HookBus()
    received = []
    def fn(x: object) -> None:
        received.append(x)
    bus.on("evt", fn)

    # Act
    bus.off("evt", fn)
    bus.emit("evt", 1)

    # Assert
    assert received == []


@pytest.mark.unit
def test_hookbus_emit_no_subscribers_does_not_raise():
    # Arrange
    bus = HookBus()

    # Act / Assert — must not raise
    bus.emit("never_subscribed_event", 1, 2, foo=3)
