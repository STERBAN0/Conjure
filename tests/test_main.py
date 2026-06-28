"""Startup-robustness tests for main.py — the pre-flight failure paths.

These cover what a first-time user sees when Conjure can't run: a clean,
actionable message and a non-zero exit code, NOT a raw Python traceback. The
pre-flight (camera + model checks) runs before ``pygame.display.set_mode``, so
these paths never create a window and need no display.
"""

from __future__ import annotations

import main


def test_friendly_exit_returns_1_and_includes_detail(capsys):
    # Arrange / Act
    rc = main._friendly_exit(
        "No webcam detected.", RuntimeError("Cannot open camera 0")
    )

    # Assert
    assert rc == 1
    err = capsys.readouterr().err
    assert "No webcam detected." in err
    assert "Cannot open camera 0" in err  # technical detail is surfaced


def test_friendly_exit_without_detail_omits_technical_line(capsys):
    rc = main._friendly_exit("Something went wrong.")

    assert rc == 1
    err = capsys.readouterr().err
    assert "Something went wrong." in err
    assert "technical detail" not in err


def test_main_exits_cleanly_when_no_webcam(monkeypatch, capsys):
    """A missing/busy webcam exits 1 with a webcam message — no traceback.

    The pre-flight constructs ``Camera()`` before the window is created, so a
    RuntimeError here returns through ``_friendly_exit`` without ever touching
    the display.
    """
    class _BoomCamera:
        def __init__(self) -> None:
            raise RuntimeError("Cannot open camera 0")

    monkeypatch.setattr(main, "Camera", _BoomCamera)

    rc = main.main()

    assert rc == 1
    assert "webcam" in capsys.readouterr().err.lower()
