"""Download MediaPipe Tasks model files needed by Conjure.

Cross-platform replacement for the curl one-liner in the README. Run from
the repo root:

    python scripts/download_model.py

Idempotent: skips a model if it is already present and its size is above the
minimum threshold (guards against corrupt partial downloads).

Models downloaded
-----------------
- hand_landmarker.task  — required for gesture / hand tracking
- face_landmarker.task  — required for face / eye tracking (laser eyes, etc.)
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class _ModelSpec(NamedTuple):
    filename: str
    url: str
    min_size_bytes: int  # guard against truncated downloads


_MODELS: tuple[_ModelSpec, ...] = (
    _ModelSpec(
        filename="hand_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        ),
        min_size_bytes=5 * 1024 * 1024,  # real file ~7.5 MB
    ),
    _ModelSpec(
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        ),
        # Real float16 bundle is ~3.6 MB. Keep the guard well below that so a
        # legitimate download is never rejected, but high enough to catch a
        # truncated file or an HTML error page (those are only a few KB).
        min_size_bytes=2 * 1024 * 1024,
    ),
)


def _download_model(spec: _ModelSpec, out_dir: Path) -> int:
    """Download a single model.  Returns 0 on success, 1 on error."""
    out_path = out_dir / spec.filename

    if out_path.exists() and out_path.stat().st_size > spec.min_size_bytes:
        log.info(
            "%s already present (%.1f MB) — skipping",
            spec.filename,
            out_path.stat().st_size / 1024 / 1024,
        )
        return 0

    log.info("Downloading %s from %s ...", spec.filename, spec.url)
    try:
        urllib.request.urlretrieve(spec.url, out_path)
    except Exception as exc:
        log.error("Download of %s failed: %r", spec.filename, exc)
        log.error("You can also fetch it manually with:")
        log.error("  curl -L -o %s %s", out_path, spec.url)
        return 1

    size = out_path.stat().st_size
    size_mb = size / 1024 / 1024
    if size < spec.min_size_bytes:
        log.error(
            "Downloaded %s is suspiciously small (%.1f MB); deleting.",
            spec.filename,
            size_mb,
        )
        out_path.unlink(missing_ok=True)
        return 1

    log.info("Saved %s to %s (%.1f MB)", spec.filename, out_path, size_mb)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "models"
    out_dir.mkdir(exist_ok=True)

    exit_code = 0
    for spec in _MODELS:
        exit_code |= _download_model(spec, out_dir)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
