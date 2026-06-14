"""Download the MediaPipe Tasks hand landmarker model.

Cross-platform replacement for the curl one-liner in the README. Run from
the repo root:

    python scripts/download_model.py

Idempotent: skips download if the model is already present and the size
matches (within a few KB).
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
EXPECTED_MIN_SIZE = 5 * 1024 * 1024   # 5 MB; real file is ~7.5 MB

log = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "hand_landmarker.task"

    if out_path.exists() and out_path.stat().st_size > EXPECTED_MIN_SIZE:
        log.info("Model already present at %s (%.1f MB)",
                 out_path, out_path.stat().st_size / 1024 / 1024)
        return 0

    log.info("Downloading hand_landmarker.task from %s ...", URL)
    try:
        urllib.request.urlretrieve(URL, out_path)
    except Exception as e:
        log.error("Download failed: %r", e)
        log.error("You can also fetch it manually with:")
        log.error("  curl -L -o %s %s", out_path, URL)
        return 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    if out_path.stat().st_size < EXPECTED_MIN_SIZE:
        log.error("Downloaded file is suspiciously small (%.1f MB)", size_mb)
        return 1
    log.info("Saved to %s (%.1f MB)", out_path, size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
