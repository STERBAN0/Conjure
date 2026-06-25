"""Offline gesture-diagnosis harness — run the real pipeline over a video file.

This replays a recorded clip (e.g. the user's ``testing.mp4``) through the
exact same ``HandTracker`` + ``PoseRecognizer`` the live app uses, then prints
a per-second summary of:

  - mean finger openness of the most-confident hand (sanity check that an
    *open* hand reads ~1.0 and a *fist* reads ~0.0),
  - the raw per-finger openness vector (thumb..pinky) at the sampled frame,
  - which discrete poses fired (raw geometry layer) and the stabilised
    ``classify`` output.

It is intentionally dependency-light and prints plain text so the output can
be diffed before/after a tuning change.

Usage::

    ./.venv/Scripts/python.exe scripts/diagnose_video.py testing.mp4
    ./.venv/Scripts/python.exe scripts/diagnose_video.py testing.mp4 --every 3

``--every N`` processes every Nth frame (faster; tracking still advances with
real timestamps so velocity/stillness stay meaningful enough for a summary).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Make the repo root importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.state import FrameState  # noqa: E402
from gestures.poses import PoseRecognizer  # noqa: E402
from vision.hand_tracker import HandTracker  # noqa: E402


def _fmt_fingers(f: np.ndarray) -> str:
    names = ("th", "ix", "md", "rg", "pk")
    return " ".join(f"{n}={v:.2f}" for n, v in zip(names, f, strict=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to the video clip")
    ap.add_argument("--every", type=int, default=2, help="process every Nth frame")
    ap.add_argument(
        "--fires", action="store_true",
        help="print one line per single-hand raw pose-fire with exact fingers",
    )
    ap.add_argument(
        "--summary", action="store_true",
        help="aggregate stats: hand-count flicker + finger-openness ambiguity",
    )
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = HandTracker()
    recog = PoseRecognizer()

    frame_idx = -1
    bucket = -1
    raw_seen: Counter[str] = Counter()
    classified_seen: Counter[str] = Counter()
    sample_line = ""

    # --summary accumulators
    hand_count_hist: Counter[int] = Counter()
    hand_count_transitions = 0
    prev_hand_count: int | None = None
    label_flips = 0
    prev_labels: tuple[str, ...] | None = None
    finger_bands = {"fold": 0, "amb": 0, "ext": 0}  # by config dead-zone
    finger_total = 0

    print(f"# diagnosing {args.video}  fps={fps:.1f}  mirror={config.MIRROR_INPUT}")
    if not args.summary:
        print("# sec | dominant raw sample (top hand) | raw poses | classified poses")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % args.every != 0:
            continue

        if config.MIRROR_INPUT:
            frame_bgr = cv2.flip(frame_bgr, 1)

        t = frame_idx / fps
        hands = tracker.process(frame_bgr, t)
        frame = FrameState(frame_bgr=frame_bgr, timestamp=t, dt=args.every / fps,
                           hands=hands, face=None)

        if args.summary:
            n = len(hands)
            hand_count_hist[n] += 1
            if prev_hand_count is not None and n != prev_hand_count:
                hand_count_transitions += 1
            prev_hand_count = n
            labels = tuple(sorted(h.label for h in hands))
            if prev_labels is not None and labels != prev_labels:
                label_flips += 1
            prev_labels = labels
            for hd in hands:
                for v in hd.fingers_open[1:5]:  # index..pinky
                    finger_total += 1
                    if v <= config.SINGLE_FINGER_FOLDED:
                        finger_bands["fold"] += 1
                    elif v >= config.SINGLE_FINGER_EXTENDED:
                        finger_bands["ext"] += 1
                    else:
                        finger_bands["amb"] += 1
            continue

        raw = recog._raw_matches(frame)
        classified = recog.classify(frame)
        for m in raw:
            raw_seen[m.name] += 1
        for m in classified:
            classified_seen[m.name] += 1

        if args.fires:
            for m in raw:
                if m.primary is None or len(hands) != 1:
                    continue
                print(
                    f"t={t:6.2f} FIRE {m.name:13s} conf={m.confidence:.2f} "
                    f"{m.primary.label[0]} [{_fmt_fingers(m.primary.fingers_open)}] "
                    f"spread={m.primary.spread:.2f} open={m.primary.openness:.2f}"
                )
            continue

        # Capture one representative sample line per second (top-openness hand).
        if hands:
            top = max(hands, key=lambda h: h.openness)
            sample_line = (
                f"{top.label[0]} open={top.openness:.2f} "
                f"[{_fmt_fingers(top.fingers_open)}]"
            )

        sec = int(t)
        if sec != bucket:
            if bucket >= 0:
                raw_str = ",".join(f"{k}:{v}" for k, v in raw_seen.most_common())
                cls_str = ",".join(f"{k}:{v}" for k, v in classified_seen.most_common())
                print(f"{bucket:4d} | {sample_line:48s} | {raw_str or '-':28s} | {cls_str or '-'}")
            bucket = sec
            raw_seen.clear()
            classified_seen.clear()

    if args.summary:
        processed = sum(hand_count_hist.values())
        print(f"# frames processed     : {processed}")
        print(f"# hand-count histogram : {dict(sorted(hand_count_hist.items()))}")
        print(f"# hand-count flickers  : {hand_count_transitions} "
              f"({100.0 * hand_count_transitions / max(1, processed):.1f}% of frames)")
        print(f"# handedness-set flips : {label_flips}")
        if finger_total:
            pct = {k: 100.0 * v / finger_total for k, v in finger_bands.items()}
            print(f"# finger reads         : {finger_total}")
            print(f"# finger bands         : fold={pct['fold']:.1f}%  "
                  f"AMBIGUOUS={pct['amb']:.1f}%  ext={pct['ext']:.1f}%")

    tracker.close()
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
