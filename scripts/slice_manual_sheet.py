"""slice_manual_sheet.py — Crop a 2×5 contact-sheet into 10 per-gesture tiles.

Usage::

    python scripts/slice_manual_sheet.py <sheet.png> [OPTIONS]

Options::

    --rows INT         Number of rows in the grid (default 2)
    --cols INT         Number of columns in the grid (default 5)
    --margin-frac FLOAT  Fractional white-space to trim from each cell edge (default 0.04)
    --out-dir PATH     Destination directory (default docs/manual_images)

The fixed gesture order matches the manual grid layout:

    Row 1: fireball  rasengan  chidori  time_freeze  laser_eyes
    Row 2: kamehameha  space_stretch  reality_tear  frost_nova

Each cropped tile is saved as ``<out-dir>/<id>.png``, overwriting any
existing file.  The script prints the path of every file it writes.

Exit codes::

    0  success
    1  argument / path error
    2  Pillow not installed

"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Pillow availability check — print a clear hint before we do anything else.
# ---------------------------------------------------------------------------

try:
    from PIL import Image
except ImportError:
    print(
        "ERROR: Pillow is not installed.\n"
        "  Install it with:  pip install Pillow\n"
        "  or inside the project venv:  .venv/Scripts/pip install Pillow",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------------
# Fixed ability order — must mirror the grid layout of any sheet created
# from the standard manual export.
# ---------------------------------------------------------------------------

_ABILITY_ORDER: list[str] = [
    "fireball",
    "rasengan",
    "chidori",
    "time_freeze",
    "laser_eyes",
    "kamehameha",
    "space_stretch",
    "reality_tear",
    "frost_nova",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="slice_manual_sheet",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sheet",
        metavar="sheet.png",
        help="Path to the contact-sheet image to slice.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        metavar="INT",
        help="Number of rows in the grid (default: 2).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        metavar="INT",
        help="Number of columns in the grid (default: 5).",
    )
    parser.add_argument(
        "--margin-frac",
        type=float,
        default=0.04,
        metavar="FLOAT",
        help="Fractional margin to trim from each cell edge (default: 0.04).",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/manual_images",
        metavar="PATH",
        help="Output directory (default: docs/manual_images).",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (sheet_path, out_dir_path) after validation.

    Raises SystemExit(1) on any error so the caller doesn't need to catch.
    """
    sheet_path = pathlib.Path(args.sheet)
    if not sheet_path.exists():
        print(f"ERROR: Sheet file not found: {sheet_path}", file=sys.stderr)
        sys.exit(1)
    if not sheet_path.is_file():
        print(f"ERROR: Sheet path is not a file: {sheet_path}", file=sys.stderr)
        sys.exit(1)

    total = args.rows * args.cols
    if total != len(_ABILITY_ORDER):
        print(
            f"ERROR: rows × cols = {total} but there are {len(_ABILITY_ORDER)} "
            f"abilities.  Adjust --rows / --cols so the product is {len(_ABILITY_ORDER)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (0.0 <= args.margin_frac < 0.5):
        print(
            f"ERROR: --margin-frac must be in [0, 0.5).  Got: {args.margin_frac}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    return sheet_path, out_dir


def _slice(
    sheet_path: pathlib.Path,
    rows: int,
    cols: int,
    margin_frac: float,
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """Slice the sheet and write one PNG per ability; return output paths."""
    try:
        sheet = Image.open(sheet_path).convert("RGBA")
    except Exception as exc:
        print(f"ERROR: Could not open image '{sheet_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    sheet_w, sheet_h = sheet.size
    cell_w = sheet_w / cols
    cell_h = sheet_h / rows

    margin_x = cell_w * margin_frac
    margin_y = cell_h * margin_frac

    written: list[pathlib.Path] = []
    index = 0

    for row in range(rows):
        for col in range(cols):
            ability_id = _ABILITY_ORDER[index]
            index += 1

            # Raw cell bounds
            left = col * cell_w
            top = row * cell_h
            right = left + cell_w
            bottom = top + cell_h

            # Apply margin trim (inset by margin_frac on each side)
            crop_box = (
                int(left + margin_x),
                int(top + margin_y),
                int(right - margin_x),
                int(bottom - margin_y),
            )

            try:
                tile = sheet.crop(crop_box)
            except Exception as exc:
                print(
                    f"ERROR: Failed to crop tile for '{ability_id}' "
                    f"at box {crop_box}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)

            out_path = out_dir / f"{ability_id}.png"
            try:
                tile.save(str(out_path))
            except Exception as exc:
                print(
                    f"ERROR: Could not write '{out_path}': {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(out_path)
            written.append(out_path)

    return written


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sheet_path, out_dir = _validate_args(args)
    written = _slice(sheet_path, args.rows, args.cols, args.margin_frac, out_dir)
    print(f"\nSliced {len(written)} tiles → {out_dir}")


if __name__ == "__main__":
    main()
