#!/usr/bin/env python3
"""One-command setup + launch for Conjure.

Run this with whatever Python you have and it does everything the README's
manual steps do, but cross-platform and in a single command:

    python run.py        # Windows / macOS / Linux — all the same

It will, in order:

  1. check your Python is new enough (>= 3.10),
  2. create a local virtual environment in ``.venv`` (only if missing),
  3. install the runtime dependencies into it,
  4. download the MediaPipe models (idempotent — skips ones already there),
  5. launch the app.

The whole point of this script is that it never uses ``source .venv/bin/
activate`` (which doesn't exist on Windows) — it calls the venv's Python by
its full path instead, so the exact same command works on every platform.

Useful flags
------------
    python run.py --setup-only    # install + download models, but don't launch
    python run.py --reinstall     # force-reinstall dependencies
    python run.py -h              # this help
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
DEPS_STAMP = VENV_DIR / ".conjure-deps"  # records which requirements.txt we installed


def say(message: str) -> None:
    """Print a clearly-prefixed progress line and flush immediately."""
    print(f"==> {message}", flush=True)


def fail(message: str) -> int:
    """Print an error and return a non-zero exit code."""
    print(f"\nError: {message}", file=sys.stderr, flush=True)
    return 1


def venv_python(venv_dir: Path) -> Path:
    """Path to the Python interpreter inside a venv, per platform."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def check_python_version() -> int:
    if sys.version_info < MIN_PYTHON:
        have = ".".join(map(str, sys.version_info[:3]))
        need = ".".join(map(str, MIN_PYTHON))
        return fail(
            f"Conjure needs Python {need} or newer, but this is Python {have}.\n"
            f"Install a newer Python from https://python.org and run this again."
        )
    return 0


def ensure_venv() -> int:
    """Create the venv if it isn't already there."""
    py = venv_python(VENV_DIR)
    if py.exists():
        say(f"Virtual environment already exists ({VENV_DIR.name}/) - reusing it.")
        return 0

    say(f"Creating virtual environment in {VENV_DIR.name}/ ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
    except subprocess.CalledProcessError:
        hint = ""
        if sys.platform.startswith("linux"):
            hint = (
                "\nOn Debian/Ubuntu the venv module ships separately - try:\n"
                "    sudo apt-get install python3-venv\n"
                "then run this script again."
            )
        return fail(f"Could not create the virtual environment.{hint}")

    if not py.exists():
        return fail(
            "The virtual environment was created but its Python interpreter is "
            f"missing at {py}. Delete the {VENV_DIR.name}/ folder and try again."
        )
    return 0


def _requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def _deps_already_installed() -> bool:
    return DEPS_STAMP.exists() and DEPS_STAMP.read_text().strip() == _requirements_hash()


def install_dependencies(force: bool) -> int:
    """Install requirements into the venv (skipped if unchanged since last run)."""
    py = venv_python(VENV_DIR)

    if not force and _deps_already_installed():
        say("Dependencies already installed and unchanged - skipping install.")
        return 0

    # A current pip resolves the mediapipe/opencv wheels far more reliably than
    # the one that ships inside a fresh venv. Best-effort: warn but continue.
    say("Upgrading pip inside the virtual environment ...")
    upgraded = subprocess.run(
        [str(py), "-m", "pip", "install", "--upgrade", "pip"]
    )
    if upgraded.returncode != 0:
        say("(pip upgrade failed - continuing with the existing pip.)")

    say("Installing dependencies (this can take a couple of minutes) ...")
    try:
        subprocess.run(
            [str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)], check=True
        )
    except subprocess.CalledProcessError:
        return fail(
            "Dependency installation failed (see the pip output above).\n"
            "Most common cause: no internet connection, or your Python version "
            "has no matching mediapipe wheel yet (3.10-3.13 are supported)."
        )

    DEPS_STAMP.write_text(_requirements_hash())
    return 0


def download_models() -> int:
    py = venv_python(VENV_DIR)
    script = REPO_ROOT / "scripts" / "download_model.py"
    say("Downloading MediaPipe models (skips any already present) ...")
    try:
        subprocess.run([str(py), str(script)], check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError:
        return fail(
            "Model download failed. Check your internet connection and retry - "
            "the app cannot track hands without these files."
        )
    return 0


def launch_app() -> int:
    py = venv_python(VENV_DIR)
    say("Launching Conjure - press Q in the window to quit.\n")
    try:
        completed = subprocess.run([str(py), "main.py"], cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        return 0
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set up and launch Conjure in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--setup-only",
        "-n",
        action="store_true",
        help="install dependencies and download models, but don't launch the app",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="force a fresh dependency install even if nothing changed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if rc := check_python_version():
        return rc
    if rc := ensure_venv():
        return rc
    if rc := install_dependencies(force=args.reinstall):
        return rc
    if rc := download_models():
        return rc

    if args.setup_only:
        say("Setup complete. Run `python run.py` again to launch the app.")
        return 0

    return launch_app()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
