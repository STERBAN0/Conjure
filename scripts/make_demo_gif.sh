#!/usr/bin/env bash
# Converts a screen recording to an optimized, looping GIF via palette pass.
#
# Usage:
#   scripts/make_demo_gif.sh input.mp4 docs/demo.gif
#   FPS=20 WIDTH=1280 scripts/make_demo_gif.sh input.mp4 docs/demo.gif
#
# Environment:
#   FPS      - output framerate (default: 15)
#   WIDTH    - output width in pixels (default: 960; scales height proportionally)

set -euo pipefail

IN="${1:?input video required}"
OUT="${2:-docs/demo.gif}"
FPS="${FPS:-15}"
WIDTH="${WIDTH:-960}"

# Generate palette for high-quality GIF
ffmpeg -i "$IN" -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen" -y /tmp/aether_palette.png

# Convert video to GIF using palette
ffmpeg -i "$IN" -i /tmp/aether_palette.png -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse" -loop 0 -y "$OUT"

echo "Wrote $OUT"
