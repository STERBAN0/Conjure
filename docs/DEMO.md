# Demo Capture

Captures a screen recording of live gesture-driven anime abilities for the demo GIF.

## How to Capture

1. **Run the app**
   ```bash
   python main.py
   ```

2. **Screen-record the window**
   - Windows: Use Game Bar (Win+G) or OBS for a 10–15 second recording.
   - macOS: Use QuickTime Player (File → New Screen Recording) or ScreenFlow.
   - Linux: Use OBS or SimpleScreenRecorder.
   - Perform 2–3 gestures during recording (e.g., Chidori, Kamehameha, Rasengan).

3. **Convert to optimized GIF**
   ```bash
   scripts/make_demo_gif.sh <input_video> docs/demo.gif
   ```
   For example:
   ```bash
   scripts/make_demo_gif.sh ~/Downloads/demo.mp4 docs/demo.gif
   ```

The script uses `ffmpeg` with a palette pass for high-quality, optimized output.

## Options

Control the output with environment variables:

```bash
FPS=20 WIDTH=1280 scripts/make_demo_gif.sh input.mp4 output.gif
```

- `FPS`: Frames per second (default: 15)
- `WIDTH`: Output width in pixels (default: 960; scales height proportionally)
