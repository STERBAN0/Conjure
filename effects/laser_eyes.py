"""Laser Eyes — twin beams that converge and combine at one shared impact point.

Visual specification
--------------------
- LAYER_FG.
- Reads FaceData (face.gaze, left_eye_px, right_eye_px), not hand data.
- **Charging** (eyes closed): a warm glow gathers on each eye as the charge ramps.
- **Active** (laser firing): BOTH eyes fire to ONE shared impact point derived from
  the midpoint between the eyes. The two beams visibly CONVERGE and combine at that
  point with a bright white-hot convergence flash. The aim follows your HEAD AND your
  EYES together (gaze is a magnitude-carrying offset — see FaceData.gaze). The instant
  you open your eyes to fire, the current gaze is captured as the baseline, so the
  impact starts ON your own face; from there the impact slides directly with how far
  you look — ``impact = eye_midpoint + (gaze - baseline) * REACH_PX`` — which grows
  continuously from zero, so EVERY pixel (including right around your face) is
  reachable with NO dead ring (see ``_impact_point``).
- A single molten "screen melt" scorch renders live at the shared impact, and the
  marks are joined into continuous strokes (consecutive impacts are linked) so you
  can WRITE words or draw a smiley. Each deposit lands on the shared **MeltCanvas**,
  an always-on overlay that survives across frames AND after the laser turns off — so
  the drawing stays on screen until you press the clear key ('R').
- Because the face position is cached, the laser keeps firing (and drawing) at the
  LAST KNOWN position even if the face briefly stops being detected.

The laser is a charge→fire ability driven entirely by the router; this effect
only draws. Face data is cached in on_enter / on_charge / on_active / on_release
because render() receives no ``frame`` parameter.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pygame

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
    AbilityState,
    FaceData,
    FrameState,
    GestureSignals,
)
from effects.base import LAYER_FG, Effect
from effects.utils import (
    additive_circle,
    additive_polyline,
    additive_ring,
    draw_screen_flash,
    radial_glow,
)

# Internal constants for the melt animation.
_FLICKER_HZ: float = 12.0   # flicker cycles per second
_DRIP_SPEED: float = 40.0   # pixels per second for drip growth


class MeltCanvas(Effect):
    """Always-on overlay holding the persistent molten "drawing".

    The laser deposits thin scorch blobs here; this surface is blitted every
    frame (even while the laser is off) so the words you write with the beams
    stay on screen. It is only erased by :meth:`clear` (wired to the 'R' key in
    main.py via ``EffectsRenderer.clear_drawings``).
    """

    layer = LAYER_FG
    name = "melt_canvas"
    ability_name = ""   # always-on (never gated on an ability)

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._has_content = False   # skip the full-screen blit while blank

    def clear(self) -> None:
        self.surface.fill((0, 0, 0, 0))
        self._has_content = False

    def deposit(self, point: tuple[int, int]) -> None:
        """Add a thin molten scorch blob at *point* onto the persistent canvas."""
        r = int(config.LASER_EYES_MELT_TRAIL_RADIUS_PX)
        a = int(config.LASER_EYES_MELT_TRAIL_ALPHA)
        additive_circle(self.surface, point, r, config.LASER_EYES_MELT_COLOR, a)
        additive_circle(
            self.surface, point, max(1, r // 2),
            config.LASER_EYES_MELT_CORE_COLOR, min(255, a + 40),
        )
        self._has_content = True

    def deposit_segment(
        self, p0: tuple[int, int], p1: tuple[int, int]
    ) -> None:
        """Join two consecutive impacts with scorch blobs so strokes are continuous.

        Filling the gap between frames is what lets the beam WRITE — a fast eye
        flick would otherwise leave a dotted line. Big jumps (a gaze teleport or a
        re-acquired face) are skipped so unrelated points are never connected.
        """
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        dist = math.hypot(dx, dy)
        if dist <= config.LASER_EYES_MELT_TRAIL_RADIUS_PX:
            return  # adjacent enough — the per-point blob already covers it
        if dist > config.LASER_EYES_MELT_TRAIL_MAX_JOIN_PX:
            return  # treat as a jump, not a stroke — don't draw across it
        step = max(1.0, config.LASER_EYES_MELT_TRAIL_RADIUS_PX * 0.6)
        n = max(1, int(dist / step))
        # range(1, n + 1) deposits through the endpoint so no tail gap is left
        # between the last interpolated blob and the next frame's point.
        for i in range(1, n + 1):
            frac = i / n
            self.deposit((int(p0[0] + dx * frac), int(p0[1] + dy * frac)))

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        if self._has_content:
            target.blit(self.surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


class LaserEyesEffect(Effect):
    layer = LAYER_FG
    name = "laser_eyes"
    ability_name = "laser_eyes"

    def __init__(self, width: int, height: int, melt: MeltCanvas) -> None:
        super().__init__(width, height)
        self._face: FaceData | None = None
        self._flash: float = 0.0
        self._age: float = 0.0          # seconds in active/releasing phase
        self._rng = random.Random()
        # Per-activation aim baseline — the gaze captured on the first firing
        # frame, so "wherever you looked when the beam started" becomes centre
        # (the impact begins on your own face). Reset on enter/exit.
        self._aim_baseline: np.ndarray | None = None
        # Last shared impact, used to join scorch strokes into continuous lines.
        self._last_impact: tuple[int, int] | None = None
        # Shared persistent trail canvas (owned by the renderer as an always-on
        # overlay). The laser writes into it; it is NOT cleared on enter/exit so
        # the drawing survives the laser turning off (cleared only by the 'R' key).
        self._melt = melt

    # --- Lifecycle (face caching) -------------------------------------------

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        if frame.face is not None and frame.face.present:
            self._face = frame.face
        self._flash = 0.0
        self._age = 0.0
        # New activation → re-capture the aim baseline on the first firing frame
        # and start a fresh stroke (don't connect to the previous drawing).
        self._aim_baseline = None
        self._last_impact = None

    def on_charge(
        self, charge: float, frame: FrameState, signals: GestureSignals
    ) -> None:
        if frame.face is not None and frame.face.present:
            self._face = frame.face

    def on_active(self, frame: FrameState, signals: GestureSignals) -> None:
        if frame.face is not None and frame.face.present:
            self._face = frame.face
            # Capture the aim baseline from the LIVE gaze on the first firing
            # frame (a confirmed present face), not a possibly-stale cached one,
            # so the impact reliably starts on the user's own face.
            if self._aim_baseline is None:
                self._aim_baseline = np.asarray(
                    frame.face.gaze, dtype=np.float32
                ).copy()

    def on_release(self, intensity: float, frame: FrameState) -> None:
        if frame.face is not None and frame.face.present:
            self._face = frame.face
        self._flash = max(self._flash, intensity)

    def on_exit(self) -> None:
        self._face = None
        self._flash = 0.0
        self._age = 0.0
        self._aim_baseline = None
        self._last_impact = None
        # NOTE: the melt trail is intentionally NOT cleared here — the drawing
        # persists after the laser turns off and is only erased via the 'R' key.

    # --- Update -------------------------------------------------------------

    def update(
        self, signals: GestureSignals, dt: float, ability: AbilityState
    ) -> None:
        self._flash = max(0.0, self._flash - dt * 5.0)
        if ability.phase in (PHASE_ACTIVE, PHASE_RELEASING):
            self._age += dt
        else:
            # Reset age outside active window so drips start short each time.
            self._age = 0.0

    # --- Render -------------------------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        face = self._face
        if face is None or not face.present:
            return

        left_eye = face.left_eye_px
        right_eye = face.right_eye_px
        charge = float(ability.charge)

        if ability.phase == PHASE_CHARGING:
            self._render_charge_glow(target, left_eye, right_eye, charge)
        elif ability.phase in (PHASE_ACTIVE, PHASE_RELEASING):
            aim = self._gaze_aim(face)
            impact = self._impact_point(left_eye, right_eye, aim)
            # Deposit the scorch at the shared impact and JOIN it to the previous
            # one so a swept beam writes a continuous line, not a dotted trail.
            self._melt.deposit(impact)
            if self._last_impact is not None:
                self._melt.deposit_segment(self._last_impact, impact)
            self._last_impact = impact
            self._render_beams(target, left_eye, right_eye, impact)
            self._render_melt(target, impact)
            self._render_convergence_flash(target, impact)
            if self._flash > 0.0:
                draw_screen_flash(
                    target, config.LASER_EYES_OUTER_COLOR,
                    alpha=int(120 * self._flash),
                )

    # --- Private helpers ----------------------------------------------------

    def _gaze_aim(self, face: FaceData) -> np.ndarray:
        """Magnitude-carrying gaze offset (screen space); (0,0) = looking ahead.

        Unlike the old unit direction, the length encodes how far the user looks
        from centre, which is what lets ``_impact_point`` reach any pixel with no
        dead ring. Returns a copy so the per-activation baseline math never
        mutates the shared FaceData.
        """
        return np.asarray(face.gaze, dtype=np.float32).copy()

    def _render_charge_glow(
        self,
        target: pygame.Surface,
        left: tuple[int, int],
        right: tuple[int, int],
        charge: float,
    ) -> None:
        glow_r = int(8 + 32 * charge)
        for eye in (left, right):
            radial_glow(
                target, eye,
                radius=glow_r,
                color=config.LASER_EYES_CHARGE_GLOW_COLOR,
                alpha=int(180 * charge),
                layers=8,
            )

    def _impact_point(
        self,
        left_eye: tuple[int, int],
        right_eye: tuple[int, int],
        aim: np.ndarray,
    ) -> tuple[int, int]:
        """Single shared impact where both beams converge — reaches ANYWHERE.

        ``aim`` is a magnitude-carrying gaze offset ((0,0) = looking straight
        ahead). The first firing frame captures it as the per-activation baseline
        so the impact starts on the user's own face; thereafter the impact is::

            impact = eye_midpoint + (aim - baseline) * REACH_PX

        Because the offset grows continuously from zero as the head turns and/or
        the eyes move, the impact can be placed on ANY pixel — including the area
        right around the face — with no dead ring. Clamped to the frame.
        """
        mid_x = (left_eye[0] + right_eye[0]) / 2.0
        mid_y = (left_eye[1] + right_eye[1]) / 2.0

        if self._aim_baseline is None:
            self._aim_baseline = aim.copy()
        offset = aim - self._aim_baseline

        reach = config.LASER_EYES_REACH_PX
        ix = int(np.clip(mid_x + float(offset[0]) * reach, 0, self.width - 1))
        iy = int(np.clip(mid_y + float(offset[1]) * reach, 0, self.height - 1))
        return ix, iy

    def _render_beams(
        self,
        target: pygame.Surface,
        left_eye: tuple[int, int],
        right_eye: tuple[int, int],
        impact: tuple[int, int],
    ) -> None:
        """Draw two thin layered beams — left and right eyes CONVERGE to impact."""
        thickness = int(config.LASER_EYES_BEAM_THICKNESS_PX)

        for eye in (left_eye, right_eye):
            beam_pts = [(int(eye[0]), int(eye[1])), (int(impact[0]), int(impact[1]))]

            # Outer soft glow layer (kept narrow so strokes stay legible).
            additive_polyline(
                target, beam_pts,
                color=config.LASER_EYES_OUTER_COLOR,
                width=max(2, thickness * 2),
                alpha=80,
            )
            # Main beam.
            additive_polyline(
                target, beam_pts,
                color=config.LASER_EYES_OUTER_COLOR,
                width=max(1, thickness),
                alpha=220,
            )
            # Bright white core.
            additive_polyline(
                target, beam_pts,
                color=config.LASER_EYES_CORE_COLOR,
                width=max(1, thickness // 3),
                alpha=255,
            )
            # Eye-socket origin glow.
            radial_glow(
                target, eye,
                radius=max(4, thickness * 2),
                color=config.LASER_EYES_CORE_COLOR,
                alpha=200,
                layers=6,
            )

    def _render_convergence_flash(
        self,
        target: pygame.Surface,
        impact: tuple[int, int],
    ) -> None:
        """Bright white-hot flash where the two beams merge at the impact point.

        A small additive hot-core circle and a thin additive ring make the
        convergence visually distinct from a single-beam impact — the two beams
        are seen to combine into one intensely bright point.
        """
        melt_r = int(config.LASER_EYES_MELT_RADIUS_PX)
        flicker = 0.85 + 0.15 * math.sin(self._age * _FLICKER_HZ * math.tau)

        # White-hot convergence core — smaller and brighter than the melt pool.
        core_r = max(3, int(melt_r * 0.4))
        additive_circle(
            target, impact,
            size=core_r,
            color=config.LASER_EYES_CORE_COLOR,
            alpha=int(255 * flicker),
        )

        # Convergence ring — marks where the two beams meet and combine.
        ring_r = max(4, int(melt_r * 0.75))
        additive_ring(
            target, impact,
            radius=ring_r,
            color=config.LASER_EYES_CORE_COLOR,
            alpha=int(180 * flicker),
            width=max(1, melt_r // 10),
        )

    def _render_melt(
        self,
        target: pygame.Surface,
        impact: tuple[int, int],
    ) -> None:
        """Render the live molten scorch (pool, core, char ring, drips) at *impact*."""
        melt_r = int(config.LASER_EYES_MELT_RADIUS_PX)
        ix, iy = int(impact[0]), int(impact[1])

        # Flicker factor: subtle sinusoidal pulse.
        flicker = 0.85 + 0.15 * math.sin(self._age * _FLICKER_HZ * math.tau)

        # Outer molten pool — warm glow.
        pool_alpha = int(180 * flicker)
        radial_glow(
            target, (ix, iy),
            radius=int(melt_r * 1.4),
            color=config.LASER_EYES_MELT_COLOR,
            alpha=pool_alpha,
            layers=10,
        )

        # Hot inner core — smaller, brighter, whiter.
        core_alpha = int(220 * flicker)
        radial_glow(
            target, (ix, iy),
            radius=int(melt_r * 0.55),
            color=config.LASER_EYES_MELT_CORE_COLOR,
            alpha=core_alpha,
            layers=8,
        )

        # Char ring — dark edge around the pool.
        additive_ring(
            target, (ix, iy),
            radius=int(melt_r * 1.05),
            color=(60, 20, 0),
            alpha=int(140 * flicker),
            width=max(2, melt_r // 8),
        )

        # Molten drips running downward from the pool.
        self._render_drips(target, ix, iy, melt_r, flicker)

    def _render_drips(
        self,
        target: pygame.Surface,
        ix: int, iy: int,
        melt_r: int,
        flicker: float,
    ) -> None:
        """Draw molten drips running downward from the impact pool."""
        n = config.LASER_EYES_MELT_DRIP_COUNT
        if n <= 0:
            return

        # Drip length grows with effect age, capped at melt_r * 2.
        max_drip = melt_r * 2.0
        drip_len = min(max_drip, self._age * _DRIP_SPEED)
        if drip_len < 2.0:
            return

        # Spread drips horizontally across the pool diameter.
        for i in range(n):
            seed = i * 137  # deterministic spread — use a local RNG so we
            rng = random.Random(seed)  # don't mutate the shared self._rng state
            spread = (i / max(1, n - 1) - 0.5) * (melt_r * 1.2)
            drip_x = ix + int(spread)
            drip_y_start = iy + int(melt_r * 0.5)

            # Each drip varies slightly in length.
            length_jitter = 0.6 + 0.4 * rng.random()
            drip_end_y = drip_y_start + int(drip_len * length_jitter)

            alpha_drip = int(160 * flicker * (0.7 + 0.3 * rng.random()))
            additive_polyline(
                target,
                [(drip_x, drip_y_start), (drip_x, drip_end_y)],
                color=config.LASER_EYES_MELT_COLOR,
                width=max(1, melt_r // 14),
                alpha=alpha_drip,
            )
            # Bright hot tip at the drip end.
            additive_circle(
                target, (drip_x, drip_end_y),
                size=max(2, melt_r // 16),
                color=config.LASER_EYES_MELT_CORE_COLOR,
                alpha=int(120 * flicker),
            )
