"""Time Freeze — desaturate and cool-tint the camera frame.

Visual specification:
- LAYER_BG.
- pre_process_frame: desaturate the BGR frame toward grey by
  TIME_FREEZE_DESATURATION, then apply a cool blue tint
  TIME_FREEZE_TINT_COLOR (additive, weighted by charge).
  Optional vignette darkens edges.
- While PHASE_ACTIVE and TIME_FREEZE_FREEZE_FRAME is True, the displayed
  frame is locked to the snapshot captured at the moment of activation so
  the user appears frozen on-screen.
- May set signals.time_scale = TIME_FREEZE_TIME_SCALE while active.
- render: nothing visible on the FG guide layer (effect is entirely in the
  camera warp).

Performance notes:
- Grey channel computed with cv2.cvtColor (highly optimised SIMD path).
- Desaturation and tint applied with cv2.addWeighted (avoids float32 alloc
  for the full frame in most cases).
- Vignette precomputed as a float32 mask in __init__; applied with a single
  vectorised multiply — no per-frame mask recomputation.
- All intermediate arrays kept at float32 or uint8; no float64.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import pygame

import config
from core.state import (
    PHASE_ACTIVE,
    PHASE_CHARGING,
    PHASE_RELEASING,
    AbilityState,
    FrameState,
    GestureSignals,
)
from effects.base import LAYER_BG, Effect

log = logging.getLogger(__name__)


class TimeFreezeEffect(Effect):
    layer = LAYER_BG
    name = "time_freeze"
    ability_name = "time_freeze"

    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)

        h, w = height, width

        # Precompute vignette mask as float32 (H, W, 1) in [0, 1].
        # Higher values = darker edges.
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xs / w - 0.5) * 2.0   # -1 … +1
        ny = (ys / h - 0.5) * 2.0
        dist = np.clip(np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0), 0.0, 1.0)
        vignette_strength = np.power(dist, 2.0).astype(np.float32)   # (H, W)
        self._vignette: np.ndarray = vignette_strength[:, :, np.newaxis]  # (H, W, 1)

        # Precompute tint colour as a float32 BGR array for addWeighted.
        tr, tg, tb = config.TIME_FREEZE_TINT_COLOR  # RGB from config
        # cv2.addWeighted needs a same-shape image; we use a 1×1 placeholder
        # and rely on broadcasting via alternative approach (see pre_process_frame).
        self._tint_bgr_f32: np.ndarray = np.array([tb, tg, tr], dtype=np.float32)

        # Preallocate a constant tint image (H, W, 3) of the tint colour.
        # Using a 3-channel image matching frame shape lets addWeighted run fast.
        tint_img = np.empty((h, w, 3), dtype=np.uint8)
        tint_img[..., 0] = int(tb)
        tint_img[..., 1] = int(tg)
        tint_img[..., 2] = int(tr)
        self._tint_img: np.ndarray = tint_img

        # DS=4 vignette: compute the expensive multiply at 1/16th the pixels.
        # Bake the max vignette amount using TIME_FREEZE_FROST_VIGNETTE at init
        # time; per-frame we scale by the frame-variable `charge`.
        _DS = 4
        sw, sh = w // _DS, h // _DS
        self._vig_sw: int = sw
        self._vig_sh: int = sh
        vig_small = cv2.resize(vignette_strength, (sw, sh)).reshape(sh, sw, 1).astype(
            np.float32
        )
        vig_small3 = np.repeat(vig_small, 3, axis=2)            # (sh, sw, 3)
        # Bake max_vignette_amount = TIME_FREEZE_FROST_VIGNETTE (0.55).
        frost_vig = float(config.TIME_FREEZE_FROST_VIGNETTE)
        self._neg_vig_small3_max: np.ndarray = (-vig_small3 * frost_vig).astype(np.float32)

        # Preallocated scratch buffers for the DS=4 vignette pipeline.
        self._scale_s: np.ndarray = np.empty((sh, sw, 3), np.float32)
        self._out_s_f32: np.ndarray = np.empty((sh, sw, 3), np.float32)
        self._out_s_u8: np.ndarray = np.empty((sh, sw, 3), np.uint8)
        self._small_buf: np.ndarray = np.empty((sh, sw, 3), np.uint8)
        self._out_u8: np.ndarray = np.empty((h, w, 3), np.uint8)

        # Preallocated buffers for desaturation and tinting steps.
        self._grey_buf: np.ndarray = np.empty((h, w), np.uint8)
        self._grey_bgr: np.ndarray = np.empty((h, w, 3), np.uint8)
        self._desat_buf: np.ndarray = np.empty((h, w, 3), np.uint8)
        self._tint_buf: np.ndarray = np.empty((h, w, 3), np.uint8)

        # Frozen frame — captured once on PHASE_ACTIVE entry.
        # None means "use the live frame" (CHARGING or freeze-frame disabled).
        self._frozen: np.ndarray | None = None

        # Progressive-slowdown state used while CHARGING: the currently held
        # display frame and a tick counter that decides when to refresh it.
        self._charge_held: np.ndarray | None = None
        self._hold_counter: int = 0

    # --- Lifecycle ----------------------------------------------------------

    def on_enter(self, frame: FrameState, signals: GestureSignals) -> None:
        # Clear any stale frozen frame so a fresh freeze always re-captures.
        self._frozen = None
        self._charge_held = None
        self._hold_counter = 0

    def on_exit(self) -> None:
        self._frozen = None
        self._charge_held = None
        self._hold_counter = 0

    # --- BG processing ------------------------------------------------------

    def pre_process_frame(
        self,
        frame_bgr: np.ndarray,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> np.ndarray:
        charge = float(ability.charge)
        if charge < 0.02:
            return frame_bgr

        # Progressive slow-down while CHARGING: hold the displayed frame for an
        # increasing number of ticks as the charge rises (charge² curve), so the
        # user's motion visibly stutters slower and slower and reaches a complete
        # stop right as the freeze (PHASE_ACTIVE) lands.
        if (
            ability.phase == PHASE_CHARGING
            and config.TIME_FREEZE_FREEZE_FRAME
            and frame_bgr.shape == (self.height, self.width, 3)
        ):
            hold_ticks = 1 + int(round(
                charge * charge * config.TIME_FREEZE_SLOWDOWN_MAX_HOLD_FRAMES
            ))
            self._hold_counter += 1
            if self._charge_held is None or self._hold_counter >= hold_ticks:
                self._charge_held = frame_bgr.copy()
                self._hold_counter = 0
            frame_bgr = self._charge_held

        # Freeze-frame: lock displayed video to the snapshot taken at
        # PHASE_ACTIVE entry so the user appears frozen on screen. The lock is
        # also held through PHASE_RELEASING — that's the TIME_FREEZE_SHATTER_DELAY
        # window where the user stays frozen until the glass shatters and the
        # frame un-freezes together (on cooldown, when this effect deactivates).
        if (
            ability.phase in (PHASE_ACTIVE, PHASE_RELEASING)
            and config.TIME_FREEZE_FREEZE_FRAME
        ):
            if self._frozen is None:
                # First active tick — capture the frame.
                if frame_bgr.shape == (self.height, self.width, 3):
                    self._frozen = frame_bgr.copy()
                else:
                    log.warning(
                        "time_freeze: unexpected frame shape %s, skipping freeze",
                        frame_bgr.shape,
                    )
            if self._frozen is not None:
                frame_bgr = self._frozen

        desat_amount = float(config.TIME_FREEZE_DESATURATION) * charge
        tint_strength = 0.18 * charge

        # 1. Desaturate: blend frame toward its greyscale version.
        #    cv2.cvtColor gives a BT.601-weighted grey (matches original weights).
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=self._grey_buf)
        cv2.cvtColor(self._grey_buf, cv2.COLOR_GRAY2BGR, dst=self._grey_bgr)

        # Blend: result = frame*(1-desat) + grey*desat
        cv2.addWeighted(
            frame_bgr, 1.0 - desat_amount,
            self._grey_bgr, desat_amount,
            0.0,
            dst=self._desat_buf,
        )

        # 2. Cool tint: blend desaturated toward tint colour.
        cv2.addWeighted(
            self._desat_buf, 1.0 - tint_strength,
            self._tint_img, tint_strength,
            0.0,
            dst=self._tint_buf,
        )
        tinted = self._tint_buf

        # 3. Vignette — DS=4: apply vignette on 1/16th-pixel count, upscale back.
        #    Full-res (H,W,3) float32 multiply costs ~8 ms; coarse (sh,sw,3)
        #    multiply costs ~0.14 ms + 0.36 ms NEAREST upscale = ~0.5 ms total.
        charge_f32 = np.float32(charge)
        cv2.resize(
            tinted, (self._vig_sw, self._vig_sh),
            dst=self._small_buf, interpolation=cv2.INTER_AREA,
        )
        # scale = 1 + neg_vig_small3_max * charge   (baked max already = -vig*0.55)
        np.multiply(self._neg_vig_small3_max, charge_f32, out=self._scale_s)
        self._scale_s += np.float32(1.0)
        np.multiply(self._small_buf, self._scale_s, out=self._out_s_f32)
        cv2.convertScaleAbs(self._out_s_f32, dst=self._out_s_u8)
        cv2.resize(
            self._out_s_u8, (self.width, self.height),
            dst=self._out_u8, interpolation=cv2.INTER_NEAREST,
        )
        result = self._out_u8

        # 4. Signal time dilation hint to the engine.
        if ability.phase == PHASE_ACTIVE:
            signals.time_scale = min(
                signals.time_scale,
                config.TIME_FREEZE_TIME_SCALE,
            )

        return result

    # --- No FG overlay needed -----------------------------------------------

    def render(
        self,
        target: pygame.Surface,
        signals: GestureSignals,
        ability: AbilityState,
    ) -> None:
        pass
