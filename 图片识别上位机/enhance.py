# -*- coding: utf-8 -*-
"""Local resolution compensation for whole-board single-shot inspection.

A full carbon-ink board in one frame cannot also give many pixels per trace.
Instead of rescanning tiles, each detected ring is deskewed and upsampled
before line-width metrology (Lanczos/cubic + unsharp). Optional burst fusion
is available if the camera later supplies multiple aligned frames.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


def unsharp_gray(image: np.ndarray, amount: float = 0.55, sigma: float = 1.05) -> np.ndarray:
    """Restore high-frequency edges after interpolation."""
    if image.size == 0:
        return image
    src = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)
    sharp = cv2.addWeighted(src, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def upsample_gray(image: np.ndarray, scale: int) -> np.ndarray:
    """Single-image super-resolution stand-in: Lanczos upsample + unsharp."""
    scale = int(max(1, scale))
    if scale == 1:
        return unsharp_gray(image, amount=0.35, sigma=0.8)
    work = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    up = cv2.resize(work, (0, 0), fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LANCZOS4)
    return unsharp_gray(up, amount=0.5, sigma=1.0)


def fuse_burst(frames: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    """Align frames to the first with ECC and take a median (true multi-frame SR)."""
    stack = [ensure_u8(frame) for frame in frames if frame is not None and getattr(frame, "size", 0)]
    if not stack:
        return None
    if len(stack) == 1:
        return stack[0]
    reference = stack[0]
    aligned = [reference]
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    for frame in stack[1:]:
        if frame.shape[:2] != reference.shape[:2]:
            frame = cv2.resize(frame, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
        try:
            _, warp = cv2.findTransformECC(
                reference,
                frame,
                warp,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                1,
            )
            aligned.append(
                cv2.warpAffine(
                    frame,
                    warp,
                    (reference.shape[1], reference.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            )
        except cv2.error:
            aligned.append(frame)
    return np.median(np.stack(aligned, axis=0), axis=0).astype(np.uint8)


def ensure_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def px_to_um(width_px: float, source_dpi: float) -> float:
    if source_dpi <= 0 or width_px <= 0:
        return 0.0
    return float(width_px) * 25400.0 / float(source_dpi)
