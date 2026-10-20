# -*- coding: utf-8 -*-
"""Try OpenCV steps on the saved full frame and write a small preview of each one.

Production inspection stays in frontlight.py. Add a step here, run this file,
look at samples/lab, and only then move a step that works into the analyzer.

    python -m qc_host.vision.lab
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "samples" / "board4.png"
OUT = ROOT / "samples" / "lab"
PREVIEW_MAX = 1600


def _odd(n: float) -> int:
    k = max(3, int(round(n)))
    return k if k % 2 else k + 1


def _px(shape: tuple[int, ...]) -> float:
    return max(1.0, float(max(shape[:2])) / 1800.0)


def _preview(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, PREVIEW_MAX / max(height, width))
    if scale >= 0.999:
        return image
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _save(name: str, image: np.ndarray) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, _preview(image))
    if not ok:
        raise RuntimeError(f"encode failed: {name}")
    path.write_bytes(buf.tobytes())
    return path


def _gold_mask(bgr: np.ndarray, gray: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(bgr)
    gold = (red.astype(np.int16) - blue.astype(np.int16) >= 8) & (
        red.astype(np.int16) >= green.astype(np.int16) - 12
    )
    raw = gold & (contrast > 2.0) & (gray > 12)
    return np.where(raw, 255, 0).astype(np.uint8)


def _ridge_mask(gray: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    raw = (contrast > 6.0) & (gray > 10)
    return np.where(raw, 255, 0).astype(np.uint8)


def _morph(mask: np.ndarray, px: float) -> np.ndarray:
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd(3 * min(px, 1.7)), _odd(3 * min(px, 1.7)))
    )
    out = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=1)


def _draw_rings(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = _preview(bgr)
    small = _preview(mask)
    contours, hierarchy = cv2.findContours(small, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return vis
    hier = hierarchy[0]
    for index, cnt in enumerate(contours):
        if hier[index][3] != -1 or hier[index][2] == -1:
            continue
        if cv2.contourArea(cnt) < 400:
            continue
        cv2.drawContours(vis, [cnt], -1, (80, 220, 80), 2, cv2.LINE_AA)
    return vis


def run(source: Path = SOURCE) -> list[Path]:
    if not source.is_file():
        raise FileNotFoundError(source)
    data = np.fromfile(source, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read {source}")
    px = _px(bgr.shape)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    den = cv2.medianBlur(gray, _odd(3 * min(px, 1.7)))
    local = cv2.GaussianBlur(den.astype(np.float32), (0, 0), 12.0 * px)
    contrast = den.astype(np.float32) - local
    flat = np.clip(contrast + 128.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    gold = _gold_mask(bgr, den, contrast)
    ridge = _ridge_mask(den, contrast)
    copper = _morph(cv2.bitwise_or(gold, ridge), px)
    rings = _draw_rings(bgr, copper)

    saved = [
        _save("01_gray.png", den),
        _save("02_light.png", np.clip(local, 0, 255).astype(np.uint8)),
        _save("03_flat.png", flat),
        _save("04_clahe.png", clahe),
        _save("05_gold.png", gold),
        _save("06_ridge.png", ridge),
        _save("07_copper.png", copper),
        _save("08_rings.png", rings),
    ]
    return saved


def main() -> None:
    paths = run()
    for path in paths:
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
