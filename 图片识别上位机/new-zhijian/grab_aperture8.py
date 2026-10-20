# -*- coding: utf-8 -*-
"""Grab one live frame after aperture ~f/8-9 and save comparison assets."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_three_ways import (  # noqa: E402
    draw_profile_panel,
    find_ring_crops,
    ge,
    gf,
    preview,
    save_image,
    to_gray,
)
from hikrobot_camera import HikrobotCamera  # noqa: E402

OUT = Path(__file__).resolve().parent / "captured" / "aperture_8"
OLD = Path(__file__).resolve().parent / "captured" / "width_now" / "full.jpg"


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    camw = HikrobotCamera()
    print("enum", camw.enumerate())
    print("connect", camw.connect("192.168.0.100"))
    cam = camw.cam
    print("Exposure", gf(cam, "ExposureTime"))
    print("Gain", gf(cam, "Gain"))
    print("PixelFormat", ge(cam, "PixelFormat"))
    frame = None
    for _ in range(4):
        frame = camw.grab_bgr(12000)
    camw.disconnect()
    print("disconnected")
    if frame is None:
        raise SystemExit("no frame")

    gray = to_gray(frame)
    h, w = gray.shape
    sat = float((gray >= 250).mean() * 100)
    print(
        f"shape {w}x{h} mean={gray.mean():.1f} p50={np.percentile(gray,50):.0f} "
        f"p95={np.percentile(gray,95):.0f} sat={sat:.2f}% sharp={sharpness(gray):.1f}"
    )
    save_image(OUT / "full.jpg", frame)
    save_image(OUT / "preview.jpg", preview(frame, 1600))

    if OLD.exists():
        old = cv2.imdecode(np.fromfile(str(OLD), dtype=np.uint8), cv2.IMREAD_COLOR)
        if old is not None:
            og = to_gray(old)
            print(
                f"OLD  mean={og.mean():.1f} sat={(og>=250).mean()*100:.2f}% "
                f"sharp={sharpness(og):.1f}"
            )

    for i, box in enumerate(find_ring_crops(gray, k=8)[:6], 1):
        save_image(OUT / f"zoom_{i}.png", draw_profile_panel(gray, box, f"ring{i}"))
        x, y, ww, hh = box
        crop = gray[y : y + hh, x : x + ww]
        if crop.size:
            near = cv2.resize(
                crop,
                (crop.shape[1] * 5, crop.shape[0] * 5),
                interpolation=cv2.INTER_NEAREST,
            )
            save_image(OUT / f"near_{i}.png", near)
    print("saved", OUT)


if __name__ == "__main__":
    main()
