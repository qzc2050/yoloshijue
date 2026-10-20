# -*- coding: utf-8 -*-
"""Grab one live frame of the replacement PCB and save analysis assets."""

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
    sample_profiles,
    save_image,
    to_gray,
)
from hikrobot_camera import HikrobotCamera  # noqa: E402
from measure_width_now import peak_widths  # noqa: E402

OUT = Path(__file__).resolve().parent / "captured" / "new_board"


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pad_scale(gray: np.ndarray) -> tuple[float | None, float]:
    thr = int(np.clip(np.percentile(gray, 88), 70, 230))
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    majors = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        ww, hh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 400:
            continue
        aspect = max(ww, hh) / max(1.0, min(ww, hh))
        if 2.2 <= aspect <= 8:
            majors.append(max(ww, hh))
    if not majors:
        return None, 0.0
    major = float(np.median(majors))
    return 9200.0 / major, major


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
    um_px, major = pad_scale(gray)
    print(
        f"shape {w}x{h} mean={gray.mean():.1f} p50={np.percentile(gray,50):.0f} "
        f"p95={np.percentile(gray,95):.0f} sat={(gray>=250).mean()*100:.2f}% "
        f"lap={sharpness(gray):.1f}"
    )
    if um_px:
        print(
            f"pad_major={major:.0f}px scale={um_px:.2f} um/px "
            f"FOV~{w*um_px/1000:.0f}x{h*um_px/1000:.0f}mm "
            f"WD~{um_px*5:.0f}mm 0.2mm={200/um_px:.1f}px"
        )

    save_image(OUT / "full.jpg", frame)
    save_image(OUT / "preview.jpg", preview(frame, 1600))
    boxes = find_ring_crops(gray, k=8)
    for i, box in enumerate(boxes[:6], 1):
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
        inners, outers, seps = [], [], []
        for prof in sample_profiles(gray, box, n=13):
            peaks = peak_widths(prof)
            if len(peaks) < 2:
                continue
            sep = abs(peaks[1]["center"] - peaks[0]["center"])
            if sep < 4 or sep > 80:
                continue
            inners.append(min(peaks[0]["fwhm"], peaks[1]["fwhm"]))
            outers.append(max(peaks[0]["fwhm"], peaks[1]["fwhm"]))
            seps.append(sep)
        if inners:
            print(
                f"  crop{i}: inner~{np.median(inners):.1f}px outer~{np.median(outers):.1f}px "
                f"sep~{np.median(seps):.1f}px n={len(inners)}"
            )
    print("saved", OUT)


if __name__ == "__main__":
    main()
