# -*- coding: utf-8 -*-
"""Grab one frame and judge if current WD can resolve 0.2 mm dual rings."""

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
    dual_peak_score,
    find_ring_crops,
    ge,
    gf,
    preview,
    sample_profiles,
    save_image,
    to_gray,
)
from hikrobot_camera import HikrobotCamera  # noqa: E402

OUT = Path(__file__).resolve().parent / "captured" / "height_check"


def fwhm_peaks(profile: np.ndarray) -> list[int]:
    p = profile.astype(np.float32)
    if p.size < 12:
        return []
    k = max(3, p.size // 50)
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    base = np.percentile(sm, 20)
    peak = np.percentile(sm, 92)
    thr = 0.5 * (base + peak)
    widths = []
    inside = False
    start = 0
    for i, v in enumerate(sm):
        if not inside and v >= thr:
            inside = True
            start = i
        elif inside and v < thr:
            inside = False
            widths.append(i - start)
    return widths


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
        frame = camw.grab_bgr(8000)
    camw.disconnect()
    print("disconnected")
    if frame is None:
        raise SystemExit("no frame")

    h, w = frame.shape[:2]
    gray = to_gray(frame)
    save_image(OUT / "full.jpg", frame)
    save_image(OUT / "preview.jpg", preview(frame, 1600))
    print(
        f"shape {w}x{h} mean={gray.mean():.1f} "
        f"p5={np.percentile(gray, 5):.0f} p50={np.percentile(gray, 50):.0f} "
        f"p95={np.percentile(gray, 95):.0f} sat={(gray >= 250).mean() * 100:.2f}%"
    )

    boxes = find_ring_crops(gray, k=6)
    print("crops", boxes[:6])
    all_w: list[float] = []
    all_sep: list[float] = []
    for i, box in enumerate(boxes[:5], 1):
        save_image(OUT / f"zoom_{i}.png", draw_profile_panel(gray, box, f"ring{i}"))
        x, y, ww, hh = box
        crop = gray[y : y + hh, x : x + ww]
        if crop.size == 0:
            continue
        near = cv2.resize(
            crop,
            (crop.shape[1] * 5, crop.shape[0] * 5),
            interpolation=cv2.INTER_NEAREST,
        )
        save_image(OUT / f"near_{i}.png", near)
        for prof in sample_profiles(gray, box, n=11):
            sc = dual_peak_score(prof)
            if sc.get("sep_px"):
                all_sep.append(float(sc["sep_px"]))
            all_w.extend(float(v) for v in fwhm_peaks(prof))

    if all_w:
        arr = np.array(all_w, dtype=float)
        arr = arr[(arr >= 2) & (arr <= 40)]
        if arr.size:
            print(
                f"trace FWHM px: n={len(arr)} median={np.median(arr):.2f} "
                f"p25={np.percentile(arr, 25):.2f} p75={np.percentile(arr, 75):.2f}"
            )
    if all_sep:
        sep = np.array(all_sep, dtype=float)
        print(f"dual-peak sep px: n={len(sep)} median={np.median(sep):.2f}")

    thr = int(np.clip(np.percentile(gray, 88), 70, 230))
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    pads = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        ww, hh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 400:
            continue
        aspect = max(ww, hh) / max(1.0, min(ww, hh))
        if aspect < 2.2:
            continue
        pads.append((max(ww, hh), min(ww, hh), area))
    pads = sorted(pads, reverse=True)[:30]
    if pads:
        majors = np.array([p[0] for p in pads], float)
        minors = np.array([p[1] for p in pads], float)
        print(
            f"elongated blobs n={len(pads)} major_med={np.median(majors):.1f}px "
            f"minor_med={np.median(minors):.1f}px"
        )
        um_major = 9200.0 / float(np.median(majors))
        um_minor = 2200.0 / float(np.median(minors))
        print(f"if pad length 9.2mm -> {um_major:.1f} um/px, 0.2mm={200 / um_major:.1f}px")
        print(f"if pad width  2.2mm -> {um_minor:.1f} um/px, 0.2mm={200 / um_minor:.1f}px")
        print(f"FOV ~ {w * um_major / 1000:.0f} x {h * um_major / 1000:.0f} mm (from length)")
        wd = um_major * 12 / 2.4
        print(f"approx WD {wd:.0f} mm")
    print("saved", OUT)


if __name__ == "__main__":
    main()
