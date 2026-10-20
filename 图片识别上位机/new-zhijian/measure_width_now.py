# -*- coding: utf-8 -*-
"""Grab one live frame and measure inner/outer copper ring widths."""

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
from ring_detector import detect_rings  # noqa: E402

OUT = Path(__file__).resolve().parent / "captured" / "width_now"


def peak_widths(profile: np.ndarray) -> list[dict]:
    """FWHM of the two strongest bright peaks along a 1D cut."""
    p = profile.astype(np.float32)
    if p.size < 16:
        return []
    k = max(3, p.size // 60)
    sm = np.convolve(p, np.ones(k) / k, mode="same")
    base = float(np.percentile(sm, 15))
    peak_h = float(np.percentile(sm, 90))
    if peak_h - base < 8:
        return []
    thr = base + 0.45 * (peak_h - base)
    segs = []
    inside = False
    start = 0
    for i, v in enumerate(sm):
        if not inside and v >= thr:
            inside = True
            start = i
        elif inside and v < thr:
            inside = False
            segs.append((start, i))
    if inside:
        segs.append((start, len(sm) - 1))
    out = []
    for a, b in segs:
        if b - a < 2:
            continue
        chunk = sm[a : b + 1]
        mx = float(chunk.max())
        mid = base + 0.5 * (mx - base)
        left = a
        right = b
        for i in range(a, b + 1):
            if sm[i] >= mid:
                left = i
                break
        for i in range(b, a - 1, -1):
            if sm[i] >= mid:
                right = i
                break
        out.append({"fwhm": float(right - left + 1), "amp": mx, "center": 0.5 * (left + right)})
    out.sort(key=lambda d: d["amp"], reverse=True)
    out = sorted(out[:2], key=lambda d: d["center"])
    return out


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

    gray = to_gray(frame)
    h, w = gray.shape
    save_image(OUT / "full.jpg", frame)
    save_image(OUT / "preview.jpg", preview(frame, 1600))
    print(f"shape {w}x{h} mean={gray.mean():.1f} sat={(gray >= 250).mean()*100:.2f}%")

    boxes = find_ring_crops(gray, k=8)
    inners, outers, seps = [], [], []
    used = 0
    for i, box in enumerate(boxes[:6], 1):
        save_image(OUT / f"zoom_{i}.png", draw_profile_panel(gray, box, f"ring{i}"))
        x, y, ww, hh = box
        crop = gray[y : y + hh, x : x + ww]
        if crop.size:
            near = cv2.resize(crop, (crop.shape[1] * 5, crop.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
            save_image(OUT / f"near_{i}.png", near)
        local_in, local_out, local_sep = [], [], []
        for prof in sample_profiles(gray, box, n=13):
            peaks = peak_widths(prof)
            if len(peaks) < 2:
                continue
            w0, w1 = peaks[0]["fwhm"], peaks[1]["fwhm"]
            sep = abs(peaks[1]["center"] - peaks[0]["center"])
            if sep < 4 or sep > 80:
                continue
            # outer peak is usually the first or the brighter one at the rim; keep both
            local_in.append(min(w0, w1))
            local_out.append(max(w0, w1))
            local_sep.append(sep)
        if local_in:
            used += 1
            inners.append(float(np.median(local_in)))
            outers.append(float(np.median(local_out)))
            seps.append(float(np.median(local_sep)))
            print(
                f"  crop{i}: inner~{inners[-1]:.2f}px  outer~{outers[-1]:.2f}px  "
                f"peak-sep~{seps[-1]:.1f}px  n={len(local_in)}"
            )

    print("\n=== profile FWHM (this lighting) ===")
    if inners:
        print(f"inner copper  median {np.median(inners):.2f} px  p25 {np.percentile(inners,25):.2f}  p75 {np.percentile(inners,75):.2f}")
        print(f"outer copper  median {np.median(outers):.2f} px  p25 {np.percentile(outers,25):.2f}  p75 {np.percentile(outers,75):.2f}")
        print(f"two-peak sep  median {np.median(seps):.2f} px")
    else:
        print("no dual-peak profiles")

    # scale from elongated pad length ~9.2 mm (previous geometry)
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
        if aspect < 2.2:
            continue
        majors.append(max(ww, hh))
    um_px = None
    if majors:
        major = float(np.median(majors))
        um_px = 9200.0 / major
        print(f"\npad major median {major:.1f}px  scale if 9.2mm -> {um_px:.1f} um/px")
        print(f"FOV ~ {w * um_px / 1000:.0f} x {h * um_px / 1000:.0f} mm")
        print(f"0.2mm would be {200 / um_px:.1f} px")
        if inners:
            print("\n=== estimated mm (NOT calibrated ruler; pad-length scale) ===")
            print(f"inner ~ {np.median(inners) * um_px / 1000:.3f} mm")
            print(f"outer ~ {np.median(outers) * um_px / 1000:.3f} mm")
            print(f"gap(center-to-center minus half widths) ~ {(np.median(seps) - 0.5 * np.median(inners) - 0.5 * np.median(outers)) * um_px / 1000:.3f} mm")

    print("\n=== detect_rings() ===")
    det = detect_rings(frame, source_dpi=1200, threshold_width_px=3, sr_scale=1, allow_small=True)
    save_image(OUT / "annotated.jpg", det.annotated_bgr)
    print(f"found {len(det.results)} rings")
    di = [r.inner_line_width_px for r in det.results if r.inner_line_width_px >= 0.8]
    do = [r.outer_line_width_px for r in det.results if r.outer_line_width_px >= 0.8]
    if di:
        print(f"detector inner median {np.median(di):.2f} px  outer {np.median(do):.2f} px  n={len(di)}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
