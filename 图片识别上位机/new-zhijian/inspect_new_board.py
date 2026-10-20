# -*- coding: utf-8 -*-
"""Inspect replacement-board geometry without the old dual-ring filters."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_three_ways import preview, save_image, to_gray  # noqa: E402
from ring_detector import detect_rings  # noqa: E402

SRC = Path(__file__).resolve().parent / "captured" / "new_board" / "full.jpg"
OUT = Path(__file__).resolve().parent / "captured" / "new_board"


def main() -> None:
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = to_gray(img)
    h, w = gray.shape
    print("shape", w, h, "mean", round(float(gray.mean()), 1))

    # Center crop of the cross
    cx, cy = w // 2, h // 2
    for name, x0, y0, ww, hh in (
        ("center", cx - 400, cy - 400, 800, 800),
        ("h_arm", 200, cy - 200, 900, 400),
        ("v_arm", cx - 200, 200, 400, 900),
    ):
        x0 = max(0, x0)
        y0 = max(0, y0)
        crop = gray[y0 : y0 + hh, x0 : x0 + ww]
        save_image(OUT / f"crop_{name}.jpg", preview(crop, 900))
        save_image(
            OUT / f"crop_{name}_near.png",
            cv2.resize(crop, (crop.shape[1] * 3, crop.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
            if max(crop.shape) < 500
            else crop,
        )

    det = detect_rings(img, source_dpi=1200, threshold_width_px=3, sr_scale=1, allow_small=True)
    print("raw detect", len(det.results))
    sizes = [(min(r.w, r.h), max(r.w, r.h), round(max(r.w, r.h) / max(1.0, min(r.w, r.h)), 2)) for r in det.results]
    minors = [s[0] for s in sizes]
    majors = [s[1] for s in sizes]
    print(
        "minor p10/50/90",
        np.percentile(minors, 10),
        np.percentile(minors, 50),
        np.percentile(minors, 90),
    )
    print(
        "major p10/50/90",
        np.percentile(majors, 10),
        np.percentile(majors, 50),
        np.percentile(majors, 90),
    )

    # Bright elongated blobs: stadiums at this height were ~240x68 on previous board
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = int(np.clip(np.percentile(blur, 85), 80, 240))
    _, bw = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    blobs = []
    for i in range(1, n):
        x, y, ww, hh, area = (int(st[i, j]) for j in range(5))
        if area < 200:
            continue
        aspect = max(ww, hh) / max(1.0, min(ww, hh))
        if aspect < 1.8 or min(ww, hh) < 18:
            continue
        blobs.append((x, y, ww, hh, area, aspect))
    blobs.sort(key=lambda b: -b[4])
    print("elongated blobs", len(blobs))
    if blobs:
        print("top sizes", [(b[2], b[3], round(b[5], 2)) for b in blobs[:12]])
        print(
            "blob minor/major median",
            float(np.median([min(b[2], b[3]) for b in blobs])),
            float(np.median([max(b[2], b[3]) for b in blobs])),
        )

    vis = img.copy()
    for x, y, ww, hh, *_ in blobs:
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 220, 255), 2)
    save_image(OUT / "blobs_preview.jpg", preview(vis, 1800))

    # Cross-section of the largest blob
    if blobs:
        x, y, ww, hh, *_ = blobs[0]
        roi = gray[y : y + hh, x : x + ww]
        save_image(
            OUT / "blob0_near.png",
            cv2.resize(roi, (roi.shape[1] * 6, roi.shape[0] * 6), interpolation=cv2.INTER_NEAREST),
        )
        if ww >= hh:
            x0, x1 = int(ww * 0.2), int(ww * 0.8)
            prof = np.median(roi[:, x0:x1].astype(np.float32), axis=1)
        else:
            y0, y1 = int(hh * 0.2), int(hh * 0.8)
            prof = np.median(roi[y0:y1, :].astype(np.float32), axis=0)
        print("largest blob profile")
        for i, v in enumerate(prof):
            print(f"{i:3d} {v:6.1f} {'#' * int(max(0.0, v) / 10)}")


if __name__ == "__main__":
    main()
