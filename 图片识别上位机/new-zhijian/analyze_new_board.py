# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_three_ways import preview, save_image, to_gray  # noqa: E402
from list_chambers import center_nms, nms  # noqa: E402
from ring_detector import _measure_dual_ring_widths, detect_rings  # noqa: E402

SRC = Path(__file__).resolve().parent / "captured" / "new_board" / "full.jpg"
OUT = Path(__file__).resolve().parent / "captured" / "new_board"
UM_PX = 41.07


def main() -> None:
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = to_gray(img)
    det = detect_rings(img, source_dpi=1200, threshold_width_px=3, sr_scale=1, allow_small=True)
    print("raw", len(det.results))
    raw = []
    for r in det.results:
        aspect = max(r.w, r.h) / max(1.0, min(r.w, r.h))
        if min(r.w, r.h) < 20 or max(r.w, r.h) < 50 or aspect < 1.55:
            continue
        raw.append(r)
    keep = nms([(r.x, r.y, r.w, r.h) for r in raw], 0.35)
    ch = center_nms([raw[i] for i in keep])
    print(
        "chambers",
        len(ch),
        "minor",
        float(np.median([min(r.w, r.h) for r in ch])),
        "major",
        float(np.median([max(r.w, r.h) for r in ch])),
    )

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    samp = sorted(ch, key=lambda r: -(r.w * r.h))[:8]
    for i, r in enumerate(samp, 1):
        a = _measure_dual_ring_widths(gray, binary, r.x, r.y, r.w, r.h)
        inner_min, outer_min, inner_med, outer_med = a[:4]
        print(
            f"#{i} {r.w}x{r.h} xy=({r.x},{r.y}) "
            f"inner_med={inner_med:.2f}px/{inner_med * UM_PX / 1000:.3f}mm "
            f"outer_med={outer_med:.2f}px/{outer_med * UM_PX / 1000:.3f}mm"
        )
        crop = gray[r.y : r.y + r.h, r.x : r.x + r.w]
        if crop.size:
            save_image(
                OUT / f"stadium_{i}.png",
                cv2.resize(
                    crop,
                    (crop.shape[1] * 6, crop.shape[0] * 6),
                    interpolation=cv2.INTER_NEAREST,
                ),
            )

    r = samp[0]
    roi = gray[r.y : r.y + r.h, r.x : r.x + r.w]
    hh, ww = roi.shape
    if ww >= hh:
        x0, x1 = int(ww * 0.2), int(ww * 0.8)
        prof = np.median(roi[:, x0:x1].astype(np.float32), axis=1)
        axis = "y"
    else:
        y0, y1 = int(hh * 0.2), int(hh * 0.8)
        prof = np.median(roi[y0:y1, :].astype(np.float32), axis=0)
        axis = "x"
    print("profile along", axis, "len", len(prof))
    for i, v in enumerate(prof):
        bar = "#" * int(max(0.0, v) / 10)
        print(f"{i:3d} {v:6.1f} {bar}")

    vis = img.copy()
    for r in ch:
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 220, 255), 2)
    save_image(OUT / "chambers_preview.jpg", preview(vis, 1800))
    print("preview saved n=", len(ch))


if __name__ == "__main__":
    main()
