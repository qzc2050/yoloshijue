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

SRC = Path(__file__).resolve().parent / "captured" / "new_board" / "full.jpg"
OUT = Path(__file__).resolve().parent / "captured" / "new_board"
UM = 41.07  # same height as previous raised shot


def main() -> None:
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = int(np.clip(np.percentile(blur, 85), 80, 240))
    _, bw = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    chambers = []
    for i in range(1, n):
        x, y, ww, hh, area = (int(st[i, j]) for j in range(5))
        minor, major = min(ww, hh), max(ww, hh)
        if minor < 50 or minor > 120 or major < 160 or major > 320:
            continue
        aspect = major / max(1.0, minor)
        if aspect < 2.2 or aspect > 5.5:
            continue
        chambers.append((x, y, ww, hh, area))
    chambers.sort(key=lambda b: (b[1] // 80, b[0]))
    print("chamber-like blobs", len(chambers))
    vis = img.copy()
    for i, (x, y, ww, hh, _) in enumerate(chambers, 1):
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 220, 255), 2)
        cv2.putText(vis, str(i), (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    save_image(OUT / "new_chambers_preview.jpg", preview(vis, 1800))

    split = 0
    merged = 0
    inners = []
    outers = []
    seps = []
    for i, (x, y, ww, hh, _) in enumerate(chambers[:12], 1):
        roi = gray[y : y + hh, x : x + ww]
        save_image(
            OUT / f"nb_stadium_{i}.png",
            cv2.resize(roi, (roi.shape[1] * 5, roi.shape[0] * 5), interpolation=cv2.INTER_NEAREST),
        )
        if ww >= hh:
            x0, x1 = int(ww * 0.2), int(ww * 0.8)
            prof = np.median(roi[:, x0:x1].astype(np.float32), axis=1)
        else:
            y0, y1 = int(hh * 0.2), int(hh * 0.8)
            prof = np.median(roi[y0:y1, :].astype(np.float32), axis=0)
        sm = np.convolve(prof, np.ones(3) / 3, mode="same")
        n2 = len(sm) // 2
        halves = [(sm[:n2], 0), (sm[n2:], n2)]
        local_in = []
        local_out = []
        local_sep = []
        for half, off in halves:
            if half.size < 8:
                continue
            base = float(np.percentile(half, 20))
            peaks = [
                j
                for j in range(2, len(half) - 2)
                if half[j] >= half[j - 1]
                and half[j] >= half[j + 1]
                and half[j] > base + 15
            ]
            keep = []
            for j in peaks:
                if not keep or j - keep[-1] > 3:
                    keep.append(j)
                elif half[j] > half[keep[-1]]:
                    keep[-1] = j
            if len(keep) >= 2:
                a, b = keep[0], keep[-1]
                local_sep.append(abs(b - a))
                local_out.append(float(min(a, b) + 1))  # placeholder
                # FWHM around each peak
                ws = []
                for pk in (a, b):
                    halfv = base + 0.5 * (half[pk] - base)
                    L = pk
                    while L > 0 and half[L] >= halfv:
                        L -= 1
                    R = pk
                    while R < len(half) - 1 and half[R] >= halfv:
                        R += 1
                    ws.append(float(R - L))
                local_out[-1] = max(ws)
                local_in.append(min(ws))
                split += 1
            else:
                merged += 1
        if local_in:
            inners.append(float(np.median(local_in)))
            outers.append(float(np.median(local_out)))
            seps.append(float(np.median(local_sep)))
            print(
                f"#{i} {ww}x{hh} inner={inners[-1]:.1f}px outer={outers[-1]:.1f}px "
                f"sep={seps[-1]:.1f}px"
            )
        else:
            print(f"#{i} {ww}x{hh} NO dual peak")

        if i == 1:
            print("profile")
            for k, v in enumerate(sm):
                print(f"{k:3d} {v:6.1f} {'#' * int(max(0.0, v) / 10)}")

    print("halves split", split, "merged", merged)
    if inners:
        print(
            "median inner", np.median(inners), "outer", np.median(outers), "sep", np.median(seps)
        )
        print(
            f"mm inner {np.median(inners)*UM/1000:.3f} outer {np.median(outers)*UM/1000:.3f} "
            f"0.2mm={200/UM:.1f}px"
        )


if __name__ == "__main__":
    main()
