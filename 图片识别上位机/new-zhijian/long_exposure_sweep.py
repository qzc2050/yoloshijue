# -*- coding: utf-8 -*-
"""Long-exposure / gain sweep after discovering the scene is optically dark."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_three_ways import (  # noqa: E402
    OUT,
    analyze,
    draw_profile_panel,
    preview,
    save_image,
    set_enum_str,
    set_float,
    to_gray,
)
from hikrobot_camera import HikrobotCamera  # noqa: E402


def grab_stable(camw: HikrobotCamera) -> np.ndarray | None:
    last = None
    for _ in range(5):
        try:
            last = camw.grab_bgr(15000)
        except RuntimeError as exc:
            print("  grab err", exc)
    return last


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    camw = HikrobotCamera()
    print("connect", camw.connect("192.168.0.100"))
    cam = camw.cam
    set_enum_str(cam, "ExposureAuto", "Off")
    set_enum_str(cam, "GainAuto", "Off")

    combos = []
    for gain in (0.0, 6.0, 12.0):
        for exp in (100000, 200000, 289000, 500000, 800000):
            combos.append((gain, exp))

    best = None
    best_frame = None
    best_gray = None
    for gain, exp in combos:
        set_float(cam, "Gain", gain)
        set_float(cam, "ExposureTime", float(exp))
        time.sleep(0.2)
        frame = grab_stable(camw)
        if frame is None:
            print(f"skip g{gain} e{int(exp)}")
            continue
        gray = to_gray(frame)
        tag = f"g{int(gain)}_e{int(exp / 1000)}ms"
        save_image(OUT / f"long_{tag}.jpg", frame)
        save_image(OUT / f"long_{tag}_preview.jpg", preview(frame))
        print(
            f"{tag:16s} mean={gray.mean():6.1f} p50={np.percentile(gray,50):5.1f} "
            f"p95={np.percentile(gray,95):6.1f} max={int(gray.max())} "
            f"sat={(gray >= 250).mean()*100:5.2f}%"
        )
        rep = analyze(frame, tag)
        take = False
        if best is None:
            take = True
        else:
            copper = rep["copper"]
            # prefer copper in 160-230, then higher contrast without sat
            def rank(r):
                c = r["copper"]
                target = 1.0 - abs(c - 200) / 200.0
                return target * 40 + r["valley"] - r["sat_pct"] * 8 + min(c, 220) * 0.15

            if rank(rep) > rank(best):
                take = True
        if take:
            best, best_frame, best_gray = rep, frame, gray

    if best_frame is not None:
        save_image(OUT / "light_best_preview.jpg", preview(best_frame))
        save_image(OUT / "light_best_full.jpg", best_frame)
        if best["boxes"]:
            save_image(
                OUT / "light_best_zoom.png",
                draw_profile_panel(best_gray, best["boxes"][0], best["tag"]),
            )
        print(
            f"\nBEST {best['tag']} copper={best['copper']:.0f} carbon={best['carbon']:.0f} "
            f"valley={best['valley']:.0f} sat={best['sat_pct']:.2f}%"
        )
    camw.disconnect()
    print("done")


if __name__ == "__main__":
    main()
