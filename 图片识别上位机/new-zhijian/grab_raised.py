# -*- coding: utf-8 -*-
"""Grab one live frame after raising the camera and save comparison assets."""

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

OUT = Path(__file__).resolve().parent / "captured" / "raised"
PREV = {
    "low": Path(__file__).resolve().parent / "captured" / "aperture_8" / "full.jpg",
    "mid": Path(__file__).resolve().parent / "captured" / "width_now" / "full.jpg",
}


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_sharp(gray: np.ndarray) -> float:
    vals = []
    for x, y, w, h in find_ring_crops(gray, k=8)[:6]:
        c = gray[y : y + h, x : x + w]
        if c.size > 100:
            vals.append(sharpness(c))
    return float(np.median(vals)) if vals else 0.0


def pad_scale(gray: np.ndarray) -> tuple[float | None, float]:
    h, w = gray.shape
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
    um_px = 9200.0 / major
    return um_px, major


def profile_stats(gray: np.ndarray) -> dict:
    inners, outers, seps = [], [], []
    for box in find_ring_crops(gray, k=8)[:8]:
        local_in, local_out, local_sep = [], [], []
        for prof in sample_profiles(gray, box, n=13):
            peaks = peak_widths(prof)
            if len(peaks) < 2:
                continue
            w0, w1 = peaks[0]["fwhm"], peaks[1]["fwhm"]
            sep = abs(peaks[1]["center"] - peaks[0]["center"])
            if sep < 6 or sep > 50:
                continue
            local_in.append(min(w0, w1))
            local_out.append(max(w0, w1))
            local_sep.append(sep)
        if local_in:
            inners.append(float(np.median(local_in)))
            outers.append(float(np.median(local_out)))
            seps.append(float(np.median(local_sep)))
    if not inners:
        return {}
    return {
        "inner": float(np.median(inners)),
        "outer": float(np.median(outers)),
        "sep": float(np.median(seps)),
        "n": len(inners),
    }


def summarize(name: str, gray: np.ndarray) -> None:
    h, w = gray.shape
    um_px, major = pad_scale(gray)
    ps = profile_stats(gray)
    print(f"\n=== {name} {w}x{h} ===")
    print(
        f"mean={gray.mean():.1f} sat={(gray>=250).mean()*100:.2f}% "
        f"lap={sharpness(gray):.1f} crop_lap={crop_sharp(gray):.1f}"
    )
    if um_px:
        wd = um_px * 5.0  # 12mm lens, 2.4um pixel: WD_mm = um_px * f / pix
        print(
            f"pad_major={major:.0f}px  scale={um_px:.2f} um/px  "
            f"FOV~{w*um_px/1000:.0f}x{h*um_px/1000:.0f}mm  "
            f"WD~{wd:.0f}mm  0.2mm={200/um_px:.1f}px"
        )
    if ps:
        print(
            f"profile inner={ps['inner']:.1f}px outer={ps['outer']:.1f}px "
            f"sep={ps['sep']:.1f}px n={ps['n']}"
        )
        if um_px:
            print(
                f"profile mm inner={ps['inner']*um_px/1000:.3f} "
                f"outer={ps['outer']*um_px/1000:.3f}"
            )


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
    save_image(OUT / "full.jpg", frame)
    save_image(OUT / "preview.jpg", preview(frame, 1600))
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

    summarize("RAISED", gray)
    for key, path in PREV.items():
        if not path.exists():
            continue
        old = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if old is None:
            continue
        summarize(key.upper(), to_gray(old))
    print("saved", OUT)


if __name__ == "__main__":
    main()
