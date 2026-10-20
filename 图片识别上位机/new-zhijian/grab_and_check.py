# -*- coding: utf-8 -*-
"""First-priority check: can we connect, grab, and optically resolve dual-ring width.

Coverage (how much board one shot sees) is later — multiple cameras.
This script answers: at what working distance can ONE camera separate inner/outer
copper traces well enough to measure line width.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hikrobot_camera import HikrobotCamera  # noqa: E402
from ring_detector import detect_rings  # noqa: E402

CAMERA_IP = "192.168.0.100"
OUT_DIR = Path(__file__).resolve().parent / "captured"
def save_image(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(suffix, image)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True

# MV-CS200-10GC / Sony IMX183
PIXEL_SIZE_UM = 2.4
SENSOR_W_MM = 5472 * PIXEL_SIZE_UM / 1000.0  # 13.1328
SENSOR_H_MM = 3648 * PIXEL_SIZE_UM / 1000.0  # 8.7552
PIXELS_W = 5472
PIXELS_H = 3648

# Width metrology: FWHM needs several samples across the bright band.
MIN_PX_PER_TRACE = 8
# Inner/outer must not merge: gap at least ~3 px.
MIN_PX_GAP = 3


def print_height_table(trace_mm: float = 0.20, gap_mm: float = 0.20) -> None:
    need_um = min(trace_mm * 1000.0 / MIN_PX_PER_TRACE, gap_mm * 1000.0 / MIN_PX_GAP)
    print()
    print("=" * 72)
    print("线宽能否量出来，只取决于：每个像素对应多少微米")
    print(f"铜线按 {trace_mm:.2f} mm、两环间隙按 {gap_mm:.2f} mm 估算")
    print(f"量线宽至少要 {MIN_PX_PER_TRACE} 像素/线  →  每像素 ≤ {trace_mm*1000/MIN_PX_PER_TRACE:.1f} μm")
    print(f"分开内外环至少要 {MIN_PX_GAP} 像素/缝  →  每像素 ≤ {gap_mm*1000/MIN_PX_GAP:.1f} μm")
    print(f"因此目标：≤ {need_um:.1f} μm/px")
    print("-" * 72)
    print(f"{'镜头焦距':>10}  {'物距WD':>10}  {'水平视野':>10}  {'垂直视野':>10}  {'μm/px':>8}")
    for f_mm in (8, 12, 16, 25, 35, 50):
        # um/px = PIXEL_SIZE * WD / f  →  WD = um/px * f / PIXEL_SIZE
        wd = need_um * f_mm / PIXEL_SIZE_UM
        fov_w = SENSOR_W_MM * wd / f_mm
        fov_h = SENSOR_H_MM * wd / f_mm
        print(f"{f_mm:8.0f} mm  {wd:8.0f} mm  {fov_w:8.0f} mm  {fov_h:8.0f} mm  {need_um:8.1f}")
    print("-" * 72)
    print("物距 WD ≈ 镜头到板面的距离（薄透镜近似，实际再减一点对焦余量）。")
    print("一台相机在这个高度只能看上表「视野」那么大；整板不够覆盖就加相机，不要抬高。")
    print("=" * 72)


def assess_frame(bgr: np.ndarray) -> None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print()
    print(f"画面尺寸 {bgr.shape[1]} x {bgr.shape[0]}  均值 {gray.mean():.1f}")
    output = detect_rings(bgr, source_dpi=1200, threshold_width_px=3, sr_scale=2, allow_small=True)
    n = len(output.results)
    dual = 0
    widths = []
    for item in output.results:
        inner, outer = item.inner_line_width_px, item.outer_line_width_px
        if inner >= 1.5 and outer >= 1.5:
            dual += 1
        if item.min_line_width_px >= 0.5:
            widths.append(item.min_line_width_px)
    print(f"找到跑道环 {n} 个，其中内外环都能分开量的 {dual} 个")
    if widths:
        arr = np.asarray(widths)
        print(
            f"最小线宽 px：中位 {np.median(arr):.2f}，"
            f"p10 {np.percentile(arr, 10):.2f}，最小 {arr.min():.2f}"
        )
        median_w = float(np.median(arr))
        if median_w < MIN_PX_PER_TRACE:
            print(
                f"结论：当前高度不够。线宽只有约 {median_w:.1f} px，"
                f"低于 {MIN_PX_PER_TRACE} px，量不准、内外环容易粘。"
            )
            print("请降低相机（或换长焦），不要靠软件放大。")
        else:
            print(f"结论：当前高度可以量线宽（约 {median_w:.1f} px/线）。广度不够再加相机。")
    else:
        print("结论：没有稳定测到双环线宽。先对焦、加光、降增益，再降低高度。")
    annotated = OUT_DIR / "grab_annotated.jpg"
    save_image(annotated, output.annotated_bgr)
    print(f"标注图：{annotated}")


def grab() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = HikrobotCamera()
    devices = cam.enumerate()
    print("枚举结果：", devices)
    if not devices:
        raise SystemExit("网口上看不到相机。检查 192.168.0.100 和网线。")
    info = cam.connect(CAMERA_IP)
    print("已连接：", info)
    try:
        frame = cam.grab_bgr(timeout_ms=5000)
    finally:
        cam.disconnect()
        print("已断开相机")
    raw = OUT_DIR / "grab_full.jpg"
    preview = OUT_DIR / "grab_preview.jpg"
    ok = save_image(raw, frame)
    h, w = frame.shape[:2]
    scale = min(1.0, 1600.0 / max(w, h))
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    save_image(preview, small)
    print(f"原图：{raw}  写入={'OK' if ok else 'FAIL'}  shape={frame.shape}")
    assess_frame(frame)


def main() -> None:
    print("相机：Hikrobot MV-CS200-10GC  5472x3648  像元 2.4μm")
    print_height_table(0.20, 0.20)
    print()
    print("开始连接并采图（请先在 MVS 里停止采集，否则会占用失败）…")
    try:
        grab()
    except Exception as exc:
        print()
        print("采图失败：", exc)
        print("先关掉 MVS 的采集/程序，再运行本脚本。")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
