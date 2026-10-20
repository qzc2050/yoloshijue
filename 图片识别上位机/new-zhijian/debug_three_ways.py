# -*- coding: utf-8 -*-
"""Live debug: camera ISP, lighting diagnosis, software processing.

Needs exclusive camera access (close MVS first).
"""

from __future__ import annotations

import sys
from ctypes import byref, memset, sizeof
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")

from CameraParams_header import (  # noqa: E402
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE,
)
from hikrobot_camera import HikrobotCamera  # noqa: E402

CAMERA_IP = "192.168.0.100"
OUT = Path(__file__).resolve().parent / "captured" / "three_ways"


def save_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if suffix in {".jpg", ".jpeg"} else []
    ok, buf = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"encode fail {path}")
    path.write_bytes(buf.tobytes())


def gf(cam, name: str):
    v = MVCC_FLOATVALUE()
    memset(byref(v), 0, sizeof(v))
    r = cam.MV_CC_GetFloatValue(name, v)
    if r != 0:
        return None
    return {"cur": float(v.fCurValue), "min": float(v.fMin), "max": float(v.fMax)}


def gi(cam, name: str):
    v = MVCC_INTVALUE()
    memset(byref(v), 0, sizeof(v))
    r = cam.MV_CC_GetIntValue(name, v)
    if r != 0:
        return None
    return {"cur": int(v.nCurValue), "min": int(v.nMin), "max": int(v.nMax)}


def ge(cam, name: str):
    v = MVCC_ENUMVALUE()
    memset(byref(v), 0, sizeof(v))
    r = cam.MV_CC_GetEnumValue(name, v)
    if r != 0:
        return None
    return {"cur": int(v.nCurValue)}


def dump_params(cam) -> None:
    names = [
        "ExposureTime",
        "Gain",
        "Gamma",
        "BlackLevel",
        "Brightness",
        "Sharpness",
        "Contrast",
        "DigitalShift",
        "AcquisitionFrameRate",
        "Width",
        "Height",
        "OffsetX",
        "OffsetY",
        "GevSCPSPacketSize",
        "BinningHorizontal",
        "BinningVertical",
        "PixelFormat",
        "GainAuto",
        "ExposureAuto",
        "BalanceWhiteAuto",
        "GammaEnable",
        "SharpnessEnable",
    ]
    print("\n=== 当前相机节点 ===")
    for name in names:
        val = gf(cam, name) or gi(cam, name) or ge(cam, name)
        if val is None:
            continue
        extra = ""
        if "min" in val:
            extra = f"  [{val['min']}, {val['max']}]"
        print(f"  {name:22s}  {val['cur']}{extra}")


def set_float(cam, name: str, value: float) -> bool:
    r = cam.MV_CC_SetFloatValue(name, float(value))
    if r != 0:
        print(f"  set {name}={value} fail 0x{r:x}")
        return False
    return True


def set_enum_str(cam, name: str, value: str) -> bool:
    r = cam.MV_CC_SetEnumValueByString(name, value)
    if r != 0:
        print(f"  set {name}={value} fail 0x{r:x}")
        return False
    return True


def set_bool(cam, name: str, value: bool) -> bool:
    r = cam.MV_CC_SetBoolValue(name, bool(value))
    if r != 0:
        print(f"  set {name}={value} fail 0x{r:x}")
        return False
    return True


def stop_start(camw: HikrobotCamera) -> None:
    cam = camw.cam
    cam.MV_CC_StopGrabbing()
    r = cam.MV_CC_StartGrabbing()
    if r != 0:
        raise RuntimeError(f"StartGrabbing 0x{r:x}")
    camw._grabbing = True


def grab_n(camw: HikrobotCamera, n: int = 3, timeout_ms: int = 15000) -> np.ndarray:
    last = None
    last_err = None
    for i in range(max(n, 1) + 4):
        try:
            last = camw.grab_bgr(timeout_ms=timeout_ms)
            if i + 1 >= n:
                return last
        except RuntimeError as exc:
            last_err = exc
    if last is not None:
        return last
    raise RuntimeError(f"grab failed after retries: {last_err}")


def to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def find_ring_crops(gray: np.ndarray, k: int = 4) -> list[tuple[int, int, int, int]]:
    """Bright elongated pads / dual rings."""
    h, w = gray.shape
    thr = int(np.clip(np.percentile(gray, 88), 80, 230))
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    cands = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        x, y = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
        ww, hh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 800 or max(ww, hh) < 40:
            continue
        aspect = max(ww, hh) / max(1.0, min(ww, hh))
        if aspect < 1.6:
            continue
        pad = 18
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + ww + pad), min(h, y + hh + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        sat = float((crop >= 250).mean())
        contrast = float(np.percentile(crop, 95) - np.percentile(crop, 15))
        cands.append((contrast * 0.7 + area / 5000.0 - sat * 20, x0, y0, x1 - x0, y1 - y0, sat))
    cands.sort(reverse=True)
    boxes = [(x, y, ww, hh) for _, x, y, ww, hh, _ in cands[:k]]
    if boxes:
        return boxes
    # fallback: center and four mid-sides
    cy, cx = h // 2, w // 2
    return [
        (max(0, cx - 160), max(0, cy - 90), 320, 180),
        (max(0, w // 4 - 160), max(0, cy - 90), 320, 180),
        (max(0, 3 * w // 4 - 160), max(0, cy - 90), 320, 180),
    ]


def dual_peak_score(profile: np.ndarray) -> dict:
    """Score a 1D brightness profile for two copper peaks + carbon valley."""
    if profile.size < 12:
        return {"ok": False, "valley": 0.0, "peaks": 0, "sep_px": 0.0}
    k = max(3, profile.size // 40)
    kernel = np.ones(k) / k
    sm = np.convolve(profile.astype(np.float32), kernel, mode="same")
    d1 = np.diff(sm)
    peaks = []
    for i in range(2, len(d1) - 2):
        if d1[i - 1] > 0 and d1[i] <= 0 and sm[i] > np.percentile(sm, 60):
            peaks.append(i)
    # keep strongest well-separated peaks
    merged = []
    for p in peaks:
        if not merged or p - merged[-1] >= 4:
            merged.append(p)
        elif sm[p] > sm[merged[-1]]:
            merged[-1] = p
    if len(merged) < 2:
        return {"ok": False, "valley": 0.0, "peaks": len(merged), "sep_px": 0.0}
    best = None
    for a, b in zip(merged, merged[1:]):
        if b - a < 4 or b - a > 80:
            continue
        valley_i = int(a + np.argmin(sm[a : b + 1]))
        peak = 0.5 * (sm[a] + sm[b])
        valley = float(sm[valley_i])
        depth = float(peak - valley)
        if best is None or depth > best[0]:
            best = (depth, float(b - a), valley, peak)
    if best is None:
        return {"ok": False, "valley": 0.0, "peaks": len(merged), "sep_px": 0.0}
    depth, sep, valley, peak = best
    return {
        "ok": depth >= 12 and sep >= 4,
        "valley": depth,
        "peaks": len(merged),
        "sep_px": sep,
        "peak": peak,
        "valley_level": valley,
    }


def sample_profiles(gray: np.ndarray, box: tuple[int, int, int, int], n: int = 9) -> list[np.ndarray]:
    x, y, w, h = box
    crop = gray[y : y + h, x : x + w]
    if crop.size == 0:
        return []
    profiles = []
    if h >= w:
        for i in np.linspace(0.25, 0.75, n):
            rr = int(h * i)
            profiles.append(crop[rr, :].astype(np.float32))
    else:
        for i in np.linspace(0.25, 0.75, n):
            cc = int(w * i)
            profiles.append(crop[:, cc].astype(np.float32))
    return profiles


def analyze(bgr: np.ndarray, tag: str) -> dict:
    gray = to_gray(bgr)
    boxes = find_ring_crops(gray)
    sat_all = float((gray >= 250).mean() * 100)
    sat_hi = float((gray >= 245).mean() * 100)
    mean = float(gray.mean())
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    chroma = 0.0
    if bgr.ndim == 3:
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        chroma = float(np.std(yuv[:, :, 1]) + np.std(yuv[:, :, 2]))

    valley_scores = []
    copper_vals = []
    carbon_vals = []
    for box in boxes:
        x, y, w, h = box
        crop = gray[y : y + h, x : x + w]
        copper_vals.append(float(np.percentile(crop, 92)))
        carbon_vals.append(float(np.percentile(crop, 20)))
        for prof in sample_profiles(gray, box):
            sc = dual_peak_score(prof)
            if sc["valley"] > 0:
                valley_scores.append(sc)

    copper = float(np.median(copper_vals)) if copper_vals else p95
    carbon = float(np.median(carbon_vals)) if carbon_vals else p05
    good = [s for s in valley_scores if s["ok"]]
    best_v = max((s["valley"] for s in valley_scores), default=0.0)
    sep = float(np.median([s["sep_px"] for s in good])) if good else 0.0
    score = best_v - max(0.0, copper - 235) * 0.4 - sat_hi * 2.0
    report = {
        "tag": tag,
        "mean": mean,
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "sat_pct": sat_all,
        "copper": copper,
        "carbon": carbon,
        "contrast": copper - carbon,
        "valley": best_v,
        "sep_px": sep,
        "dual_ok": len(good),
        "lap": lap,
        "chroma": chroma,
        "score": score,
        "boxes": boxes,
    }
    print(
        f"[{tag:22s}] mean={mean:5.1f} copper={copper:5.1f} carbon={carbon:5.1f} "
        f"sat={sat_all:4.2f}% valley={best_v:5.1f} sep={sep:4.1f}px "
        f"dual={len(good):2d} lap={lap:.0f} chroma={chroma:.1f} score={score:.1f}"
    )
    return report


def draw_profile_panel(gray: np.ndarray, box: tuple[int, int, int, int], title: str) -> np.ndarray:
    x, y, w, h = box
    crop = gray[y : y + h, x : x + w]
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    zoom = cv2.resize(vis, (vis.shape[1] * 4, vis.shape[0] * 4), interpolation=cv2.INTER_NEAREST)
    profs = sample_profiles(gray, box, n=5)
    panel_h, panel_w = 160, zoom.shape[1]
    panel = np.full((panel_h, panel_w, 3), 20, np.uint8)
    if profs:
        p = profs[len(profs) // 2]
        xs = np.linspace(0, panel_w - 1, len(p)).astype(int)
        ys = (panel_h - 8 - p / 255.0 * (panel_h - 16)).astype(int)
        for i in range(1, len(xs)):
            cv2.line(panel, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (0, 220, 255), 1, cv2.LINE_AA)
        sc = dual_peak_score(p)
        cv2.putText(
            panel,
            f"valley {sc['valley']:.0f}  sep {sc['sep_px']:.1f}px  peaks {sc['peaks']}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(zoom, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([zoom, panel])


def software_variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    out = {"raw": gray}
    lut = np.array([min(255, int((i / 255.0) ** 0.6 * 255)) for i in range(256)], np.uint8)
    out["gamma0.6"] = cv2.LUT(gray, lut)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out["clahe"] = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    out["unsharp"] = np.clip(cv2.addWeighted(gray, 1.5, blur, -0.5, 0), 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    out["tophat"] = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    # copper vs carbon: stretch mid
    lo, hi = np.percentile(gray, [8, 96])
    out["stretch"] = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo), 0, 255).astype(np.uint8)
    combo = clahe.apply(out["stretch"])
    blur2 = cv2.GaussianBlur(combo, (0, 0), 0.9)
    out["clahe+unsharp"] = np.clip(cv2.addWeighted(combo, 1.35, blur2, -0.35, 0), 0, 255).astype(np.uint8)
    return out


def lighting_advice(rep: dict) -> list[str]:
    tips = []
    if rep["copper"] >= 245 or rep["sat_pct"] > 0.4:
        tips.append(
            "铜线已经顶白（过曝）。请把灯调暗、抬高，或把灯从正上方改成两侧斜打；"
            "不要再加大曝光。目标：铜线峰值约 180–230，不要出现大片 255。"
        )
    elif rep["copper"] < 140:
        tips.append("铜线偏暗。灯不够或角度太偏。两侧各加一盏斜 30–45° 的白光，灯距板面约 20–40 cm。")
    else:
        tips.append("铜线亮度大致可用，先不要加灯，优先保中间炭黑的黑缝。")

    if rep["carbon"] > 90:
        tips.append(
            "炭黑不够黑，多半是灯太散、有环境光或铜的反光洒进缝里。"
            "关掉房间大灯，用两盏条形灯左右对称斜打，避开正顶同轴。"
        )
    if rep["valley"] < 18:
        tips.append(
            "剖面上看不到稳定的双峰+中间谷：两圈铜在光学上已经粘了，或反光把缝填亮。"
            "试：①两侧低角度侧光（暗场）让铜边亮、缝更黑；②镜头前加偏振片，灯前加正交偏振，压掉金面高光。"
        )
    elif rep["sep_px"] and rep["sep_px"] < 5:
        tips.append(
            f"两圈间距大约只有 {rep['sep_px']:.1f} 像素。灯光再好也加不出像素，只能降相机或换长焦。"
        )
    else:
        tips.append("剖面能分出两圈。灯光/曝光维持这个水平即可，后面用软件测宽，不要靠放大图判断。")

    tips.append(
        "推荐打光（针对亮铜+炭黑）：左右各一条 LED 条形灯，与板面成 30–45°，"
        "高度相近、亮度对称；相机在正上方；可选偏振片。不要只用一盏台灯从一侧硬打。"
    )
    return tips


def apply_camera_base(cam) -> None:
    set_enum_str(cam, "ExposureAuto", "Off")
    set_enum_str(cam, "GainAuto", "Off")
    set_float(cam, "Gain", 0.0)
    set_bool(cam, "GammaEnable", False)
    set_bool(cam, "SharpnessEnable", False)
    # 289ms + max gain was the live stream state; that blows copper and looks mosaic.
    set_float(cam, "ExposureTime", 12000.0)
    set_bool(cam, "AcquisitionFrameRateEnable", False)
    gi_bin = gi(cam, "BinningHorizontal")
    if gi_bin and gi_bin["cur"] != 1:
        cam.MV_CC_SetIntValue("BinningHorizontal", 1)
        cam.MV_CC_SetIntValue("BinningVertical", 1)


def restart_grab(camw: HikrobotCamera) -> None:
    import time

    cam = camw.cam
    cam.MV_CC_StopGrabbing()
    packet = cam.MV_CC_GetOptimalPacketSize()
    if int(packet) > 0:
        cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet))
        print(f"  packet size -> {int(packet)}")
    r = cam.MV_CC_StartGrabbing()
    if r != 0:
        raise RuntimeError(f"StartGrabbing 0x{r:x}")
    camw._grabbing = True
    time.sleep(0.4)


def exposure_list(cur_us: float) -> list[float]:
    # Keep the sweep in a range that still leaves copper unsaturated.
    cur_us = float(np.clip(cur_us, 1500.0, 40000.0))
    vals = [cur_us]
    for v in (2000, 4000, 8000, 12000, 18000, 25000, 40000, 60000):
        vals.append(float(v))
    return sorted(set(round(v, 1) for v in vals))


def preview(img: np.ndarray, max_side: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    camw = HikrobotCamera()
    print("枚举：", camw.enumerate())
    info = camw.connect(CAMERA_IP)
    print("已连接：", info)
    cam = camw.cam
    try:
        dump_params(cam)
        print(
            "\n发现现场参数异常：BayerGB8 + 曝光≈289ms（上次还曾满增益）。"
            "先不断流改参数并采一张，确认链路。"
        )
        print("先采一张「改参数前」现场图…")
        dirty = grab_n(camw, 2, timeout_ms=8000)
        save_image(OUT / "00_as_found.jpg", dirty)
        save_image(OUT / "00_as_found_preview.jpg", preview(dirty))
        analyze(dirty, "as_found")

        apply_camera_base(cam)
        dump_params(cam)
        exp = gf(cam, "ExposureTime")
        cur_exp = exp["cur"] if exp else 12000.0
        print(f"\n已强制（不断流）：Gain=0，曝光={cur_exp:.1f} us")

        formats = [("BayerGB8", None), ("Mono8", "Mono8")]
        all_reports = []
        best = None
        best_frame = None
        best_gray = None

        for fmt_name, fmt_val in formats:
            if fmt_val:
                print(f"\n切换 PixelFormat -> {fmt_val}")
                try:
                    cam.MV_CC_StopGrabbing()
                    if not set_enum_str(cam, "PixelFormat", fmt_val):
                        print("  Mono8 切不过，跳过（继续用 Bayer）")
                        restart_grab(camw)
                        continue
                    restart_grab(camw)
                except RuntimeError as exc:
                    print(f"  切格式失败：{exc}，保持 Bayer")
                    try:
                        restart_grab(camw)
                    except RuntimeError:
                        pass
                    continue

            if fmt_name == "BayerGB8":
                frame = grab_n(camw, 4)
                save_image(OUT / "01_gain0_exp12ms.jpg", frame)
                save_image(OUT / "01_gain0_preview.jpg", preview(frame))
                rep = analyze(frame, "Bayer_gain0_12ms")
                all_reports.append(rep)
                best, best_frame, best_gray = rep, frame, to_gray(frame)

            times = exposure_list(cur_exp)
            print(f"\n--- 曝光扫描 ({fmt_name}) ---")
            for t in times:
                set_float(cam, "ExposureTime", t)
                frame = grab_n(camw, 3)
                tag = f"{fmt_name}_exp{int(t)}"
                save_image(OUT / f"{tag}.jpg", frame)
                rep = analyze(frame, tag)
                all_reports.append(rep)
                if best is None or rep["score"] > best["score"]:
                    best, best_frame, best_gray = rep, frame, to_gray(frame)

            if fmt_name == "Mono8":
                if best and "exp" in best["tag"]:
                    try:
                        t = float(best["tag"].split("exp")[-1])
                        set_float(cam, "ExposureTime", t)
                    except ValueError:
                        pass
                if set_bool(cam, "GammaEnable", True):
                    set_float(cam, "Gamma", 0.6)
                    frame = grab_n(camw, 3)
                    tag = "Mono8_gamma0.6"
                    save_image(OUT / f"{tag}.jpg", frame)
                    rep = analyze(frame, tag)
                    all_reports.append(rep)
                    if best is None or rep["score"] > best["score"]:
                        best, best_frame, best_gray = rep, frame, to_gray(frame)
                    set_bool(cam, "GammaEnable", False)

        if best_frame is None:
            raise RuntimeError("没有采到有效帧")

        save_image(OUT / "best_full.jpg", best_frame)
        save_image(OUT / "best_preview.jpg", preview(best_frame))
        print(f"\n相机侧最佳：{best['tag']}  score={best['score']:.1f}")

        # --- 3) software processing on best gray ---
        print("\n=== 软件处理（基于最佳相机帧）===")
        variants = software_variants(best_gray)
        sw_best = None
        panels = []
        box = best["boxes"][0] if best["boxes"] else (best_gray.shape[1] // 2 - 120, best_gray.shape[0] // 2 - 80, 240, 160)
        for name, img in variants.items():
            fake = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            rep = analyze(fake, f"sw_{name}")
            save_image(OUT / f"sw_{name}.jpg", preview(fake))
            panels.append(draw_profile_panel(img, box, name))
            if sw_best is None or rep["score"] > sw_best[0]["score"]:
                sw_best = (rep, img)

        # mosaic vs lanczos zoom comparison
        x, y, w, h = box
        crop = best_gray[y : y + h, x : x + w]
        near = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
        lanc = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6), interpolation=cv2.INTER_LANCZOS4)
        save_image(OUT / "zoom_nearest.png", near)
        save_image(OUT / "zoom_lanczos.png", lanc)
        combo_zoom = np.hstack(
            [
                cv2.cvtColor(near, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(lanc, cv2.COLOR_GRAY2BGR),
            ]
        )
        cv2.putText(combo_zoom, "NEAREST = mosaic you see", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(
            combo_zoom,
            "LANCZOS = prettier, not more resolution",
            (near.shape[1] + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        save_image(OUT / "zoom_compare.jpg", combo_zoom)

        # stack first few panels
        row = panels[0]
        for p in panels[1:]:
            if p.shape[0] != row.shape[0]:
                p = cv2.resize(p, (int(p.shape[1] * row.shape[0] / p.shape[0]), row.shape[0]))
            row = np.hstack([row, p])
        save_image(OUT / "software_profiles.jpg", preview(row, 2200))
        save_image(OUT / "best_ring_zoom.png", draw_profile_panel(best_gray, box, best["tag"]))

        print("\n=== 1 相机参数结论 ===")
        ranked = sorted(all_reports, key=lambda r: r["score"], reverse=True)[:8]
        for r in ranked:
            print(
                f"  {r['tag']:22s} score={r['score']:6.1f} copper={r['copper']:.0f} "
                f"valley={r['valley']:.0f} sat={r['sat_pct']:.2f}%"
            )

        print("\n=== 2 外部灯光（根据这帧诊断，请你动手改灯）===")
        for i, tip in enumerate(lighting_advice(best), 1):
            print(f"  {i}. {tip}")

        print("\n=== 3 软件算法 ===")
        if sw_best:
            print(f"  在最佳相机帧上，软件最好的是 {sw_best[0]['tag']}  valley={sw_best[0]['valley']:.1f}")
            print("  软件只能加深已有的黑缝、降噪、让边更好测；不能把马赛克变成真实细节。")
        print(f"\n对比图目录：{OUT}")
        print("看这些文件：best_preview.jpg, best_ring_zoom.png, zoom_compare.jpg, software_profiles.jpg")
    finally:
        camw.disconnect()
        print("已断开相机")


if __name__ == "__main__":
    main()
