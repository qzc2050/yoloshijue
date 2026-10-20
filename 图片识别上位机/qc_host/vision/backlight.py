# -*- coding: utf-8 -*-
"""Backlight inspection: glowing FR4 slot = ring, bleed = local narrow, void = extra glow inside."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROCESS_MAX_SIDE = 1800
BLEED_WIDTH_RATIO = 0.58
BLEED_MIN_RUN = 3
VOID_AREA_FRAC = 0.004
VOID_MIN_PX = 40


@dataclass
class Defect:
    kind: str
    x: int
    y: int
    score: float
    note: str = ""


@dataclass
class ChamberResult:
    index: int
    ok: bool
    median_width: float
    outer: np.ndarray
    inner: Optional[np.ndarray]
    bleeds: List[Defect] = field(default_factory=list)
    voids: List[Defect] = field(default_factory=list)


@dataclass
class BacklightResult:
    annotated_bgr: np.ndarray
    scale: float
    chambers: List[ChamberResult]
    summary: str


def inspect_backlight(bgr: np.ndarray) -> BacklightResult:
    """Find glowing stadium slots and flag bleed / missing-carbon leaks."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    work, scale = _downscale(bgr, PROCESS_MAX_SIDE)
    vis = work.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    glow = _glow(work)
    mask = _ring_mask(glow)
    chambers = _find_chambers(mask, glow)

    n_bleed = n_void = n_ok = 0
    labels: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []
    for ch in chambers:
        color = (80, 210, 80) if ch.ok else (40, 40, 220)
        cv2.drawContours(vis, [ch.outer], -1, color, 2, cv2.LINE_AA)
        if ch.inner is not None:
            cv2.drawContours(vis, [ch.inner], -1, color, 1, cv2.LINE_AA)
        for d in ch.bleeds:
            cv2.circle(vis, (d.x, d.y), 7, (0, 0, 255), 2, cv2.LINE_AA)
            labels.append(("渗墨", (d.x + 8, d.y - 18), (0, 0, 255)))
        for d in ch.voids:
            cv2.circle(vis, (d.x, d.y), 8, (0, 220, 255), 2, cv2.LINE_AA)
            labels.append(("缺墨", (d.x + 8, d.y - 18), (0, 220, 255)))
        x, y, _w, _h = cv2.boundingRect(ch.outer)
        tag = "合格" if ch.ok else "NG"
        labels.append((f"#{ch.index} {tag}", (x, max(4, y - 22)), color))
        if ch.bleeds:
            n_bleed += 1
        if ch.voids:
            n_void += 1
        if ch.ok:
            n_ok += 1

    if not chambers:
        summary = "没有找到亮环。确认背面透光、对焦到板面。"
        labels.append((summary, (16, 8), (0, 200, 255)))
    else:
        summary = f"电离室 {len(chambers)}  合格 {n_ok}  渗墨 {n_bleed}  缺墨 {n_void}"
        labels.append((summary, (16, 8), (230, 230, 230)))
    vis = _draw_cn(vis, labels)

    return BacklightResult(annotated_bgr=vis, scale=scale, chambers=chambers, summary=summary)


def _cn_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_cn(
    bgr: np.ndarray,
    labels: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
) -> np.ndarray:
    if not labels:
        return bgr
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    font = _cn_font(18)
    small = _cn_font(16)
    for i, (text, xy, color) in enumerate(labels):
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        use = font if i == len(labels) - 1 or text.startswith("#") else small
        if "电离室" in text or "没有找到" in text:
            use = font
        draw.text(xy, text, font=use, fill=rgb)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _downscale(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 0.999:
        return image, 1.0
    out = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def _glow(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    _b, g, r = cv2.split(bgr)
    return cv2.max(g, r)


def _ring_mask(glow: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(glow, (5, 5), 0)
    ksize = max(15, (min(glow.shape) // 22) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    p99 = float(np.percentile(blur, 99.5))
    floor = max(16.0, p99 * 0.22)
    hat_thr = max(6.0, float(np.percentile(tophat, 98.5)) * 0.20)
    mask = ((tophat >= hat_thr) | (blur >= floor)) & (blur >= max(10.0, floor * 0.45))
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, close_k, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, close_k, iterations=1)
    return mask_u8


def _find_chambers(mask: np.ndarray, glow: np.ndarray) -> List[ChamberResult]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hier = hierarchy[0]
    h, w = mask.shape
    img_area = float(h * w)
    min_band = max(180.0, 0.00008 * img_area)
    chambers: List[ChamberResult] = []
    index = 1
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:
            continue
        area = cv2.contourArea(cnt)
        if area < min_band or area > 0.96 * img_area:
            continue
        if len(cnt) < 40:
            continue
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if min(rw, rh) < 8:
            continue
        aspect = max(rw, rh) / max(1.0, float(min(rw, rh)))
        if aspect < 1.35:
            continue
        child = hier[i][2]
        inner = None
        inner_area = 0.0
        if child != -1:
            best_child = child
            best_area = cv2.contourArea(contours[child])
            nxt = hier[child][0]
            while nxt != -1:
                a = cv2.contourArea(contours[nxt])
                if a > best_area:
                    best_area = a
                    best_child = nxt
                nxt = hier[nxt][0]
            if best_area > area * 0.12:
                inner = contours[best_child]
                inner_area = best_area

        band = area - inner_area
        if inner is not None and band < min_band:
            continue

        if inner is None:
            if area < 0.05 * img_area:
                continue
            ch = ChamberResult(
                index=index,
                ok=False,
                median_width=0.0,
                outer=cnt,
                inner=None,
                bleeds=[Defect("bleed", int(rx + rw / 2), int(ry + rh / 2), 1.0, "亮环不闭合")],
            )
            chambers.append(ch)
            index += 1
            continue

        widths, samples = _sample_widths(mask, cnt)
        median_w = float(np.median(widths)) if len(widths) else 0.0
        bleeds = _bleed_defects(widths, samples, median_w)
        bleeds.extend(_glow_dip_defects(glow, samples))
        bleeds = _dedupe_defects(bleeds)
        voids = _void_defects(mask, glow, inner, inner_area, median_w)
        ok = not bleeds and not voids
        chambers.append(
            ChamberResult(
                index=index,
                ok=ok,
                median_width=median_w,
                outer=cnt,
                inner=inner,
                bleeds=bleeds,
                voids=voids,
            )
        )
        index += 1
    chambers.sort(key=lambda c: cv2.boundingRect(c.outer)[0])
    for i, ch in enumerate(chambers, start=1):
        ch.index = i
    return chambers


def _sample_widths(mask: np.ndarray, outer: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts = outer.reshape(-1, 2).astype(np.float32)
    if len(pts) < 20:
        return np.zeros((0,), np.float32), np.zeros((0, 2), np.float32)
    step = max(1, len(pts) // 180)
    pts = pts[::step]
    centroid = pts.mean(axis=0)
    h, w = mask.shape
    max_steps = max(12, int(min(h, w) * 0.08))
    widths = []
    samples = []
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        prev_p = pts[(i - 1) % n]
        next_p = pts[(i + 1) % n]
        tx, ty = next_p - prev_p
        nx, ny = -ty, tx
        inward = np.array([centroid[0] - x, centroid[1] - y], dtype=np.float32)
        if nx * inward[0] + ny * inward[1] < 0:
            nx, ny = -nx, -ny
        norm = float(np.hypot(nx, ny)) + 1e-6
        dx, dy = nx / norm, ny / norm
        width = 0
        started = False
        for s in range(max_steps):
            xx = int(round(x + dx * s))
            yy = int(round(y + dy * s))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                break
            on = mask[yy, xx] > 0
            if on:
                started = True
                width += 1
            elif started:
                break
        widths.append(float(width))
        samples.append((float(x + dx * max(width, 1) * 0.5), float(y + dy * max(width, 1) * 0.5)))
    return np.asarray(widths, np.float32), np.asarray(samples, np.float32)


def _bleed_defects(widths: np.ndarray, samples: np.ndarray, median_w: float) -> List[Defect]:
    if widths.size < 12 or median_w < 2.0:
        return []
    limit = max(1.5, median_w * BLEED_WIDTH_RATIO)
    narrow = widths < limit
    defects: List[Defect] = []
    n = len(narrow)
    i = 0
    while i < n:
        if not narrow[i]:
            i += 1
            continue
        j = i
        while j < n and narrow[j]:
            j += 1
        run = j - i
        wrap = False
        if i == 0:
            k = n - 1
            while k > j and narrow[k]:
                k -= 1
            extra = n - 1 - k
            if extra > 0:
                run += extra
                wrap = True
        if run >= BLEED_MIN_RUN:
            mid = (i + j) // 2 if not wrap else 0
            mid = min(mid, len(samples) - 1)
            x, y = samples[mid]
            score = float(1.0 - widths[i:j].mean() / max(median_w, 1e-3))
            defects.append(Defect("bleed", int(round(x)), int(round(y)), score, "亮环局部变窄"))
        i = j
    return defects


def _circular_median(values: np.ndarray, radius: int) -> np.ndarray:
    if values.size == 0:
        return values
    k = max(1, int(radius))
    pad = np.concatenate([values[-k:], values, values[:k]])
    out = np.empty_like(values, dtype=np.float32)
    win = 2 * k + 1
    for i in range(len(values)):
        out[i] = float(np.median(pad[i : i + win]))
    return out


def _glow_dip_defects(glow: np.ndarray, samples: np.ndarray) -> List[Defect]:
    if samples.shape[0] < 16:
        return []
    h, w = glow.shape[:2]
    xs = np.clip(np.round(samples[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(samples[:, 1]).astype(int), 0, h - 1)
    vals = glow[ys, xs].astype(np.float32)
    radius = max(8, len(vals) // 10)
    local = _circular_median(vals, radius)
    limit = np.maximum(local * 0.42, 8.0)
    dip = vals < limit
    defects: List[Defect] = []
    n = len(dip)
    i = 0
    while i < n:
        if not dip[i]:
            i += 1
            continue
        j = i
        while j < n and dip[j]:
            j += 1
        run = j - i
        if i == 0:
            k = n - 1
            while k > j and dip[k]:
                k -= 1
            run += n - 1 - k
        if run >= 3:
            mid = min((i + j) // 2, len(samples) - 1)
            x, y = samples[mid]
            defects.append(Defect("bleed", int(round(x)), int(round(y)), 1.0, "亮环局部变暗"))
        i = j
    return defects


def _dedupe_defects(defects: List[Defect], min_dist: float = 18.0) -> List[Defect]:
    kept: List[Defect] = []
    for d in defects:
        if any((d.x - k.x) ** 2 + (d.y - k.y) ** 2 < min_dist * min_dist for k in kept):
            continue
        kept.append(d)
    return kept


def _void_defects(
    mask: np.ndarray,
    glow: np.ndarray,
    inner: np.ndarray,
    inner_area: float,
    median_w: float,
) -> List[Defect]:
    hole = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(hole, [inner], -1, 255, -1)
    erode_r = max(2, int(round(median_w * 0.45)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r * 2 + 1, erode_r * 2 + 1))
    hole = cv2.erode(hole, kernel, iterations=1)
    if cv2.countNonZero(hole) < 20:
        return []
    ring_level = float(np.median(glow[mask > 0])) if np.any(mask > 0) else 40.0
    leak = ((glow >= max(18.0, ring_level * 0.45)) & (hole > 0)).astype(np.uint8) * 255
    leak = cv2.morphologyEx(leak, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n_lbl, labels, stats, centroids = cv2.connectedComponentsWithStats(leak, connectivity=8)
    min_area = max(VOID_MIN_PX, inner_area * VOID_AREA_FRAC)
    defects: List[Defect] = []
    M = cv2.moments(inner)
    cx = (M["m10"] / M["m00"]) if M["m00"] > 1e-6 else float(centroids[0][0] if n_lbl > 1 else 0)
    cy = (M["m01"] / M["m00"]) if M["m00"] > 1e-6 else float(centroids[0][1] if n_lbl > 1 else 0)
    for lbl in range(1, n_lbl):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(round(centroids[lbl][0]))
        y = int(round(centroids[lbl][1]))
        ww = int(stats[lbl, cv2.CC_STAT_WIDTH])
        hh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        dist = float(np.hypot(x - cx, y - cy))
        hole_r = np.sqrt(inner_area / np.pi)
        if area < max(min_area * 1.8, 90) and dist < hole_r * 0.22 and max(ww, hh) / max(1, min(ww, hh)) > 1.6:
            continue
        defects.append(Defect("void", x, y, float(area / max(inner_area, 1.0)), "内芯透光"))
    return defects
