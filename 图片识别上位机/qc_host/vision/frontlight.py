# -*- coding: utf-8 -*-
"""Front-light inspection per qc_host/质检规则.md.

Pipeline:
  1. Locate each chamber by the ideal outer-copper stadium
     (smooth capsule). Visible gold is irregular where carbon covers copper;
     that is ignored.
  2. Draw the 0.3 mm FR4 isolation band (not the outer copper frame).
     Analyse only:
       渗墨 = carbon in the FR4 0.3 mm gap
       缺墨 = missing carbon inside the inner copper
     Carbon on inner or outer copper is not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROCESS_MAX_SIDE = 1800  # only for debug overview; locate/analyse use the original frame
REF_SIDE = 1800.0
VERDICTS = ("合格", "渗墨", "缺墨", "不能识别")
COLOR_OK = (80, 210, 80)
COLOR_BLEED = (40, 40, 220)
COLOR_VOID = (0, 210, 255)
COLOR_UNKNOWN = (0, 165, 255)


@dataclass
class Mark:
    kind: str
    x: int
    y: int
    note: str = ""


@dataclass
class ChamberResult:
    index: int
    verdict: str
    outer: np.ndarray
    slot_width: float = 0.0
    marks: List[Mark] = field(default_factory=list)
    gap_line: np.ndarray | None = None


@dataclass
class _Located:
    outer: np.ndarray
    hole: np.ndarray
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class FrontlightResult:
    annotated_bgr: np.ndarray
    chambers: List[ChamberResult]
    summary: str


def inspect_frontlight(bgr: np.ndarray) -> FrontlightResult:
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    work = bgr
    px = _px(work)
    located = _locate_chambers(work, px)

    chambers: List[ChamberResult] = []
    for i, loc in enumerate(located, start=1):
        chambers.append(_analyse_chamber(work, loc, i, px))

    counts = {k: 0 for k in VERDICTS}
    vis, vscale = _downscale(work, PROCESS_MAX_SIDE)
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    labels: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []
    for ch in chambers:
        counts[ch.verdict] = counts.get(ch.verdict, 0) + 1
        color = {
            "合格": COLOR_OK,
            "渗墨": COLOR_BLEED,
            "缺墨": COLOR_VOID,
            "不能识别": COLOR_UNKNOWN,
        }[ch.verdict]
        ring = ch.gap_line
        if ring is not None:
            cnt = np.round(ring.astype(np.float32) * vscale).astype(np.int32)
            cv2.drawContours(vis, [cnt], -1, color, 2, cv2.LINE_AA)
            x, y, _w, _h = cv2.boundingRect(cnt)
        else:
            x, y, _w, _h = cv2.boundingRect(np.round(ch.outer.astype(np.float32) * vscale).astype(np.int32))
        labels.append((f"#{ch.index} {ch.verdict}", (x, max(4, y - 22)), color))
        for m in ch.marks:
            mk = COLOR_BLEED if m.kind == "渗墨" else COLOR_VOID
            mx, my = int(round(m.x * vscale)), int(round(m.y * vscale))
            cv2.circle(vis, (mx, my), 7, mk, 2, cv2.LINE_AA)
            labels.append((m.kind, (mx + 8, my - 16), mk))

    if not chambers:
        summary = "不能识别：没有找到电离室。请先保证整颗跑道形腔室在画面里、铜环清楚。"
        labels.append((summary, (16, 8), COLOR_UNKNOWN))
    else:
        summary = (
            f"电离室 {len(chambers)}  合格 {counts['合格']}  "
            f"渗墨 {counts['渗墨']}  缺墨 {counts['缺墨']}  不能识别 {counts['不能识别']}"
        )
        labels.append((summary, (16, 8), (230, 230, 230)))
    vis = _draw_cn(vis, labels)
    return FrontlightResult(annotated_bgr=vis, chambers=chambers, summary=summary)


def _px(image: np.ndarray) -> float:
    return max(1.0, float(max(image.shape[:2])) / REF_SIDE)


def _odd(n: float) -> int:
    k = max(3, int(round(n)))
    return k if k % 2 else k + 1


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


def _fill(cnt: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if cnt is not None and len(cnt) >= 3:
        cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask


def _outer_stadium_mask(work: np.ndarray, px: float = 1.0) -> np.ndarray:
    """Copper rings by local contrast. A ring light is a smooth blob and must not set the threshold."""
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    den = cv2.medianBlur(gray, _odd(3 * min(px, 1.7)))
    local = cv2.GaussianBlur(den.astype(np.float32), (0, 0), 12.0 * px)
    contrast = den.astype(np.float32) - local
    # Global percentiles follow the lamp. Chambers at the edge of the ring then fall under the cut.
    ridge = (contrast > 6.0) & (den > 10)
    if work.ndim == 3:
        _b, g, r = cv2.split(work)
        gold = (r.astype(np.int16) - _b.astype(np.int16) >= 8) & (r.astype(np.int16) >= g.astype(np.int16) - 12)
        gold_cu = gold & (contrast > 2.0) & (den > 12)
        raw = ridge | gold_cu
    else:
        raw = ridge
    mask = np.where(raw, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(3 * min(px, 1.7)), _odd(3 * min(px, 1.7))))
    # Close only small breaks. Must stay well below the FR4 gap (~18 px on 20 MP).
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def _is_stadium(cnt: np.ndarray, hole: np.ndarray, img_area: float, min_area: float) -> bool:
    area = cv2.contourArea(cnt)
    if area < min_area or area > 0.95 * img_area:
        return False
    if len(cnt) < 40:
        return False
    _x, _y, rw, rh = cv2.boundingRect(cnt)
    if min(rw, rh) < 8:
        return False
    if max(rw, rh) / max(1.0, float(min(rw, rh))) < 2.2:
        return False
    hole_area = cv2.contourArea(hole)
    if hole_area < area * 0.18:
        return False
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area < 1.0 or area / hull_area < 0.78:
        return False
    # Hollow ring, not a solid blob: the outline should be a thin stadium.
    ring_area = max(area - hole_area, 1.0)
    if hole_area / area < 0.25:
        return False
    if ring_area / area > 0.82:
        return False
    return True


def _stadium_candidates(ring: np.ndarray, min_area: float) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    contours, hierarchy = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    hier = hierarchy[0]
    h, w = ring.shape
    img_area = float(h * w)
    found: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1:
            continue
        child = hier[i][2]
        if child == -1:
            continue
        hole = contours[child]
        hole_area = cv2.contourArea(hole)
        nxt = hier[child][0]
        while nxt != -1:
            a = cv2.contourArea(contours[nxt])
            if a > hole_area:
                hole_area = a
                hole = contours[nxt]
            nxt = hier[nxt][0]
        if not _is_stadium(cnt, hole, img_area, min_area):
            continue
        found.append((cv2.contourArea(cnt), cnt, hole))
    found.sort(key=lambda t: t[0], reverse=True)
    return found


def _centroid(cnt: np.ndarray) -> Tuple[int, int]:
    M = cv2.moments(cnt)
    if M["m00"] < 1e-6:
        x, y, w, h = cv2.boundingRect(cnt)
        return int(x + w / 2), int(y + h / 2)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def _pca_axes(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return mean, np.array([1.0, 0.0]), np.array([0.0, 1.0])
    _eigval, eigvec = np.linalg.eigh(cov)
    axis = eigvec[:, int(np.argmax(_eigval))]
    nrm = float(np.hypot(axis[0], axis[1])) + 1e-12
    axis = axis / nrm
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    return mean, axis, perp


def _stadium_polygon(
    mean: np.ndarray,
    axis: np.ndarray,
    perp: np.ndarray,
    u_c: float,
    v_c: float,
    half_straight: float,
    radius: float,
    n: int = 240,
) -> np.ndarray:
    hs = max(0.0, float(half_straight))
    r = max(1.0, float(radius))
    n_cap = max(20, n // 4)
    n_line = max(12, n // 4)
    us: List[float] = []
    vs: List[float] = []
    # Top straight, left → right.
    for t in np.linspace(-hs, hs, n_line, endpoint=False):
        us.append(float(t))
        vs.append(r)
    # Right cap, top → bottom through the outside.
    for a in np.linspace(0.5 * np.pi, -0.5 * np.pi, n_cap, endpoint=False):
        us.append(hs + r * float(np.cos(a)))
        vs.append(r * float(np.sin(a)))
    # Bottom straight, right → left.
    for t in np.linspace(hs, -hs, n_line, endpoint=False):
        us.append(float(t))
        vs.append(-r)
    # Left cap, bottom → top through the outside (3π/2 → π/2).
    for a in np.linspace(1.5 * np.pi, 0.5 * np.pi, n_cap, endpoint=False):
        us.append(-hs + r * float(np.cos(a)))
        vs.append(r * float(np.sin(a)))
    uv = np.stack([np.asarray(us) + u_c, np.asarray(vs) + v_c], axis=1)
    xy = mean.reshape(1, 2) + uv[:, 0:1] * axis.reshape(1, 2) + uv[:, 1:2] * perp.reshape(1, 2)
    return np.round(xy).astype(np.int32).reshape(-1, 1, 2)


def _stadium_params(pts: np.ndarray, px: float, edge_pct: float = 78.0) -> dict | None:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 24:
        return None
    mean, axis, perp = _pca_axes(pts)
    rel = pts - mean
    u = rel @ axis
    v = rel @ perp
    u_c = float(np.median(u))
    v_c = float(np.median(v))
    u0, u1 = np.percentile(u, [4.0, 96.0])
    mid = (u >= u0 + 0.20 * (u1 - u0)) & (u <= u1 - 0.20 * (u1 - u0))
    if int(mid.sum()) < 16:
        mid = np.ones(len(pts), dtype=bool)
    r = float(np.percentile(np.abs(v[mid] - v_c), edge_pct))
    r = max(6.0 * px, r)
    hs = max(0.35 * r, 0.5 * (u1 - u0) - r)
    poly = _stadium_polygon(mean, axis, perp, u_c, v_c, hs, r)
    return {
        "mean": mean,
        "axis": axis,
        "perp": perp,
        "u_c": u_c,
        "v_c": v_c,
        "hs": hs,
        "r": r,
        "poly": poly,
    }


def _stadium_from_points(pts: np.ndarray, px: float) -> Tuple[np.ndarray, float, float] | None:
    p = _stadium_params(pts, px)
    if p is None:
        return None
    return p["poly"], p["hs"], p["r"]


def _fit_stadium(cnt: np.ndarray, px: float = 1.0) -> np.ndarray:
    """Ideal outer-copper stadium. Do not follow carbon bites or scratch blobs."""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    fitted = _stadium_from_points(pts, px)
    if fitted is None:
        return cnt
    poly, _hs, _r = fitted
    if cv2.contourArea(poly) < 0.45 * cv2.contourArea(cnt):
        return cnt
    return poly


def _is_chamber_outer(copper: np.ndarray, outer: np.ndarray, px: float = 1.0) -> bool:
    """Outer copper sits in carbon. Inner copper hits the outer ring a few pixels outside."""
    pts = outer.reshape(-1, 2).astype(np.float32)
    if len(pts) < 24:
        return False
    pts = pts[:: max(1, len(pts) // 80)]
    cx, cy = pts.mean(axis=0)
    h, w = copper.shape
    near = 0
    n = 0
    s0 = max(2, int(round(2 * px)))
    s1 = max(s0 + 1, int(round(9 * px)))
    for x, y in pts:
        vx, vy = float(x) - cx, float(y) - cy
        nrm = float(np.hypot(vx, vy)) + 1e-6
        dx, dy = vx / nrm, vy / nrm
        hit = False
        for s in range(s0, s1):
            xx = int(round(x + dx * s))
            yy = int(round(y + dy * s))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                break
            if copper[yy, xx] > 0:
                hit = True
                break
        n += 1
        if hit:
            near += 1
    # Inner ring: most outward rays hit the outer copper (~0.7+).
    # Outer ring: rays go into carbon; board scratches only clip a minority.
    # 0.40 was too tight on 20 MP — a scratch 20 px outside looked like a second ring.
    return n > 0 and (near / float(n)) < 0.62


def _locate_chambers(work: np.ndarray, px: float | None = None) -> List[_Located]:
    """Step 1: frame each chamber with the ideal outer-copper stadium."""
    if px is None:
        px = _px(work)
    ring = _outer_stadium_mask(work, px)
    h, w = ring.shape
    min_area = max(280.0 * px * px, 0.00012 * float(h * w))
    raw = _stadium_candidates(ring, min_area)
    located: List[_Located] = []
    accepted: List[np.ndarray] = []
    for _area, outer, hole in raw:
        cx, cy = _centroid(outer)
        nested = False
        for prev in accepted:
            if cv2.pointPolygonTest(prev, (float(cx), float(cy)), False) >= 0:
                nested = True
                break
        if nested:
            continue
        if not _is_chamber_outer(ring, outer, px):
            continue
        outer = _fit_stadium(outer, px)
        accepted.append(outer)
        x, y, bw, bh = cv2.boundingRect(outer)
        pad = max(int(round(8 * px)), int(0.10 * min(bw, bh)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw + pad)
        y1 = min(h, y + bh + pad)
        located.append(_Located(outer=outer, hole=hole, x0=x0, y0=y0, x1=x1, y1=y1))
    located = _complete_by_pitch(located, work, px)
    located.sort(key=lambda c: cv2.boundingRect(c.outer)[1] * 10000 + cv2.boundingRect(c.outer)[0])
    return located


def _shift_ring(cnt: np.ndarray, dx: float, dy: float) -> np.ndarray:
    delta = np.array([[[int(round(dx)), int(round(dy))]]], dtype=np.int32)
    return cnt.astype(np.int32) + delta


def _contour_support(gray: np.ndarray, cnt: np.ndarray, px: float) -> float:
    """Median of (copper near the outline) minus (carbon just outside)."""
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if len(pts) < 24:
        return 0.0
    pts = pts[:: max(1, len(pts) // 96)]
    cx, cy = pts.mean(axis=0)
    h, w = gray.shape
    span = max(3, int(round(5 * px)))
    outside = span + max(4, int(round(8 * px)))
    deltas: List[float] = []
    for x, y in pts:
        vx, vy = float(x) - cx, float(y) - cy
        nrm = float(np.hypot(vx, vy)) + 1e-6
        dx, dy = vx / nrm, vy / nrm
        best = None
        for dist in range(-span, span + 1, max(1, span // 3)):
            xx = int(round(x + dx * dist))
            yy = int(round(y + dy * dist))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                continue
            val = float(gray[yy, xx])
            best = val if best is None else max(best, val)
        xx = int(round(x + dx * outside))
        yy = int(round(y + dy * outside))
        if best is None or xx < 0 or yy < 0 or xx >= w or yy >= h:
            continue
        deltas.append(best - float(gray[yy, xx]))
    if len(deltas) < 16:
        return 0.0
    return float(np.median(deltas))


def _complete_by_pitch(located: List[_Located], work: np.ndarray, px: float) -> List[_Located]:
    """Insert chambers the lamp broke open, using the pitch of the ones already found."""
    if len(located) < 2:
        return located
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    h, w = gray.shape
    centers = np.array([_centroid(c.outer) for c in located], dtype=np.float64)
    shorts: List[float] = []
    nn_vecs: List[np.ndarray] = []
    nn_dist: List[float] = []
    for i, loc in enumerate(located):
        _x, _y, bw, bh = cv2.boundingRect(loc.outer)
        shorts.append(float(min(bw, bh)))
        dist = np.linalg.norm(centers - centers[i], axis=1)
        dist[i] = 1e9
        j = int(np.argmin(dist))
        if dist[j] < 1e8:
            nn_vecs.append(centers[j] - centers[i])
            nn_dist.append(float(dist[j]))
    short = float(np.median(shorts)) if shorts else 0.0
    if short < 8.0 or not nn_dist:
        return located
    band = [d for d in nn_dist if 0.85 * short <= d <= 3.2 * short]
    pitch = float(np.median(band if band else nn_dist))
    if pitch < 10.0:
        return located
    ref = max(nn_vecs, key=lambda v: float(np.hypot(v[0], v[1])))
    acc = np.zeros(2, dtype=np.float64)
    for vec in nn_vecs:
        use = vec if float(np.dot(vec, ref)) >= 0.0 else -vec
        length = float(np.hypot(use[0], use[1]))
        if length < 1.0:
            continue
        acc += use / length
    nrm = float(np.hypot(acc[0], acc[1]))
    if nrm < 1e-6:
        return located
    direction = acc / nrm
    origin = centers[0]
    coord = (centers - origin) @ direction
    taken = coord.tolist()
    out = list(located)

    def near(t: float) -> bool:
        return any(abs(t - prev) < 0.42 * pitch for prev in taken)

    slots = [float(coord.min()) - pitch, float(coord.max()) + pitch]
    t = float(coord.min())
    while t < float(coord.max()) - 0.4 * pitch:
        t += pitch
        slots.append(t)
    for t in slots:
        if near(t):
            continue
        src_i = int(np.argmin(np.abs(coord - t)))
        src = located[src_i]
        delta = (t - float(coord[src_i])) * direction
        moved = _shift_ring(src.outer, float(delta[0]), float(delta[1]))
        xs = moved.reshape(-1, 2)
        inside = (
            (xs[:, 0] >= 0) & (xs[:, 0] < w) & (xs[:, 1] >= 0) & (xs[:, 1] < h)
        )
        if float(inside.mean()) < 0.75:
            continue
        if _contour_support(gray, moved, px) < 7.0:
            continue
        cx, cy = _centroid(moved)
        if any(cv2.pointPolygonTest(prev.outer, (float(cx), float(cy)), False) >= 0 for prev in out):
            continue
        hole = _shift_ring(src.hole, float(delta[0]), float(delta[1]))
        x, y, bw, bh = cv2.boundingRect(moved)
        pad = max(int(round(8 * px)), int(0.10 * min(bw, bh)))
        out.append(
            _Located(
                outer=moved,
                hole=hole,
                x0=max(0, x - pad),
                y0=max(0, y - pad),
                x1=min(w, x + bw + pad),
                y1=min(h, y + bh + pad),
            )
        )
        taken.append(t)
    return out


def _analyse_chamber(work: np.ndarray, loc: _Located, index: int, px: float = 1.0) -> ChamberResult:
    crop = work[loc.y0 : loc.y1, loc.x0 : loc.x1]
    if crop.size == 0:
        return ChamberResult(index, "不能识别", loc.outer, marks=[Mark("不能识别", *_centroid(loc.outer), "裁切失败")])

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    origin = np.array([[[loc.x0, loc.y0]]], dtype=loc.outer.dtype)
    outer = loc.outer - origin
    copper = _outer_stadium_mask(crop, px)

    sat_mask = _fill(outer, gray.shape)
    sat_vals = gray[sat_mask > 0]
    if sat_vals.size and float((sat_vals >= 250).mean()) > 0.35:
        return ChamberResult(index, "不能识别", loc.outer, marks=[Mark("不能识别", *_centroid(loc.outer), "铜环过曝")])

    layers = _trace_from_outer(gray, copper, outer, px)
    if layers["n"] < 16:
        return ChamberResult(
            index, "不能识别", loc.outer, marks=[Mark("不能识别", *_centroid(loc.outer), "套不到内圈铜/隔离缝")]
        )
    median_w = float(np.median(layers["gap_w"]))
    if median_w < 1.2 * px:
        return ChamberResult(
            index, "不能识别", loc.outer, 0.0, [Mark("不能识别", *_centroid(loc.outer), "隔离缝看不见")]
        )

    marks: List[Mark] = []
    marks.extend(_offset_marks(_bleed_from_layers(layers), loc))
    marks.extend(_offset_marks(_void_from_layers(layers, px), loc))

    if any(m.kind == "渗墨" for m in marks):
        verdict = "渗墨"
    elif any(m.kind == "缺墨" for m in marks):
        verdict = "缺墨"
    else:
        verdict = "合格"
    gap_line = _gap_centerline(copper, outer, px, loc)
    return ChamberResult(index, verdict, loc.outer, median_w, marks, gap_line)


def _offset_marks(marks: List[Mark], loc: _Located) -> List[Mark]:
    out: List[Mark] = []
    for m in marks:
        out.append(Mark(m.kind, m.x + loc.x0, m.y + loc.y0, m.note))
    return out


def _shift_contour(cnt: np.ndarray, x0: int, y0: int) -> np.ndarray:
    return cnt.astype(np.int32) + np.array([[[int(x0), int(y0)]]], dtype=np.int32)


def _fr4_midpoints(copper: np.ndarray, outer: np.ndarray, px: float) -> np.ndarray | None:
    """Walk inward from the outer copper: outer Cu → narrow FR4 → inner Cu.

    The center hole is farther in. A dark run that wide is the cavity, not the seam.
    """
    pts = outer.reshape(-1, 2).astype(np.float64)
    if len(pts) < 40:
        return None
    step = max(1, len(pts) // 180)
    pts = pts[::step]
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    h, w = copper.shape
    max_steps = max(12, int(min(h, w) * 0.45))
    max_crack = max(3, int(round(2.5 * px)))
    # 0.3 mm is about 6 px at REF_SIDE. Oblique rays and a little bleed run longer.
    # The cavity is much wider, so those rays are dropped instead of drawn inside.
    max_gap = max(max_crack + 2, int(round(26 * px)))
    mids: List[Tuple[float, float]] = []
    widths: List[float] = []
    for x0, y0 in pts:
        vx, vy = cx - float(x0), cy - float(y0)
        nrm = float(np.hypot(vx, vy)) + 1e-6
        dx, dy = vx / nrm, vy / nrm
        seen_outer = False
        gap_pts: List[Tuple[float, float]] = []
        for s in range(max_steps):
            xx = int(round(float(x0) + dx * s))
            yy = int(round(float(y0) + dy * s))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                break
            on = copper[yy, xx] > 0
            if not seen_outer:
                if on:
                    seen_outer = True
                continue
            if on:
                if not gap_pts:
                    continue
                if len(gap_pts) <= max_crack:
                    gap_pts.clear()
                    continue
                if len(gap_pts) <= max_gap:
                    mids.append(gap_pts[len(gap_pts) // 2])
                    widths.append(float(len(gap_pts)))
                break
            gap_pts.append((float(xx), float(yy)))
            if len(gap_pts) > max_gap:
                break
    if len(mids) < 24:
        return None
    arr = np.asarray(mids, np.float64)
    wds = np.asarray(widths, np.float64)
    med = float(np.median(wds))
    keep = (wds >= 0.55 * med) & (wds <= 1.70 * med)
    if int(keep.sum()) >= 24:
        arr = arr[keep]
    return arr


def _seam_stadium(samples: np.ndarray, px: float) -> np.ndarray | None:
    """Stadium through the seam samples. Each long side sets its own median.

    Folding both sides with abs() around the outer-copper center inflates the
    radius to the farther side and throws the nearer side off the seam.
    """
    pts = np.asarray(samples, np.float64).reshape(-1, 2)
    if len(pts) < 24:
        return None
    mean, axis, perp = _pca_axes(pts)
    rel = pts - mean
    u = rel @ axis
    v = rel @ perp
    u0, u1 = np.percentile(u, [2.0, 98.0])
    span = float(u1 - u0)
    if span < 8.0 * px:
        return None
    band = (u >= u0 + 0.22 * span) & (u <= u1 - 0.22 * span)
    vv = v[band] if int(band.sum()) >= 8 else v
    vv = np.sort(vv)
    if len(vv) < 8:
        return None
    gaps = np.diff(vv)
    split = int(np.argmax(gaps))
    lo = vv[: split + 1]
    hi = vv[split + 1 :]
    if len(lo) < 4 or len(hi) < 4 or float(gaps[split]) < 4.0 * px:
        return None
    v_lo = float(np.median(lo))
    v_hi = float(np.median(hi))
    v_c = 0.5 * (v_hi + v_lo)
    r = 0.5 * (v_hi - v_lo)
    if r < 4.0 * px:
        return None
    u_c = 0.5 * (float(u0) + float(u1))
    hs = max(0.35 * r, 0.5 * span - r)
    return _stadium_polygon(mean, axis, perp, u_c, v_c, hs, r)


def _gap_centerline(
    copper: np.ndarray,
    outer: np.ndarray,
    px: float,
    loc: _Located,
) -> np.ndarray | None:
    """Single line in the middle of the 0.3 mm FR4 seam."""
    samples = _fr4_midpoints(copper, outer, px)
    if samples is None:
        return None
    poly = _seam_stadium(samples, px)
    if poly is None:
        return None
    # A fit around the center hole is much smaller than the outer copper.
    if cv2.contourArea(poly) < 0.45 * max(cv2.contourArea(outer), 1.0):
        return None
    return _shift_contour(poly, loc.x0, loc.y0)


def _trace_from_outer(
    gray: np.ndarray,
    copper: np.ndarray,
    outer: np.ndarray,
    px: float = 1.0,
) -> dict:
    """Walk inward from the ideal outer copper: outer Cu → FR4 → inner Cu → inner carbon.

    Short dark breaks in the copper mask are carbon covering copper — same ring, skip.
    The isolation gap is the run between the two copper rings (olive or, if 渗墨, dark).
    """
    pts = outer.reshape(-1, 2).astype(np.float32)
    if len(pts) < 40:
        return {"n": 0}
    step = max(1, len(pts) // 180)
    pts = pts[::step]
    cx, cy = pts.mean(axis=0)
    h, w = copper.shape
    max_steps = max(12, int(min(h, w) * 0.45))
    core_band = max(6, int(round(6 * px)))
    max_crack = max(3, int(round(2.5 * px)))
    gap_w, gap_g, gap_xy = [], [], []
    gap_outer_xy, gap_inner_xy = [], []
    gap_s0, gap_s1 = [], []
    core_g, core_xy, core_r = [], [], []
    for x, y in pts:
        vx, vy = cx - x, cy - y
        nrm = float(np.hypot(vx, vy)) + 1e-6
        dx, dy = vx / nrm, vy / nrm
        on_outer = False
        in_inner = False
        in_core = False
        pending_vals: List[float] = []
        pending_pts: List[Tuple[float, float]] = []
        gap_vals: List[float] = []
        gap_pts: List[Tuple[float, float]] = []
        core_vals: List[float] = []
        core_pts: List[Tuple[float, float]] = []
        hit_inner = False
        inner_break = 0
        pending_s0 = 0
        commit_s1 = 0
        for s in range(max_steps):
            xx = int(round(x + dx * s))
            yy = int(round(y + dy * s))
            if xx < 0 or yy < 0 or xx >= w or yy >= h:
                break
            on = copper[yy, xx] > 0
            gv = float(gray[yy, xx])
            if not on_outer:
                if on:
                    on_outer = True
                continue
            if on_outer and not in_inner and not in_core:
                if on:
                    if pending_vals:
                        if len(pending_vals) <= max_crack:
                            pending_vals.clear()
                            pending_pts.clear()
                        else:
                            gap_vals = pending_vals
                            gap_pts = pending_pts
                            pending_vals = []
                            pending_pts = []
                            in_inner = True
                            hit_inner = True
                            inner_break = 0
                            commit_s1 = s
                    continue
                if not pending_vals:
                    pending_s0 = s
                pending_vals.append(gv)
                pending_pts.append((float(xx), float(yy)))
                continue
            if in_inner and not in_core:
                if on:
                    inner_break = 0
                    continue
                inner_break += 1
                if inner_break <= max_crack:
                    continue
                in_core = True
                core_vals.append(gv)
                core_pts.append((float(xx), float(yy)))
                continue
            if in_core:
                if on:
                    break
                if len(core_vals) >= core_band:
                    break
                core_vals.append(gv)
                core_pts.append((float(xx), float(yy)))
        if hit_inner and len(gap_vals) > 0:
            mid = gap_pts[len(gap_pts) // 2]
            gap_w.append(float(len(gap_vals)))
            gap_g.append(float(np.mean(gap_vals)))
            gap_xy.append(mid)
            gap_outer_xy.append(gap_pts[0])
            gap_inner_xy.append(gap_pts[-1])
            gap_s0.append(float(pending_s0))
            gap_s1.append(float(commit_s1))
            if core_vals:
                cmid = core_pts[0]
                core_g.append(float(np.mean(core_vals)))
                core_xy.append(cmid)
                core_r.append(float(np.hypot(cmid[0] - cx, cmid[1] - cy)))
            else:
                core_g.append(np.nan)
                core_xy.append((float("nan"), float("nan")))
                core_r.append(np.nan)
    return {
        "n": len(gap_w),
        "gap_w": np.asarray(gap_w, np.float32),
        "gap_g": np.asarray(gap_g, np.float32),
        "gap_xy": np.asarray(gap_xy, np.float32),
        "gap_outer_xy": np.asarray(gap_outer_xy, np.float32),
        "gap_inner_xy": np.asarray(gap_inner_xy, np.float32),
        "gap_s0": np.asarray(gap_s0, np.float32),
        "gap_s1": np.asarray(gap_s1, np.float32),
        "core_g": np.asarray(core_g, np.float32),
        "core_xy": np.asarray(core_xy, np.float32),
        "core_r": np.asarray(core_r, np.float32),
        "mean_r": float(np.median(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))),
    }


def _circular_runs(
    flag: np.ndarray,
    samples: np.ndarray,
    kind: str,
    note: str,
    min_run: int,
) -> List[Mark]:
    marks: List[Mark] = []
    n = len(flag)
    if n == 0 or samples.shape[0] != n:
        return marks
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < n and flag[j]:
            j += 1
        run = j - i
        mid_idx = min((i + j) // 2, n - 1)
        if i == 0:
            k = n - 1
            while k > j and flag[k]:
                k -= 1
            extra = n - 1 - k
            if extra > 0:
                run += extra
                mid_idx = 0
        if run >= min_run:
            x, y = samples[mid_idx]
            if np.isfinite(x) and np.isfinite(y):
                marks.append(Mark(kind, int(round(float(x))), int(round(float(y))), note))
        i = j
    return marks


def _bleed_from_layers(layers: dict) -> List[Mark]:
    gray = layers["gap_g"]
    if gray.size < 16:
        return []
    med = float(np.median(gray))
    mad = float(np.median(np.abs(gray - med)))
    floor = med - max(10.0, 1.8 * max(mad, 3.0), 0.18 * med)
    if med < 38.0 and float(np.percentile(gray, 80)) < 55.0:
        xy = layers["gap_xy"]
        x, y = xy[len(xy) // 2]
        return [Mark("渗墨", int(round(x)), int(round(y)), "间隔被碳糊满")]
    dark = gray < floor
    return _circular_runs(dark, layers["gap_xy"], "渗墨", "间隔进碳", 6)


def _void_from_layers(layers: dict, px: float = 1.0) -> List[Mark]:
    gray = layers["core_g"]
    xy = layers["core_xy"]
    radius = layers["core_r"]
    if gray.size < 16:
        return []
    valid = np.isfinite(gray) & np.isfinite(radius)
    if int(valid.sum()) < 12:
        return []
    gap_med = float(np.median(layers["gap_g"]))
    gap_w = float(np.median(layers["gap_w"]))
    mean_r = max(layers["mean_r"], 1.0)
    inset = mean_r - radius
    wall = valid & (radius >= 0.38 * mean_r) & (inset >= (gap_w + 2.0 * max(px, 1.0)))
    if int(wall.sum()) < 12:
        return []
    carbon = float(np.nanmedian(gray[wall]))
    if gap_med - carbon < 15.0:
        return []
    thresh = max(carbon + 0.45 * (gap_med - carbon), carbon + 18.0)
    bright = np.zeros(len(gray), dtype=bool)
    bright[wall] = gray[wall] >= thresh
    return _circular_runs(bright, xy, "缺墨", "内圈碳油露板", 8)


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
    for text, xy, color in labels:
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        use = font if text.startswith("电离室") or text.startswith("不能识别：") else small
        draw.text(xy, text, font=use, fill=rgb)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
