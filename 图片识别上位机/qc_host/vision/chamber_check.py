# -*- coding: utf-8 -*-
"""Check one located chamber for carbon in the gap (渗墨) and bare board in the cavity (缺墨).

The chamber is placed from the hole and the layout sizes in hole_frame. Only two regions are
looked at: the 0.3 mm gap between the two copper rings, and the cavity inside the inner copper
without the hole and stem. Carbon on either copper is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Keep clear of the copper glare at the gap edges and of the copper step at the cavity edge.
GAP_MARGIN = 2.5
CAVITY_MARGIN = 2.0
HOLE_MARGIN = 3.0
STEM_HALF_WIDTH = 4.0
MIN_BLOB = 4

MIN_SCORE = 0.6
FRAME_MARGIN = 3.0
V_RANGE = (35.0, 200.0)
MIN_SEPARATION = 30.0
MIN_COPPER_LIFT = 30.0
MIN_SHARPNESS = 20.0
MIN_CAP_RATIO = 0.5
SEAM_HUE = (25, 55)
# Along the ring the gap saturation drifts (the caps read paler than the straights), so a pixel only
# counts as carbon when it sits close to the cavity carbon, and as bare board when close to the gap.
CARBON_FRACTION = 0.3
BOARD_FRACTION = 0.7
# Most of each copper ring is under carbon; the bare part near the gap is the bright tail.
COPPER_PERCENTILE = 80
# Provisional: good chambers on light_low.png top out at 16 px. Set from defect samples later.
DEFECT_AREA = 30
# Copper-edge glare and the copper step often look like thin carbon along the gap boundary.
# Real seepage reaches the middle of the 0.3 mm gap; strips that only hug the edge are dropped.
# The usable gap after GAP_MARGIN is only ~8 px wide, so 3 px from each wall leaves the mid band.
CARBON_EDGE_CLEAR = 3.0
# Shadow that covers both the gap and the copper rings also fills the mid band. A blob only counts
# when the copper beside it is still clearly brighter than the gap.
CARBON_SIDE_LIFT = 25.0
CARBON_SIDE_RADIUS = 10
# The board can sit a few degrees off square; the stem only gives the nearest 90 degrees.
MAX_TILT = 6.0
TILT_STEP = 0.25
# The ring is cut into segments and every one must show copper brighter than the gap and a gap
# more saturated than the cavity carbon. Good chambers bottom out near 30 on both; a box that is
# off the rings, or a hit on something that is not a chamber, goes to 0 or below.
SEGMENTS = 12
SAMPLES_PER_SEGMENT = 20
MIN_SEGMENT_LIFT = 15.0
MIN_SEGMENT_SEPARATION = 15.0


@dataclass
class Blob:
    area: int
    box: tuple[int, int, int, int]


@dataclass
class ChamberReport:
    usable: bool
    reasons: list[str]
    stats: dict[str, float]
    gap_mask: np.ndarray
    cavity_mask: np.ndarray
    carbon_in_gap: np.ndarray
    board_in_cavity: np.ndarray
    long_angle: float = 0.0
    tilt: float = 0.0
    gap_blobs: list[Blob] = field(default_factory=list)
    cavity_blobs: list[Blob] = field(default_factory=list)


def _ring(center, long_angle: float, radius: float, half: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Points evenly spaced along one stadium outline, starting at one straight side."""
    length = 4.0 * half + 2.0 * np.pi * radius
    s = (np.arange(count) + 0.5) / count * length
    lx = np.empty(count)
    ly = np.empty(count)
    arc = np.pi * radius
    side = s < 2.0 * half
    lx[side], ly[side] = -half + s[side], radius
    cap = (s >= 2.0 * half) & (s < 2.0 * half + arc)
    t = (s[cap] - 2.0 * half) / radius
    lx[cap], ly[cap] = half + radius * np.sin(t), radius * np.cos(t)
    back = (s >= 2.0 * half + arc) & (s < 4.0 * half + arc)
    lx[back], ly[back] = half - (s[back] - 2.0 * half - arc), -radius
    rest = s >= 4.0 * half + arc
    t = (s[rest] - 4.0 * half - arc) / radius
    lx[rest], ly[rest] = -half - radius * np.sin(t), -radius * np.cos(t)
    along_x, along_y = np.cos(long_angle), np.sin(long_angle)
    cx, cy = center
    return cx + lx * along_x - ly * along_y, cy + lx * along_y + ly * along_x


def _bands(plane: np.ndarray, center, long_angle: float, radii, half: float) -> np.ndarray:
    """Plane values along several outlines, one row per radius."""
    count = SEGMENTS * SAMPLES_PER_SEGMENT
    rows = []
    for radius in radii:
        xs, ys = _ring(center, long_angle, radius, half, count)
        sampled = cv2.remap(
            plane, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        rows.append(sampled.reshape(-1))
    return np.asarray(rows)


def _copper_radii(edge_radii) -> list[float]:
    inner_in, inner_out, outer_in, outer_out = edge_radii
    return list(np.linspace(inner_in + 2.0, inner_out - 2.0, 4)) + list(np.linspace(outer_in + 2.0, outer_out - 2.0, 4))


def _gap_radii(edge_radii) -> list[float]:
    _inner_in, inner_out, outer_in, _outer_out = edge_radii
    return list(np.linspace(inner_out + 3.0, outer_in - 3.0, 3))


def frame_fit(val: np.ndarray, hole, base_angle: float, edge_radii, half_straight: float) -> tuple[float, float]:
    """Degrees to turn the frame about the hole so the copper rings land on the bright copper,
    and how much brighter the copper then is than the gap. A hole that is not a chamber hole
    finds no turn that puts both rings on copper."""
    best_score, best_tilt = -np.inf, 0.0
    for tilt in np.arange(-MAX_TILT, MAX_TILT + 1e-6, TILT_STEP):
        angle = base_angle + np.deg2rad(tilt)
        copper = _bands(val, hole, angle, _copper_radii(edge_radii), half_straight).mean()
        gap = _bands(val, hole, angle, _gap_radii(edge_radii), half_straight).mean()
        if copper - gap > best_score:
            best_score, best_tilt = float(copper - gap), float(tilt)
    return best_tilt, best_score


def _segment_worst(val, sat, hole, long_angle, edge_radii, half_straight, cavity_s) -> tuple[float, float]:
    """Weakest segment: copper brightness over the gap, and gap saturation over the carbon."""
    copper = _bands(val, hole, long_angle, _copper_radii(edge_radii), half_straight)
    gap_v = _bands(val, hole, long_angle, _gap_radii(edge_radii), half_straight)
    gap_s = _bands(sat, hole, long_angle, _gap_radii(edge_radii), half_straight)
    lifts, separations = [], []
    for k in range(SEGMENTS):
        part = slice(k * SAMPLES_PER_SEGMENT, (k + 1) * SAMPLES_PER_SEGMENT)
        lifts.append(float(np.percentile(copper[:, part], COPPER_PERCENTILE) - np.median(gap_v[:, part])))
        separations.append(float(np.median(gap_s[:, part]) - cavity_s))
    return min(lifts), min(separations)


def _poly(center: tuple[float, float], long_angle: float, radius: float, half: float) -> np.ndarray:
    cx, cy = center
    along_x, along_y = float(np.cos(long_angle)), float(np.sin(long_angle))
    side_x, side_y = -along_y, along_x
    points = []
    for end, degrees in ((half, np.arange(-90, 91, 2)), (-half, np.arange(90, 271, 2))):
        for degree in degrees:
            angle = np.deg2rad(degree)
            lx = end + radius * np.cos(angle)
            ly = radius * np.sin(angle)
            points.append((cx + lx * along_x + ly * side_x, cy + lx * along_y + ly * side_y))
    # fillPoly with shift=4 keeps quarter-pixel positions.
    return np.round(np.asarray(points) * 16).astype(np.int32).reshape(-1, 1, 2)


def _fill(shape: tuple[int, int], center, long_angle: float, radius: float, half: float) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [_poly(center, long_angle, radius, half)], 255, cv2.LINE_8, 4)
    return mask


def _blobs(mask: np.ndarray) -> list[Blob]:
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = [
        Blob(int(stats[i, cv2.CC_STAT_AREA]), tuple(int(v) for v in stats[i, :4]))
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB
    ]
    out.sort(key=lambda b: -b.area)
    return out


def _keep_blobs(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_BLOB
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _seepage_blobs(
    carbon: np.ndarray,
    gap_mask: np.ndarray,
    inner_copper: np.ndarray,
    outer_copper: np.ndarray,
    val: np.ndarray,
    gap_v: float,
) -> np.ndarray:
    """Keep carbon blobs that sit in the gap mid-band between still-bright copper rings.

    Thin edge strips are already removed by the mid-band mask. A dark shadow that covers the gap
    together with one copper ring fails the two-sided copper check and is dropped.
    """
    mask = _keep_blobs(np.where(carbon, 255, 0).astype(np.uint8))
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CARBON_SIDE_RADIUS + 1, 2 * CARBON_SIDE_RADIUS + 1))
    keep = np.zeros(count, bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < MIN_BLOB:
            continue
        blob = labels == index
        near = cv2.dilate(blob.astype(np.uint8), kernel, iterations=1) > 0
        side_in = near & (inner_copper > 0) & (gap_mask == 0)
        side_out = near & (outer_copper > 0) & (gap_mask == 0)
        if not side_in.any() or not side_out.any():
            continue
        lift_in = float(np.percentile(val[side_in], COPPER_PERCENTILE)) - gap_v
        lift_out = float(np.percentile(val[side_out], COPPER_PERCENTILE)) - gap_v
        if min(lift_in, lift_out) >= CARBON_SIDE_LIFT:
            keep[index] = True
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def verdict(report: ChamberReport, px_per_mm: float) -> tuple[str, list[str]]:
    """Result word and detail lines for the operator."""
    if not report.usable:
        return "无法判定", report.reasons
    lines: list[str] = []
    found: list[str] = []
    mm2 = px_per_mm * px_per_mm
    for name, blobs, what in (
        ("渗墨", report.gap_blobs, "缝里疑似炭黑"),
        ("缺墨", report.cavity_blobs, "内腔疑似露板"),
    ):
        largest = blobs[0].area if blobs else 0
        total = sum(b.area for b in blobs)
        lines.append(f"{what}：最大一块 {largest} 像素（{largest / mm2:.3f} mm²），共 {total} 像素")
        if largest >= DEFECT_AREA:
            found.append(name)
    stats = report.stats
    lines.append(
        f"缝饱和度 {stats['gap_s']:.0f}，炭黑饱和度 {stats['cavity_s']:.0f}，"
        f"低于 {stats['carbon_below']:.0f} 算炭黑，高于 {stats['board_above']:.0f} 算露板，"
        f"一块达到 {DEFECT_AREA} 像素报问题（暂定）"
    )
    return ("、".join(found) if found else "未发现问题"), lines


def check_chamber(
    image: np.ndarray,
    hole: tuple[float, float],
    stem: tuple[int, int],
    stem_shift: float,
    hole_radius: float,
    score: float,
    edge_radii: tuple[float, float, float, float],
    half_straight: float,
) -> ChamberReport:
    height, width = image.shape[:2]
    shape = (height, width)
    x, y = hole
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(hsv, (3, 3), 0)
    smooth_s, smooth_v = smooth[:, :, 1].astype(np.float32), smooth[:, :, 2].astype(np.float32)
    tilt, _fit = frame_fit(smooth_v, hole, float(np.arctan2(stem[0], -stem[1])), edge_radii, half_straight)
    turn = np.deg2rad(tilt)
    dx = float(stem[0] * np.cos(turn) - stem[1] * np.sin(turn))
    dy = float(stem[0] * np.sin(turn) + stem[1] * np.cos(turn))
    long_angle = float(np.arctan2(dx, -dy))
    inner_in, inner_out, outer_in, outer_out = edge_radii

    gap_mask = cv2.subtract(
        _fill(shape, hole, long_angle, outer_in - GAP_MARGIN, half_straight),
        _fill(shape, hole, long_angle, inner_out + GAP_MARGIN, half_straight),
    )
    cavity_mask = _fill(shape, hole, long_angle, inner_in - CAVITY_MARGIN, half_straight)
    cv2.circle(cavity_mask, (int(round(x * 16)), int(round(y * 16))), int(round((hole_radius + HOLE_MARGIN) * 16)), 0, -1, cv2.LINE_8, 4)
    side_x, side_y = -dy * stem_shift, dx * stem_shift
    stem_a = (x + side_x, y + side_y)
    stem_b = (x + side_x + dx * inner_in, y + side_y + dy * inner_in)
    cv2.line(
        cavity_mask,
        (int(round(stem_a[0] * 16)), int(round(stem_a[1] * 16))),
        (int(round(stem_b[0] * 16)), int(round(stem_b[1] * 16))),
        0,
        int(round(2 * STEM_HALF_WIDTH)),
        cv2.LINE_8,
        4,
    )
    inner_copper = cv2.subtract(
        _fill(shape, hole, long_angle, inner_out - 2.0, half_straight),
        _fill(shape, hole, long_angle, inner_in + 2.0, half_straight),
    )
    outer_copper = cv2.subtract(
        _fill(shape, hole, long_angle, outer_out - 2.0, half_straight),
        _fill(shape, hole, long_angle, outer_in + 2.0, half_straight),
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gap = gap_mask > 0
    cavity = cavity_mask > 0

    stats: dict[str, float] = {"score": float(score), "tilt": tilt}
    reasons: list[str] = []
    if abs(tilt) >= MAX_TILT:
        reasons.append(f"板子歪了 {tilt:+.1f}° 以上，摆正一点再拍")
    if score < MIN_SCORE:
        reasons.append(f"圆孔匹配得分 {score:.2f} 低于 {MIN_SCORE}")

    along_x, along_y = float(np.cos(long_angle)), float(np.sin(long_angle))
    side_ux, side_uy = -along_y, along_x
    reach = half_straight + outer_out + FRAME_MARGIN
    corners = [
        (x + sa * reach * along_x + sb * (outer_out + FRAME_MARGIN) * side_ux,
         y + sa * reach * along_y + sb * (outer_out + FRAME_MARGIN) * side_uy)
        for sa in (-1, 1)
        for sb in (-1, 1)
    ]
    if any(cx < 0 or cy < 0 or cx >= width or cy >= height for cx, cy in corners):
        reasons.append("电离室没有完整框在选区里")

    if not gap.any() or not cavity.any():
        reasons.append("缝或内腔区域是空的")
        empty = np.zeros(shape, np.uint8)
        return ChamberReport(False, reasons, stats, gap_mask, cavity_mask, empty, empty.copy(), long_angle, tilt)

    gap_s = float(np.median(sat[gap]))
    cavity_s = float(np.median(sat[cavity]))
    gap_v = float(np.median(val[gap]))
    cavity_v = float(np.median(val[cavity]))
    inner_v = float(np.percentile(val[inner_copper > 0], COPPER_PERCENTILE))
    outer_v = float(np.percentile(val[outer_copper > 0], COPPER_PERCENTILE))
    stats.update(gap_s=gap_s, cavity_s=cavity_s, gap_v=gap_v, cavity_v=cavity_v, inner_v=inner_v, outer_v=outer_v)

    if not (V_RANGE[0] <= gap_v <= V_RANGE[1]) or not (V_RANGE[0] <= cavity_v <= V_RANGE[1]):
        reasons.append(f"亮度不合适（缝 {gap_v:.0f}，内腔 {cavity_v:.0f}）")
    separation = gap_s - cavity_s
    stats["separation"] = separation
    if separation < MIN_SEPARATION:
        reasons.append(f"缝和炭黑的饱和度差 {separation:.0f}，分不开")
    lift = min(inner_v, outer_v) - max(gap_v, cavity_v)
    stats["copper_lift"] = lift
    if lift < MIN_COPPER_LIFT:
        reasons.append(f"铜环不比缝和内腔亮（差 {lift:.0f}），框可能没套准")
    worst_lift, worst_separation = _segment_worst(
        smooth_v, smooth_s, hole, long_angle, edge_radii, half_straight, cavity_s
    )
    stats.update(segment_lift=worst_lift, segment_separation=worst_separation)
    if worst_lift < MIN_SEGMENT_LIFT:
        reasons.append(f"有一段铜环不比缝亮（差 {worst_lift:.0f}），框没套准或尺寸不符")
    if worst_separation < MIN_SEGMENT_SEPARATION:
        reasons.append(f"有一段缝的颜色和炭黑分不开（差 {worst_separation:.0f}），框没套准或尺寸不符")

    chamber = cv2.bitwise_or(outer_copper, _fill(shape, hole, long_angle, outer_in, half_straight)) > 0
    sharp = float(cv2.Laplacian(gray, cv2.CV_32F)[chamber].var())
    stats["sharpness"] = sharp
    if sharp < MIN_SHARPNESS:
        reasons.append(f"对焦不清（清晰度 {sharp:.0f}）")

    cap_values = []
    for sign in (-1.0, 1.0):
        cap = np.zeros(shape, np.uint8)
        cap_x = x + sign * half_straight * along_x
        cap_y = y + sign * half_straight * along_y
        cv2.circle(cap, (int(round(cap_x)), int(round(cap_y))), int(round(outer_out)), 255, -1)
        cap = cv2.bitwise_and(cap, cv2.bitwise_or(inner_copper, outer_copper))
        cap_values.append(float(np.median(val[cap > 0])) if (cap > 0).any() else 0.0)
    cap_ratio = min(cap_values) / max(max(cap_values), 1.0)
    stats["cap_ratio"] = cap_ratio
    if cap_ratio < MIN_CAP_RATIO:
        reasons.append(f"两端光照不均（暗的一头只有亮的 {cap_ratio:.0%}）")

    carbon_below = cavity_s + CARBON_FRACTION * separation
    board_above = cavity_s + BOARD_FRACTION * separation
    stats["carbon_below"] = carbon_below
    stats["board_above"] = board_above
    # Drop pixels that only sit on the gap's copper-side boundary; keep what reaches the middle.
    gap_depth = cv2.distanceTransform(gap_mask, cv2.DIST_L2, 3)
    carbon = gap & (gap_depth >= CARBON_EDGE_CLEAR) & (sat < carbon_below) & (val < gap_v + 40.0)
    seam_hue = (hue >= SEAM_HUE[0]) & (hue <= SEAM_HUE[1])
    board = cavity & (sat > board_above) & seam_hue
    carbon_mask = _seepage_blobs(carbon, gap_mask, inner_copper, outer_copper, val, gap_v)
    board_mask = _keep_blobs(np.where(board, 255, 0).astype(np.uint8))
    stats["gap_area"] = float(gap.sum())
    stats["cavity_area"] = float(cavity.sum())

    return ChamberReport(
        usable=not reasons,
        reasons=reasons,
        stats=stats,
        gap_mask=gap_mask,
        cavity_mask=cavity_mask,
        carbon_in_gap=carbon_mask,
        board_in_cavity=board_mask,
        long_angle=long_angle,
        tilt=tilt,
        gap_blobs=_blobs(carbon_mask),
        cavity_blobs=_blobs(board_mask),
    )
