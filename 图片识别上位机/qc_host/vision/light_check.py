# -*- coding: utf-8 -*-
"""Lighting health check for one located chamber.

Each number is scored by how far it sits from the gate chamber_check uses, not against an ideal
look: 不行 fails the gate (the chamber would be 无法判定), 余量小 passes close to it, 余量足 is
where the good photos so far sit. Copper glare is only reported: on every photo that judged well
13-36 % of the copper pixels were already at 255.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from qc_host.vision import chamber_check as cc
from qc_host.vision.grab import CameraSettings

BAD, LOW, GOOD = "不行", "余量小", "余量足"
LEVEL_ORDER = {GOOD: 0, LOW: 1, BAD: 2}
CLIP_VALUE = 250
# Lighting targets for the panel. These are stricter than the gates in chamber_check: a chamber can
# still be judged below them, but the light check calls it 不行 so the lamp gets turned up first.
CAVITY_V_MIN = 60.0
SEPARATION_MIN = 50.0
SEGMENT_SEPARATION_MIN = 40.0
COPPER_LIFT_MIN = 100.0


@dataclass
class LightItem:
    name: str
    text: str
    level: str
    hint: str = ""


def _band(shape, hole, long_angle, outer: float, inner: float, half: float) -> np.ndarray:
    return cv2.subtract(cc._fill(shape, hole, long_angle, outer, half), cc._fill(shape, hole, long_angle, inner, half)) > 0


def _level_min(value: float, gate: float, good: float) -> str:
    if value < gate:
        return BAD
    return GOOD if value >= good else LOW


def _level_range(value: float, gate: tuple[float, float], good: tuple[float, float]) -> str:
    if not (gate[0] <= value <= gate[1]):
        return BAD
    return GOOD if good[0] <= value <= good[1] else LOW


def light_items(
    image: np.ndarray,
    report: cc.ChamberReport,
    hole: tuple[float, float],
    edge_radii: tuple[float, float, float, float],
    half_straight: float,
    settings: CameraSettings | None,
) -> list[LightItem]:
    stats = report.stats
    items: list[LightItem] = []
    if "gap_v" not in stats:
        return [LightItem("定位", "缝或内腔区域是空的", BAD, "重新框一颗完整的电离室")]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]
    gap = report.gap_mask > 0
    shape = image.shape[:2]
    inner_in, inner_out, outer_in, outer_out = edge_radii
    copper = _band(shape, hole, report.long_angle, inner_out, inner_in, half_straight) | _band(
        shape, hole, report.long_angle, outer_out, outer_in, half_straight
    )

    gap_v = stats["gap_v"]
    items.append(LightItem(
        "缝亮度", f"{gap_v:.0f}（门槛 {cc.V_RANGE[0]:.0f}～{cc.V_RANGE[1]:.0f}）",
        _level_range(gap_v, cc.V_RANGE, (45.0, 180.0)),
        "太暗：灯调亮或曝光加长；太亮：反过来",
    ))
    cavity_v = stats["cavity_v"]
    items.append(LightItem(
        "碳油亮度", f"{cavity_v:.0f}（门槛 {CAVITY_V_MIN:.0f}～{cc.V_RANGE[1]:.0f}）",
        _level_range(cavity_v, (CAVITY_V_MIN, cc.V_RANGE[1]), (CAVITY_V_MIN + 10.0, 180.0)),
        "太暗：灯调亮或曝光加长；太亮：反过来",
    ))
    # Copper glare spills onto the gap edge; seepage is only looked for in the mid-band, so only
    # glare there washes out what the check needs.
    gap_depth = cv2.distanceTransform(report.gap_mask, cv2.DIST_L2, 3)
    mid_band = gap & (gap_depth >= cc.CARBON_EDGE_CLEAR)
    gap_clip = float((val[mid_band] >= CLIP_VALUE).mean() * 100.0) if mid_band.any() else 0.0
    items.append(LightItem(
        "缝中间过曝", f"{gap_clip:.1f}% 顶到白",
        GOOD if gap_clip <= 0.5 else (LOW if gap_clip <= 2.0 else BAD),
        "缝被照白会洗掉颜色：灯调暗或曝光缩短",
    ))
    separation = stats["separation"]
    items.append(LightItem(
        "缝和碳油饱和度差", f"{separation:.0f}（门槛 {SEPARATION_MIN:.0f}）",
        _level_min(separation, SEPARATION_MIN, SEPARATION_MIN + 10.0),
        "渗墨和缺墨都靠它分开；过亮过暗都会变小，先把缝亮度调进范围",
    ))
    worst_sep = stats["segment_separation"]
    items.append(LightItem(
        "最差一段饱和度差", f"{worst_sep:.0f}（门槛 {SEGMENT_SEPARATION_MIN:.0f}）",
        _level_min(worst_sep, SEGMENT_SEPARATION_MIN, SEGMENT_SEPARATION_MIN + 10.0),
        "某一段颜色出不来：多半是那一头光不够",
    ))
    lift = stats["copper_lift"]
    items.append(LightItem(
        "铜比缝亮", f"{lift:.0f}（门槛 {COPPER_LIFT_MIN:.0f}）",
        _level_min(lift, COPPER_LIFT_MIN, COPPER_LIFT_MIN + 20.0),
        "铜不够亮框会套不准：灯调亮",
    ))
    worst_lift = stats["segment_lift"]
    items.append(LightItem(
        "最差一段铜比缝亮", f"{worst_lift:.0f}（门槛 {cc.MIN_SEGMENT_LIFT:.0f}）",
        _level_min(worst_lift, cc.MIN_SEGMENT_LIFT, 30.0),
        "某一段铜发暗：灯不均，或电离室在画面边上",
    ))
    cap = stats["cap_ratio"]
    items.append(LightItem(
        "两端亮度比", f"{cap:.0%}（门槛 {cc.MIN_CAP_RATIO:.0%}）",
        _level_min(cap, cc.MIN_CAP_RATIO, 0.65),
        "一头亮一头暗：灯不均，或电离室离画面中心太远",
    ))
    sharp = stats["sharpness"]
    items.append(LightItem(
        "清晰度", f"{sharp:.0f}（门槛 {cc.MIN_SHARPNESS:.0f}）",
        _level_min(sharp, cc.MIN_SHARPNESS, 300.0),
        "调对焦；光圈开太大景深会变浅",
    ))
    hue = float(np.median(hsv[:, :, 0][gap]))
    items.append(LightItem(
        "缝的颜色（色相）", f"{hue:.0f}（缺墨按 {cc.SEAM_HUE[0]}～{cc.SEAM_HUE[1]} 认）",
        _level_range(hue, (float(cc.SEAM_HUE[0]), float(cc.SEAM_HUE[1])), (28.0, 50.0)),
        "颜色偏了多半是白平衡变了：关掉自动白平衡",
    ))
    copper_clip = float((val[copper] >= CLIP_VALUE).mean() * 100.0)
    items.append(LightItem(
        "铜环过曝（参考）", f"{copper_clip:.0f}% 顶到白，好照片是 13～36%",
        GOOD if copper_clip <= 45.0 else LOW,
        "铜几乎全白时铜边会糊：灯稍调暗",
    ))

    if settings is not None:
        exposure = "?" if settings.exposure_us is None else f"{settings.exposure_us / 1000:.1f} ms"
        gain = "?" if settings.gain_db is None else f"{settings.gain_db:.1f} dB"
        white = settings.white_balance_auto or "?"
        items.append(LightItem(
            "相机", f"曝光 {exposure}，增益 {gain}，自动白平衡 {white}",
            BAD if settings.white_balance_auto == "连续自动" else GOOD,
            "自动白平衡会让颜色漂，在 MVS 里关掉",
        ))
    return items


def overall(items: list[LightItem]) -> str:
    worst = max((LEVEL_ORDER[item.level] for item in items), default=0)
    return ("光照合适", "能用，但余量小", "光照不合格")[worst]
