# -*- coding: utf-8 -*-
"""Run the hole-frame checker on one chamber crop and suggest a YOLO label.

The checker only knows the BeamCheck_160A18E10 layout. A suggestion is filled in when the
result maps cleanly onto 合格 / 渗墨 / 缺墨 / 看不清. A mismatched frame is left blank so the
operator decides, instead of teaching the model that a different chamber is 看不清.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from qc_host.vision import hole_frame as hf
from qc_host.vision.light_check import BAD, light_items, overall

PAD = 96
PHOTO_REASONS = ("亮度不合适", "对焦不清", "光照不均", "歪了")
# These mean the frame itself is wrong. A dark photo also fails the copper-brightness test, and that
# one should still become 看不清 rather than "please decide".
HARD_TOOL = ("没有完整框", "得分", "区域是空的", "尺寸不符")
PHOTO_LIGHT = {"缝亮度", "碳油亮度", "缝中间过曝", "清晰度", "两端亮度比"}


@dataclass
class Assist:
    title: str
    detail: str
    label: str | None
    overlay: np.ndarray


_templates: list[np.ndarray] | None = None
_lock = threading.Lock()


def _templates_cached() -> list[np.ndarray]:
    global _templates
    with _lock:
        if _templates is None:
            _templates = hf._load_templates()
        return _templates


def _locate(image: np.ndarray):
    match = hf.match_hole(image, _templates_cached())
    if match is None:
        return None
    x, y, direction, score = match
    x, y, radius = hf.refine_hole(image, x, y, direction)
    shift = hf.stem_offset(image, x, y, direction)
    report = hf.check_chamber(
        image, (x, y), direction, shift, radius, score, hf.EDGE_RADII, hf.HALF_STRAIGHT
    )
    return report, (x, y), direction, radius


def _paint(image: np.ndarray, report, hole: tuple[float, float], direction, radius: float) -> np.ndarray:
    shown = image.copy()
    for mask, tint in ((report.gap_mask, (0, 160, 0)), (report.cavity_mask, (160, 80, 0))):
        if mask is None or mask.shape[:2] != shown.shape[:2]:
            continue
        area = mask > 0
        if area.any():
            shown[area] = (shown[area] * 0.75 + np.array(tint) * 0.25).astype(np.uint8)
    if report.carbon_in_gap is not None and report.carbon_in_gap.shape[:2] == shown.shape[:2]:
        shown[report.carbon_in_gap > 0] = (0, 0, 255)
    if report.board_in_cavity is not None and report.board_in_cavity.shape[:2] == shown.shape[:2]:
        shown[report.board_in_cavity > 0] = (255, 0, 255)
    x, y = hole
    for radius_px, color in zip(hf.EDGE_RADII, hf.EDGE_COLORS):
        pts = np.round(np.asarray(hf.stadium_points((x, y), report.long_angle, radius_px))).astype(np.int32)
        cv2.polylines(shown, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    cv2.circle(shown, (int(round(x)), int(round(y))), max(2, int(round(radius))), hf.HOLE_COLOR, 2, cv2.LINE_AA)
    ux, uy = direction
    cv2.line(
        shown,
        (int(round(x)), int(round(y))),
        (int(round(x + ux * 28)), int(round(y + uy * 28))),
        hf.HOLE_COLOR,
        2,
        cv2.LINE_AA,
    )
    return shown


def _suggest(word: str, reasons: list[str], items) -> str | None:
    """Map a clean checker result onto one training class. Unclear results stay unlabeled."""
    if word == "渗墨":
        return "shenmo"
    if word == "缺墨":
        return "quemo"
    if word not in ("未发现问题", "无法判定"):
        return None
    hard = any(key in reason for reason in reasons for key in HARD_TOOL)
    photo = any(key in reason for reason in reasons for key in PHOTO_REASONS)
    photo_light = any(item.level == BAD and item.name in PHOTO_LIGHT for item in items)
    if hard:
        return None
    if word == "无法判定":
        return "kanbuqing" if photo or photo_light else None
    if photo_light:
        return "kanbuqing"
    return "hege"


def _detail(word: str, lines: list[str], items, light_word: str, direction) -> str:
    stem = hf.DIRECTION_NAMES.get(direction, "")
    parts = [f"判定：{word}" + (f"　　柄朝{stem}" if stem else "")]
    parts.extend(lines[:4])
    parts.append(f"光照：{light_word}")
    noted = [item for item in items if item.level != "余量足"]
    for item in noted[:6]:
        extra = f"　{item.hint}" if item.hint and item.level == BAD else ""
        parts.append(f"{item.name} {item.text}　{item.level}{extra}")
    if not noted:
        parts.append("光照各项都在合适范围。")
    return "\n".join(parts)


def _title(word: str, light_word: str, label: str | None) -> str:
    if label == "hege":
        return f"圆孔画框建议：合格。没发现渗墨或缺墨。光照：{light_word}。同意就点「采用圆孔结果」，不对按 1。"
    if label == "shenmo":
        return "圆孔画框建议：渗墨。红点是疑似进缝的黑，请你看准再点「采用」，不对按别的数字。"
    if label == "quemo":
        return "圆孔画框建议：缺墨。品红是内腔疑似露板，请你看准再点「采用」，不对按别的数字。"
    if label == "kanbuqing" and word == "未发现问题":
        return "圆孔画框：没检出渗墨或缺墨，但这颗光照不合格，建议标看不清。你要是看得清，按 1 标合格。"
    if label == "kanbuqing":
        return f"圆孔画框建议：看不清（{word}，{light_word}）。灯暗、糊了或一头没照到。同意就采用，看清了就按 1。"
    if word in ("渗墨、缺墨", "缺墨、渗墨"):
        return "圆孔画框：渗墨和缺墨都疑似有。请你自己按 2 或 3，这次不自动采用。"
    return f"圆孔画框：{word}，{light_word}。框可能没套上这一种电离室，请按 1 到 4 自己定。"


def analyse(image: np.ndarray) -> Assist:
    """Suggest a label and return the crop with the four edges drawn on it."""
    try:
        return _analyse(image)
    except Exception as exc:
        blank = image if image is not None else np.zeros((32, 32, 3), np.uint8)
        return Assist(title=f"圆孔画框出错：{exc}", detail="这张先按 1 到 4 自己标。", label=None, overlay=blank)


def _analyse(image: np.ndarray) -> Assist:
    found = _locate(image)
    if found is None:
        return Assist(
            title="圆孔画框没找到圆孔。请按 1 到 4 自己定。",
            detail="小图中间对不上圆孔。换一种尺寸的电离室时，圆孔画框不会认。",
            label=None,
            overlay=image,
        )
    report, hole, direction, radius = found
    canvas = image
    if any("没有完整框在选区里" in reason for reason in report.reasons):
        padded = cv2.copyMakeBorder(image, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        again = _locate(padded)
        if again is not None:
            report, hole, direction, radius = again
            canvas = padded
    word, lines = hf.verdict(report, hf.PX_PER_MM)
    items = light_items(canvas, report, hole, hf.EDGE_RADII, hf.HALF_STRAIGHT, None)
    light_word = overall(items)
    label = _suggest(word, report.reasons, items)
    overlay = _paint(canvas, report, hole, direction, radius)
    if canvas is not image:
        overlay = overlay[PAD:-PAD, PAD:-PAD]
    return Assist(
        title=_title(word, light_word, label),
        detail=_detail(word, lines, items, light_word, direction),
        label=label,
        overlay=overlay,
    )
