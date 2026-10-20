# -*- coding: utf-8 -*-
"""Find the center hole in a crop, then draw the chamber from the known stadium."""

from __future__ import annotations

import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from qc_host.vision.chamber_check import ChamberReport, check_chamber, frame_fit, verdict
from qc_host.vision.grab import CameraSettings, grab_frame
from qc_host.vision.lab import SOURCE
from qc_host.vision.light_check import BAD, GOOD, LOW, light_items, overall

# Allegro layout of BeamCheck_160A18E10: half-widths from the hole, in mm.
# inner copper inner, inner copper outer, outer copper inner, outer copper outer.
# The inner copper inner edge reads 0.698 mm off the layout screenshot, but on the photos the stem
# ends at about 0.63 mm and the stem joins that edge, so the photo value is used until the layout
# number is confirmed.
EDGE_RADII_MM = (0.63, 1.002, 1.303, 1.591)
HALF_STRAIGHT_MM = 4.008
# light_low.png: both gap edges on the straights and the cap position agree with the layout at this scale.
PX_PER_MM = 44.4
EDGE_RADII = tuple(r * PX_PER_MM for r in EDGE_RADII_MM)
HALF_STRAIGHT = HALF_STRAIGHT_MM * PX_PER_MM
EDGE_COLORS = (
    (255, 0, 255),
    (0, 140, 255),
    (0, 255, 255),
    (255, 220, 0),
)
HOLE_COLOR = (80, 220, 80)
# Stem down, hole center at (14, 14) in every patch. hole_template.png: dim hole, cut from
# light_low.png at (3257, 2021). hole_template_bright.png: shiny hole, mean of 11 horizontal
# chambers on board4.png. hole_template_side.png: mean of 11 vertical chambers on board4.png,
# which catch the light from the side and read grey.
TEMPLATE_PATHS = tuple(
    SOURCE.parent / name for name in ("hole_template.png", "hole_template_bright.png", "hole_template_side.png")
)
TEMPLATE_CENTER = (14, 14)
CANDIDATES_PER_ROTATION = 3
CANDIDATE_SPACING = 10
MIN_CANDIDATE_SCORE = 0.5
SAME_HOLE = 4.0
HOLE_RADIUS = 8.5
STEM_LENGTH = 30.0
DIRECTION_NAMES = {(0, 1): "下", (0, -1): "上", (1, 0): "右", (-1, 0): "左"}
GRAB_DIR = SOURCE.parent / "grabs"
LEVEL_COLORS = {GOOD: "#208020", LOW: "#b07000", BAD: "#c02020"}


def _load_templates() -> list[np.ndarray]:
    templates = []
    for path in TEMPLATE_PATHS:
        template = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(path)
        templates.append(template)
    return templates


def _rotated_templates(template: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
    """The template in four stem directions, each with where the hole center lands."""
    th, tw = template.shape[:2]
    cx, cy = TEMPLATE_CENTER
    return [
        (template, (cx, cy), (0, 1)),
        (cv2.rotate(template, cv2.ROTATE_180), (tw - 1 - cx, th - 1 - cy), (0, -1)),
        (cv2.rotate(template, cv2.ROTATE_90_COUNTERCLOCKWISE), (cy, tw - 1 - cx), (1, 0)),
        (cv2.rotate(template, cv2.ROTATE_90_CLOCKWISE), (th - 1 - cy, cx), (-1, 0)),
    ]


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    xs = np.clip(xs, 0, width - 1.001)
    ys = np.clip(ys, 0, height - 1.001)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def refine_hole(
    image: np.ndarray, x: float, y: float, direction: tuple[int, int]
) -> tuple[float, float, float]:
    """Fit the hole rim near the template hit. Returns center and radius to a quarter pixel.

    The rim is where the disk falls off to the cavity. The sector toward the stem is skipped so
    the stem does not pull the center.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    angles = np.deg2rad(np.arange(0, 360, 10))
    keep = np.cos(angles) * direction[0] + np.sin(angles) * direction[1] <= 0.8
    ux, uy = np.cos(angles[keep]), np.sin(angles[keep])
    shifts = np.arange(-3.0, 3.01, 0.25)
    grid_x, grid_y = np.meshgrid(shifts, shifts)
    cx = (x + grid_x).reshape(-1, 1)
    cy = (y + grid_y).reshape(-1, 1)
    best = (-np.inf, x, y, HOLE_RADIUS)
    for radius in np.arange(6.0, 10.01, 0.5):
        inside = _bilinear(blurred, cx + ux * (radius - 2.0), cy + uy * (radius - 2.0))
        outside = _bilinear(blurred, cx + ux * (radius + 2.0), cy + uy * (radius + 2.0))
        score = (inside - outside).mean(axis=1)
        index = int(np.argmax(score))
        if score[index] > best[0]:
            best = (float(score[index]), float(cx[index, 0]), float(cy[index, 0]), float(radius))
    return best[1], best[2], best[3]


def stem_offset(image: np.ndarray, x: float, y: float, direction: tuple[int, int]) -> float:
    """Sideways shift of the real stem from the hole center, in pixels. The stem is often 1-2 px off."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    ux, uy = direction
    vx, vy = -uy, ux
    offsets: list[float] = []
    for radius in range(12, 27):
        profile = np.array(
            [_sample(blurred, x + radius * ux + o * vx, y + radius * uy + o * vy) for o in range(-5, 6)]
        )
        peak = int(np.argmax(profile))
        shift = float(peak - 5)
        if 0 < peak < len(profile) - 1:
            left, mid, right = profile[peak - 1], profile[peak], profile[peak + 1]
            bend = left - 2.0 * mid + right
            if bend < 0:
                shift += 0.5 * (left - right) / bend
        offsets.append(shift)
    return float(np.median(offsets))


def stem_end(image: np.ndarray, x: float, y: float, direction: tuple[int, int], shift: float) -> float | None:
    """Distance from the hole to where the stem meets the inner copper inner edge.

    Beside the stem the cavity stays dark; at the inner edge it brightens. Returns None when the
    edge is too dim to see, then the model value applies.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    ux, uy = direction
    vx, vy = -uy, ux

    def beside(radius: int) -> float:
        cx = x + radius * ux + shift * vx
        cy = y + radius * uy + shift * vy
        return float(np.mean([_sample(blurred, cx + o * vx, cy + o * vy) for o in (-8, -7, -6, 6, 7, 8)]))

    base = float(np.median([beside(r) for r in range(10, 21)]))
    for radius in range(22, 35):
        if beside(radius) > base + 8.0:
            return radius - 0.5
    return None


def match_hole(
    image: np.ndarray, templates: list[np.ndarray]
) -> tuple[float, float, tuple[int, int], float] | None:
    """Hole center, stem direction and template score of the chamber hole in the box.

    The box is drawn around one chamber, so only the middle half of it is searched. Mounting
    holes and trace bends near the box edge score as high as a dim real hole. Inside, a copper
    bend can score as high as the hole too, so the few best hits are each tried as the chamber
    hole and the one whose frame lands best on the copper rings wins.
    """
    candidates = _hole_candidates(image, templates)
    if not candidates:
        return None
    strong = [c for c in candidates if c[3] >= MIN_CANDIDATE_SCORE]
    if not strong:
        return max(candidates, key=lambda c: c[3])
    smooth_v = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (3, 3), 0)[:, :, 2].astype(np.float32)
    best, best_fit = None, -np.inf
    for x, y, direction, score in strong:
        x, y, _radius = refine_hole(image, x, y, direction)
        _tilt, fit = frame_fit(smooth_v, (x, y), float(np.arctan2(direction[0], -direction[1])), EDGE_RADII, HALF_STRAIGHT)
        if fit > best_fit:
            best, best_fit = (x, y, direction, score), fit
    x, y, direction, score = best
    # Each template also hits the same hole; report the better of them.
    for cx, cy, other, other_score in candidates:
        if other == direction and abs(cx - x) <= SAME_HOLE and abs(cy - y) <= SAME_HOLE:
            score = max(score, other_score)
    return x, y, direction, score


def _hole_candidates(image: np.ndarray, templates: list[np.ndarray]) -> list[tuple[float, float, tuple[int, int], float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    lo_x, hi_x = width * 0.25, width * 0.75
    lo_y, hi_y = height * 0.25, height * 0.75
    out: list[tuple[float, float, tuple[int, int], float]] = []
    rotations = [r for template in templates for r in _rotated_templates(template)]
    for rotated, (ox, oy), direction in rotations:
        th, tw = rotated.shape[:2]
        if height < th or width < tw:
            continue
        result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
        rx = np.arange(result.shape[1]) + ox
        ry = np.arange(result.shape[0]) + oy
        inside = ((ry >= lo_y) & (ry <= hi_y))[:, None] & ((rx >= lo_x) & (rx <= hi_x))[None, :]
        if not inside.any():
            continue
        result[~inside] = -1.0
        for _ in range(CANDIDATES_PER_ROTATION):
            _low, score, _low_loc, (x, y) = cv2.minMaxLoc(result)
            if score <= -1.0:
                break
            out.append((float(x + ox), float(y + oy), direction, float(score)))
            cv2.circle(result, (x, y), CANDIDATE_SPACING, -1.0, -1)
    return out


def _sample(gray: np.ndarray, x: float, y: float) -> float:
    height, width = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return 0.0
    return float(gray[yi, xi])


def stadium_points(center: tuple[float, float], long_angle: float, radius: float) -> list[tuple[float, float]]:
    """One closed stadium edge around the hole, in crop pixels."""
    cx, cy = center
    along_x, along_y = float(np.cos(long_angle)), float(np.sin(long_angle))
    side_x, side_y = -along_y, along_x
    points: list[tuple[float, float]] = []
    for end, degrees in ((HALF_STRAIGHT, range(-90, 91, 3)), (-HALF_STRAIGHT, range(90, 271, 3))):
        for degree in degrees:
            angle = np.deg2rad(degree)
            local_x = end + radius * np.cos(angle)
            local_y = radius * np.sin(angle)
            points.append((cx + local_x * along_x + local_y * side_x, cy + local_x * along_y + local_y * side_y))
    return points


def _stadium(center: tuple[float, float], long_angle: float, radius: float) -> np.ndarray:
    points = np.round(np.asarray(stadium_points(center, long_angle, radius))).astype(np.int32)
    return points.reshape(-1, 1, 2)


def _short_axis(gray: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Direction toward the two copper rings. Both sides must be bright, the long axis dark."""
    best: tuple[float, float] | None = None
    # Chambers on this board sit upright or on their side.
    for degree in (0, 90):
        angle = np.deg2rad(degree)
        copper_sides: list[float] = []
        cavity_sides: list[float] = []
        for radius in (40.0, 59.0):
            for sign in (1.0, -1.0):
                copper_sides.append(
                    max(
                        _sample(
                            gray,
                            x + sign * (radius + delta) * np.cos(angle),
                            y + sign * (radius + delta) * np.sin(angle),
                        )
                        for delta in (-4.0, -2.0, 0.0, 2.0, 4.0)
                    )
                )
                cavity_sides.append(
                    _sample(
                        gray,
                        x + sign * radius * np.cos(angle + np.pi / 2),
                        y + sign * radius * np.sin(angle + np.pi / 2),
                    )
                )
        if min(copper_sides) < 140 or max(cavity_sides) > 110:
            continue
        # Halfway to the inner copper is still the cavity. A point on the copper stays bright there.
        cavity_gap = [
            _sample(gray, x + sign * 22.0 * np.cos(angle), y + sign * 22.0 * np.sin(angle))
            for sign in (1.0, -1.0)
        ]
        if max(cavity_gap) > 100:
            continue
        score = float(sum(copper_sides) - sum(cavity_sides))
        if best is None or score > best[0]:
            best = (score, angle)
    if best is None:
        return None
    return best[1], best[0]


def _hole_response(gray: np.ndarray) -> np.ndarray:
    """Bright disk of about 8 px sitting in a darker ring. The stem is allowed to break that ring."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    disk = np.zeros((21, 21), np.float32)
    ring = np.zeros((21, 21), np.float32)
    cv2.circle(disk, (10, 10), 4, 1.0, -1)
    cv2.circle(ring, (10, 10), 10, 1.0, -1)
    cv2.circle(ring, (10, 10), 6, 0.0, -1)
    disk /= float(disk.sum())
    ring /= float(ring.sum())
    return cv2.filter2D(blurred, cv2.CV_32F, disk) - cv2.filter2D(blurred, cv2.CV_32F, ring)


def _center_on_copper(gray: np.ndarray, x: float, y: float, long_angle: float) -> tuple[float, float]:
    """Move the hole onto the midpoint of the opposite copper peaks."""
    along_x, along_y = float(np.cos(long_angle)), float(np.sin(long_angle))
    side_x, side_y = -along_y, along_x

    def shift(axis_x: float, axis_y: float, radius: float) -> float:
        signed: list[float] = []
        for sign in (1.0, -1.0):
            best_value = -1.0
            best_radius = radius
            for delta in range(-10, 11):
                current = radius + delta
                value = _sample(
                    gray,
                    x + sign * current * axis_x,
                    y + sign * current * axis_y,
                )
                if value > best_value:
                    best_value = value
                    best_radius = current
            signed.append(sign * best_radius)
        return (signed[0] + signed[1]) / 2.0

    along = shift(along_x, along_y, HALF_STRAIGHT + 40.0)
    side = shift(side_x, side_y, 40.0)
    return x + along * along_x + side * side_x, y + along * along_y + side * side_y


def find_hole(image: np.ndarray) -> tuple[float, float, float] | None:
    """Return hole center (x, y) and the long-axis angle, in crop pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    response = _hole_response(gray)
    height, width = gray.shape[:2]
    # A hole needs the copper samples 63 px away, so stay inside that margin.
    margin = 70
    if height <= margin * 2 or width <= margin * 2:
        return None
    best: tuple[float, float, float, float] | None = None
    ys, xs = np.where(response[margin : height - margin, margin : width - margin] >= 8)
    for y, x in zip(ys + margin, xs + margin):
        value = float(response[y, x])
        if value < float(response[y - 1, x]) or value < float(response[y + 1, x]):
            continue
        if value < float(response[y, x - 1]) or value < float(response[y, x + 1]):
            continue
        center = float(gray[y, x])
        if center < 60 or center > 190:
            continue
        axis = _short_axis(gray, float(x), float(y))
        if axis is None:
            continue
        short_angle, contrast = axis
        score = value * contrast
        if best is None or score > best[0]:
            best = (score, float(x), float(y), short_angle + np.pi / 2)
    if best is None:
        return None
    return best[1], best[2], best[3]


def stem_mark(gray: np.ndarray, x: float, y: float, long_angle: float) -> tuple[float, float, str, int]:
    """The brighter short-axis tail is the stem. Returns its tip, a direction name, and length."""
    short = long_angle - np.pi / 2.0
    along_x, along_y = float(np.cos(long_angle)), float(np.sin(long_angle))

    def side_peak(sign: float) -> tuple[float, float, float]:
        best_value = -1.0
        best_x, best_y = x, y
        for radius in range(12, 28):
            base_x = x + sign * radius * np.cos(short)
            base_y = y + sign * radius * np.sin(short)
            for offset in range(-6, 7):
                px = base_x + offset * along_x
                py = base_y + offset * along_y
                value = _sample(gray, px, py)
                if value > best_value:
                    best_value = value
                    best_x, best_y = px, py
        return best_value, best_x, best_y

    plus = side_peak(1.0)
    minus = side_peak(-1.0)
    sign = 1.0 if plus[0] >= minus[0] else -1.0
    length = 16
    tip_x = x + sign * length * float(np.cos(short))
    tip_y = y + sign * length * float(np.sin(short))
    dx = tip_x - x
    dy = tip_y - y
    if abs(dx) >= abs(dy):
        name = "右" if dx > 0 else "左"
    else:
        name = "下" if dy > 0 else "上"
    return tip_x, tip_y, name, length


def draw_hole_stem(image: np.ndarray, hole: tuple[float, float, float]) -> tuple[np.ndarray, str, int]:
    shown = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    x, y, long_angle = hole
    tip_x, tip_y, name, length = stem_mark(gray, x, y, long_angle)
    cv2.line(
        shown,
        (int(round(x)), int(round(y))),
        (int(round(tip_x)), int(round(tip_y))),
        HOLE_COLOR,
        2,
        cv2.LINE_8,
    )
    cv2.circle(shown, (int(round(x)), int(round(y))), 4, HOLE_COLOR, 1, cv2.LINE_8)
    return shown, name, length


class HoleFrameApp:
    def __init__(self) -> None:
        data = np.fromfile(SOURCE, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(SOURCE)
        self.src = image
        self.templates = _load_templates()
        self.match: tuple[float, float, tuple[int, int], float] | None = None
        self.stem_shift = 0.0
        self.stem_length: float | None = None
        self.hole_radius = HOLE_RADIUS
        self.report: ChamberReport | None = None
        self.settings: CameraSettings | None = None
        self._grab_result: tuple[np.ndarray | None, CameraSettings | None, str] | None = None
        self._grabbing = False
        self.height, self.width = image.shape[:2]
        self.zoom: float | None = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag: tuple[float, float] | None = None
        self._pan_start: tuple[int, int, float, float] | None = None
        self.rect: tuple[int, int, int, int] | None = None
        self.crop: np.ndarray | None = None
        self.hole: tuple[float, float, float] | None = None
        self.crop_zoom = 2.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0
        self._crop_pan_start: tuple[int, int, float, float] | None = None
        self._detail = ""
        self._overview_photo: ImageTk.PhotoImage | None = None
        self._crop_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("圆孔画框")
        self.root.geometry("1280x900")
        self.status = tk.StringVar(value=f"原图 {self.width}×{self.height}    框住一颗电离室，要包住中间的圆孔")

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT)
        self.grab_button = ttk.Button(top, text="拍照并体检", command=self.grab)
        self.grab_button.pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(top, text="识别圆孔和柄", command=self.locate).pack(side=tk.RIGHT)
        self.show_frame = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="画四条铜边", variable=self.show_frame, command=self.paint_crop).pack(
            side=tk.RIGHT, padx=8
        )
        self.show_regions = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="显示检查区域", variable=self.show_regions, command=self.paint_crop).pack(
            side=tk.RIGHT, padx=8
        )

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        self.view = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0, cursor="crosshair")
        self.view.pack(fill=tk.BOTH, expand=True)
        self.view.bind("<Configure>", lambda _e: self.paint_view())
        self.view.bind("<ButtonPress-1>", self._rect_down)
        self.view.bind("<B1-Motion>", self._rect_move)
        self.view.bind("<ButtonRelease-1>", self._rect_up)
        self.view.bind("<ButtonPress-3>", self._pan_down)
        self.view.bind("<B3-Motion>", self._pan_move)
        self.view.bind("<MouseWheel>", self._view_wheel)

        self.crop_canvas = tk.Canvas(right, bg="#111111", highlightthickness=0)
        self.crop_canvas.pack(fill=tk.BOTH, expand=True)
        self.crop_canvas.bind("<Configure>", lambda _e: self.paint_crop())
        self.crop_canvas.bind("<MouseWheel>", self._crop_wheel)
        self.crop_canvas.bind("<ButtonPress-3>", self._crop_pan_down)
        self.crop_canvas.bind("<B3-Motion>", self._crop_pan_move)
        self.result_word = tk.StringVar(value="")
        self.result_word_label = tk.Label(right, textvariable=self.result_word, font=("Microsoft YaHei", 16, "bold"))
        self.result_word_label.pack(fill=tk.X, pady=(6, 0))
        self.result_detail = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.result_detail, wraplength=420, justify=tk.LEFT).pack(fill=tk.X)
        self.light_text = tk.Text(right, height=15, wrap=tk.WORD, font=("Microsoft YaHei", 9), relief=tk.FLAT, bg="#f4f4f4")
        self.light_text.pack(fill=tk.X, pady=(6, 0))
        self.light_text.tag_configure("title", font=("Microsoft YaHei", 11, "bold"))
        self.light_text.tag_configure("hint", foreground="#666666")
        for level, color in LEVEL_COLORS.items():
            self.light_text.tag_configure(level, foreground=color)
        self._show_light(None)
        ttk.Label(
            right,
            text="框住一颗电离室后点识别，用标准圆孔带柄模板在四个方向上匹配。"
            "得分 1 表示完全一样，0.5 以下多半不是。认到圆孔后按固定尺寸画四条铜边："
            "紫 内圈铜内边，橙 内圈铜外边，黄 外圈铜内边，青 外圈铜外边，都按设计尺寸画。"
            "橙和黄之间就是 0.3 mm 间隙。红色是缝里疑似炭黑（渗墨），品红是内腔疑似露板（缺墨）。"
            "勾「显示检查区域」可以看只查了哪些地方：绿色是缝，蓝色是内腔。"
            "滚轮按鼠标位置放大，右键拖动可以移动。",
            wraplength=420,
        ).pack(fill=tk.X, pady=(6, 0))

    def _fit_zoom(self) -> float:
        return min(max(1, self.view.winfo_width()) / self.width, max(1, self.view.winfo_height()) / self.height)

    def _zoom(self) -> float:
        return self._fit_zoom() if self.zoom is None else self.zoom

    def _origin(self) -> tuple[float, float]:
        zoom = self._zoom()
        cw = max(1, self.view.winfo_width())
        ch = max(1, self.view.winfo_height())
        return (cw - self.width * zoom) / 2.0 + self.pan_x, (ch - self.height * zoom) / 2.0 + self.pan_y

    def _to_src(self, x: int, y: int) -> tuple[float, float]:
        x0, y0 = self._origin()
        zoom = self._zoom()
        return (x - x0) / zoom, (y - y0) / zoom

    def paint_view(self) -> None:
        if self.view.winfo_width() < 2:
            return
        zoom = self._zoom()
        x0, y0 = self._origin()
        cw, ch = self.view.winfo_width(), self.view.winfo_height()
        sx0 = max(0.0, (0 - x0) / zoom)
        sy0 = max(0.0, (0 - y0) / zoom)
        sx1 = min(float(self.width), (cw - x0) / zoom)
        sy1 = min(float(self.height), (ch - y0) / zoom)
        self.view.delete("all")
        if sx1 <= sx0 or sy1 <= sy0:
            return
        fx0, fy0, fx1, fy1 = int(sx0), int(sy0), int(np.ceil(sx1)), int(np.ceil(sy1))
        crop = self.src[fy0:fy1, fx0:fx1]
        out_w = max(1, int(round(crop.shape[1] * zoom)))
        out_h = max(1, int(round(crop.shape[0] * zoom)))
        interp = cv2.INTER_NEAREST if zoom >= 1.0 else cv2.INTER_AREA
        shown = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        self._overview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.create_image(int(x0 + fx0 * zoom), int(y0 + fy0 * zoom), anchor=tk.NW, image=self._overview_photo)
        self.view.delete("rect")
        if self.rect is not None:
            a, b, c, d = self.rect
            self.view.create_rectangle(
                x0 + a * zoom, y0 + b * zoom, x0 + c * zoom, y0 + d * zoom, outline="#50d250", width=2, tags="rect"
            )

    def _rect_down(self, event: tk.Event) -> None:
        self._drag = self._to_src(event.x, event.y)

    def _rect_move(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        x0, y0 = self._drag
        x1, y1 = self._to_src(event.x, event.y)
        self.rect = self._clamp_rect(x0, y0, x1, y1)
        self.paint_view()

    def _rect_up(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        x0, y0 = self._drag
        x1, y1 = self._to_src(event.x, event.y)
        self._drag = None
        rect = self._clamp_rect(x0, y0, x1, y1)
        if rect[2] - rect[0] < 2 or rect[3] - rect[1] < 2:
            return
        self.rect = rect
        x, y, x2, y2 = rect
        self.crop = self.src[y:y2, x:x2].copy()
        self.hole = None
        self.match = None
        self.report = None
        self.result_word.set("")
        self.result_detail.set("")
        self._show_light(None)
        self.crop_zoom = 2.0
        self.crop_pan_x = 0.0
        self.crop_pan_y = 0.0
        self._detail = f"选区 ({x},{y})  {x2 - x}×{y2 - y}    点「识别圆孔和柄」"
        self._show_status()
        self.paint_crop()

    def _clamp_rect(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int]:
        a, c = sorted((int(round(x0)), int(round(x1))))
        b, d = sorted((int(round(y0)), int(round(y1))))
        a = min(max(0, a), self.width - 1)
        c = min(max(0, c), self.width)
        b = min(max(0, b), self.height - 1)
        d = min(max(0, d), self.height)
        return a, b, c, d

    def _pan_down(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _pan_move(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        x, y, px, py = self._pan_start
        self.pan_x = px + (event.x - x)
        self.pan_y = py + (event.y - y)
        if self.zoom is None:
            self.zoom = self._fit_zoom()
        self.paint_view()

    def _view_wheel(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        old = self._zoom()
        new = min(16.0, max(0.02, old * (1.15 if event.delta > 0 else 1 / 1.15)))
        x0, y0 = self._origin()
        src_x = (event.x - x0) / old
        src_y = (event.y - y0) / old
        self.zoom = new
        cw, ch = max(1, self.view.winfo_width()), max(1, self.view.winfo_height())
        self.pan_x = event.x - src_x * new - (cw - self.width * new) / 2.0
        self.pan_y = event.y - src_y * new - (ch - self.height * new) / 2.0
        self.paint_view()

    def _crop_origin(self) -> tuple[float, float]:
        assert self.crop is not None
        zoom = self.crop_zoom
        height, width = self.crop.shape[:2]
        cw = max(1, self.crop_canvas.winfo_width())
        ch = max(1, self.crop_canvas.winfo_height())
        return (cw - width * zoom) / 2.0 + self.crop_pan_x, (ch - height * zoom) / 2.0 + self.crop_pan_y

    def _center_crop(self, x: float, y: float) -> None:
        if self.crop is None:
            return
        height, width = self.crop.shape[:2]
        self.crop_pan_x = (width / 2.0 - x) * self.crop_zoom
        self.crop_pan_y = (height / 2.0 - y) * self.crop_zoom

    def _show_status(self) -> None:
        zoom = f"右侧 {self.crop_zoom:.1f} 倍"
        self.status.set(f"{self._detail}    {zoom}" if self._detail else zoom)

    def _crop_pan_down(self, event: tk.Event) -> None:
        self._crop_pan_start = (event.x, event.y, self.crop_pan_x, self.crop_pan_y)

    def _crop_pan_move(self, event: tk.Event) -> None:
        if self._crop_pan_start is None or self.crop is None:
            return
        x, y, px, py = self._crop_pan_start
        self.crop_pan_x = px + (event.x - x)
        self.crop_pan_y = py + (event.y - y)
        self.paint_crop()

    def _crop_wheel(self, event: tk.Event) -> None:
        if self.crop is None or event.delta == 0:
            return
        old = self.crop_zoom
        new = min(32.0, max(0.25, old * (1.15 if event.delta > 0 else 1 / 1.15)))
        x0, y0 = self._crop_origin()
        src_x = (event.x - x0) / old
        src_y = (event.y - y0) / old
        self.crop_zoom = new
        height, width = self.crop.shape[:2]
        cw = max(1, self.crop_canvas.winfo_width())
        ch = max(1, self.crop_canvas.winfo_height())
        self.crop_pan_x = event.x - src_x * new - (cw - width * new) / 2.0
        self.crop_pan_y = event.y - src_y * new - (ch - height * new) / 2.0
        self._show_status()
        self.paint_crop()

    def _shown_crop(self) -> np.ndarray:
        assert self.crop is not None
        if self.report is None:
            return self.crop
        shown = self.crop.copy()
        if self.show_regions.get():
            for mask, tint in ((self.report.gap_mask, (0, 160, 0)), (self.report.cavity_mask, (160, 80, 0))):
                area = mask > 0
                shown[area] = (shown[area] * 0.6 + np.array(tint) * 0.4).astype(np.uint8)
        shown[self.report.carbon_in_gap > 0] = (0, 0, 255)
        shown[self.report.board_in_cavity > 0] = (255, 0, 255)
        return shown

    def paint_crop(self) -> None:
        self.crop_canvas.delete("all")
        if self.crop is None or self.crop_canvas.winfo_width() < 2:
            self.crop_canvas.create_text(
                max(1, self.crop_canvas.winfo_width()) // 2,
                max(1, self.crop_canvas.winfo_height()) // 2,
                text="在左侧拖一个长方形，包住圆孔",
                fill="#dddddd",
                font=("Microsoft YaHei", 14),
            )
            return
        zoom = self.crop_zoom
        x0, y0 = self._crop_origin()
        cw, ch = self.crop_canvas.winfo_width(), self.crop_canvas.winfo_height()
        height, width = self.crop.shape[:2]
        sx0 = max(0.0, (0 - x0) / zoom)
        sy0 = max(0.0, (0 - y0) / zoom)
        sx1 = min(float(width), (cw - x0) / zoom)
        sy1 = min(float(height), (ch - y0) / zoom)
        if sx1 > sx0 and sy1 > sy0:
            fx0, fy0 = int(sx0), int(sy0)
            fx1, fy1 = int(np.ceil(sx1)), int(np.ceil(sy1))
            patch = self._shown_crop()[fy0:fy1, fx0:fx1]
            out_w = max(1, int(round(patch.shape[1] * zoom)))
            out_h = max(1, int(round(patch.shape[0] * zoom)))
            interp = cv2.INTER_NEAREST if zoom >= 1.0 else cv2.INTER_AREA
            resized = cv2.resize(patch, (out_w, out_h), interpolation=interp)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self._crop_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.crop_canvas.create_image(x0 + fx0 * zoom, y0 + fy0 * zoom, anchor=tk.NW, image=self._crop_photo)
        if self.match is None:
            return
        x, y, (dx, dy), _score = self.match
        long_angle = float(np.arctan2(dx, -dy))
        if self.report is not None:
            long_angle = self.report.long_angle
            dx, dy = float(np.sin(long_angle)), float(-np.cos(long_angle))
        cx, cy = x0 + x * zoom, y0 + y * zoom
        side_x = -dy * self.stem_shift
        side_y = dx * self.stem_shift
        length = EDGE_RADII[0]
        tx = x0 + (x + side_x + dx * length) * zoom
        ty = y0 + (y + side_y + dy * length) * zoom
        radius = self.hole_radius * zoom
        cx_edge = x0 + (x + side_x + dx * self.hole_radius) * zoom
        cy_edge = y0 + (y + side_y + dy * self.hole_radius) * zoom
        stroke = max(1, int(round(zoom * 0.6)))
        self.crop_canvas.create_line(cx_edge, cy_edge, tx, ty, fill="#50dc50", width=stroke, capstyle=tk.ROUND)
        self.crop_canvas.create_oval(
            cx - radius, cy - radius, cx + radius, cy + radius, outline="#50dc50", width=stroke
        )
        if not self.show_frame.get():
            return
        edge_width = max(1, int(round(zoom * 0.4)))
        for radius_px, (blue, green, red) in zip(EDGE_RADII, EDGE_COLORS):
            flat: list[float] = []
            for px, py in stadium_points((x, y), long_angle, radius_px):
                flat.extend((x0 + px * zoom, y0 + py * zoom))
            flat.extend(flat[:2])
            self.crop_canvas.create_line(*flat, fill=f"#{red:02x}{green:02x}{blue:02x}", width=edge_width)

    def locate(self) -> None:
        if self.crop is None:
            messagebox.showinfo("提示", "先拖一个长方形，框住一颗电离室，圆孔要在框里。")
            return
        self.match = match_hole(self.crop, self.templates)
        if self.match is None:
            self.report = None
            self._detail = "框比模板还小，放大一点再框。"
        else:
            x, y, direction, score = self.match
            x, y, self.hole_radius = refine_hole(self.crop, x, y, direction)
            self.match = (x, y, direction, score)
            self.stem_shift = stem_offset(self.crop, x, y, direction)
            span = 2.0 * (HALF_STRAIGHT + EDGE_RADII[-1]) + 40.0
            depth = 2.0 * EDGE_RADII[-1] + 40.0
            long_w, long_h = (depth, span) if direction[0] != 0 else (span, depth)
            cw = max(1, self.crop_canvas.winfo_width())
            ch = max(1, self.crop_canvas.winfo_height())
            self.crop_zoom = max(0.25, min(32.0, cw / long_w, ch / long_h))
            self._center_crop(x, y)
            name = DIRECTION_NAMES[direction]
            self.stem_length = stem_end(self.crop, x, y, direction, self.stem_shift)
            low_note = "" if score >= 0.5 else "，得分偏低，可能不是圆孔"
            if self.stem_length is None:
                stem_text = "柄末端看不清"
            else:
                stem_text = f"柄长 {self.stem_length:.1f}（设计内边 {EDGE_RADII[0]:.1f}）"
            self._detail = f"圆孔 ({x:.0f},{y:.0f})，柄朝{name}，得分 {score:.2f}{low_note}，{stem_text}，单位像素"
            self.report = check_chamber(
                self.crop, (x, y), direction, self.stem_shift, self.hole_radius, score, EDGE_RADII, HALF_STRAIGHT
            )
            self._detail += f"，框转了 {self.report.tilt:+.2f}°"
            word, lines = verdict(self.report, PX_PER_MM)
            color = {"无法判定": "#b07000", "未发现问题": "#208020"}.get(word, "#c02020")
            self.result_word.set(word)
            self.result_word_label.configure(fg=color)
            self.result_detail.set("\n".join(lines))
        if self.report is None or self.match is None:
            self._show_light(None)
        else:
            x, y, _direction, _score = self.match
            self._show_light(light_items(self.crop, self.report, (x, y), EDGE_RADII, HALF_STRAIGHT, self.settings))
        self._show_status()
        self.paint_crop()

    def _show_light(self, items) -> None:
        text = self.light_text
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        if items is None:
            text.insert(tk.END, "光照体检\n", "title")
            text.insert(tk.END, "框住一颗画面中间的电离室，点「识别圆孔和柄」或「拍照并体检」。\n", "hint")
            text.insert(tk.END, "拧灯或调曝光后点「拍照并体检」，会按同一个框重新体检。", "hint")
        else:
            word = overall(items)
            level = {"光照合适": GOOD, "能用，但余量小": LOW}.get(word, BAD)
            text.insert(tk.END, f"光照体检：{word}\n", ("title", level))
            for item in items:
                text.insert(tk.END, f"{item.level}  ", item.level)
                text.insert(tk.END, f"{item.name}  {item.text}\n")
                if item.level != GOOD and item.hint:
                    text.insert(tk.END, f"      {item.hint}\n", "hint")
        text.configure(state=tk.DISABLED)

    def grab(self) -> None:
        if self._grabbing:
            return
        self._grabbing = True
        self.grab_button.configure(state=tk.DISABLED)
        self._detail = "正在拍照（不改相机的曝光和增益，拍完就断开）…"
        self._show_status()

        def work() -> None:
            try:
                image, settings = grab_frame()
                self._grab_result = (image, settings, "")
            except Exception as error:  # noqa: BLE001 - shown to the operator
                self._grab_result = (None, None, str(error))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(200, self._poll_grab)

    def _poll_grab(self) -> None:
        if self._grab_result is None:
            self.root.after(200, self._poll_grab)
            return
        image, settings, error = self._grab_result
        self._grab_result = None
        self._grabbing = False
        self.grab_button.configure(state=tk.NORMAL)
        if image is None:
            self._detail = f"拍照失败：{error}"
            self._show_status()
            messagebox.showerror("拍照失败", error)
            return
        GRAB_DIR.mkdir(parents=True, exist_ok=True)
        path = GRAB_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.png"
        ok, encoded = cv2.imencode(".png", image)
        if ok:
            path.write_bytes(encoded.tobytes())
        self.src = image
        self.settings = settings
        self.height, self.width = image.shape[:2]
        self.root.title(f"圆孔画框  {path.name}")
        if self.rect is None:
            self._detail = f"已拍照 {path.name}，框住一颗电离室再点「识别圆孔和柄」"
            self._show_status()
            self.paint_view()
            return
        x, y, x2, y2 = self._clamp_rect(*self.rect)
        self.rect = (x, y, x2, y2)
        self.crop = self.src[y:y2, x:x2].copy()
        self.paint_view()
        self.locate()
        self._detail = f"{path.name}  {self._detail}"
        self._show_status()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    HoleFrameApp().run()


if __name__ == "__main__":
    main()
