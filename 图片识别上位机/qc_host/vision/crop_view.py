# -*- coding: utf-8 -*-
"""Drag a rectangle on the full frame and work on that original-pixel crop."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from qc_host.vision.lab import ROOT, SOURCE


# 外圈铜外沿、隔离层外沿、隔离层内沿、内圈铜内沿
EDGE_COLORS = (
    (255, 220, 0),
    (0, 255, 255),
    (0, 140, 255),
    (255, 0, 255),
)

RING_COLOR = (80, 220, 80)


def find_chamber_rings(crop: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """CLAHE then Otsu contours. Keep the long thin copper rings, drop specks and solid blobs."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rings: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 400:
            continue
        _x, _y, width, height = cv2.boundingRect(contour)
        short = max(1, min(width, height))
        if max(width, height) / short < 4.0:
            continue
        rings.append((area, contour))
    if not rings:
        return []
    largest = max(area for area, _contour in rings)
    return [
        (contour, RING_COLOR)
        for area, contour in rings
        if area >= largest * 0.25
    ]


def _median_indexes(values: list[float | None], window: int = 7, tol: float = 4.0) -> list[float | None]:
    """Drop a knee that jumps off the neighboring ones, then fill gaps of a few stations."""
    count = len(values)
    raw = np.array([np.nan if value is None else value for value in values], np.float64)
    half = window // 2
    kept = raw.copy()
    for index in range(count):
        neighbors = [
            raw[other]
            for other in range(max(0, index - half), min(count, index + half + 1))
            if np.isfinite(raw[other])
        ]
        if not np.isfinite(raw[index]) or len(neighbors) < 3:
            continue
        med = float(np.median(neighbors))
        kept[index] = med if abs(float(raw[index]) - med) <= tol else np.nan
    filled = kept.copy()
    for index in range(count):
        if np.isfinite(filled[index]):
            continue
        neighbors = []
        for offset in range(1, 5):
            for other in (index - offset, index + offset):
                if 0 <= other < count and np.isfinite(kept[other]):
                    neighbors.append(float(kept[other]))
        if len(neighbors) >= 2:
            filled[index] = float(np.median(neighbors))
    return [None if not np.isfinite(value) else float(value) for value in filled]


def _chains(
    points: list[tuple[float, float] | None], max_gap: float = 16.0, min_len: int = 5
) -> list[np.ndarray]:
    """Connect neighboring hits. A missed ray breaks the line instead of inventing it."""
    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for point in points:
        if point is None:
            if len(current) >= min_len:
                runs.append(current)
            current = []
            continue
        if current:
            dist = float(np.hypot(point[0] - current[-1][0], point[1] - current[-1][1]))
            if dist > max_gap:
                if len(current) >= min_len:
                    runs.append(current)
                current = [point]
                continue
        current.append(point)
    if len(current) >= min_len:
        runs.append(current)
    polylines: list[np.ndarray] = []
    for run in runs:
        arr = np.asarray(run, np.float64)
        if len(arr) >= 5:
            smoothed = arr.copy()
            for col in (0, 1):
                pad = np.pad(arr[:, col], 2, mode="edge")
                smoothed[:, col] = [float(np.median(pad[i : i + 5])) for i in range(len(arr))]
            arr = smoothed
        polylines.append(np.round(arr).astype(np.int32).reshape(-1, 1, 2))
    return polylines


def _bright_peaks(smooth: np.ndarray) -> list[int]:
    peaks: list[int] = []
    for i in range(3, len(smooth) - 3):
        if smooth[i] < smooth[i - 1] or smooth[i] < smooth[i + 1]:
            continue
        if smooth[i] < 140:
            continue
        left = smooth[max(0, i - 8) : i]
        left = left[left >= 15]
        if len(left) == 0 or smooth[i] < float(np.min(left)) + 40:
            continue
        if peaks and i - peaks[-1] < 6:
            if smooth[i] > smooth[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def _profile_edges(samples: np.ndarray) -> tuple[int, int, int, int] | None:
    """Indexes of the four gray knees, from outside the ring toward the cavity.

    A missing knee is -1. The dark isolation gap is the valley between the two
    copper peaks; carbon on the copper does not add another knee.
    """
    if len(samples) < 24:
        return None
    smooth = cv2.GaussianBlur(samples.reshape(-1, 1), (1, 0), 1.1).ravel()
    peaks = _bright_peaks(smooth)
    if not peaks:
        return None
    first = peaks[0]
    second = -1
    valley = -1
    for nxt in peaks[1:]:
        if nxt - first > 28:
            break
        dip = first + int(np.argmin(smooth[first : nxt + 1]))
        if float(smooth[first]) - float(smooth[dip]) < 40:
            continue
        if float(smooth[nxt]) - float(smooth[dip]) < 40:
            continue
        second = nxt
        valley = dip
        break
    if valley < 0:
        end = min(len(smooth) - 1, first + 28)
        if end <= first + 6:
            return None
        valley = first + 4 + int(np.argmin(smooth[first + 4 : end + 1]))
        if smooth[valley] > 100 or float(smooth[first]) - float(smooth[valley]) < 40:
            return None
    floor = float(smooth[valley])

    gap_outer = valley
    for i in range(valley, first, -1):
        if smooth[i] <= floor + 12:
            gap_outer = i
        elif smooth[i] >= floor + 30:
            break

    outer = first
    while outer > 1 and first - outer < 8 and abs(float(smooth[outer - 1]) - float(smooth[outer])) <= 4:
        outer -= 1
    while (
        outer > 1
        and first - outer < 18
        and smooth[outer - 1] >= 20
        and smooth[outer - 1] < smooth[outer] - 3
    ):
        outer -= 1
    if first - outer < 2:
        outer = -1

    gap_inner = -1
    inner = -1
    if second >= 0:
        gap_inner = valley
        for i in range(valley, second):
            if smooth[i] <= floor + 12:
                gap_inner = i
            ahead = float(np.max(smooth[i : min(second, i + 4)]))
            if ahead >= floor + 30 and smooth[i] >= floor + 8:
                gap_inner = max(valley, i - 1)
                break
        inner = second
        limit = min(len(smooth) - 2, second + 16)
        while inner < limit and second - inner > -8 and abs(float(smooth[inner + 1]) - float(smooth[inner])) <= 4:
            inner += 1
        cavity = float(np.min(smooth[second : limit + 1]))
        while inner < limit and smooth[inner] > cavity + 14 and smooth[inner + 1] < smooth[inner] - 3:
            inner += 1
    return outer, gap_outer, gap_inner, inner


def find_four_edges(
    bgr: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int, int]]] | None:
    """Trace the four boundaries by walking inward across the gray steps."""
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 3)
    height, width = gray.shape[:2]
    sigma = max(2.0, min(height, width) * 0.035)
    local = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    mask = ((gray.astype(np.float32) - local > 7.0) & (gray > 12)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) < 80 or float(mask.mean()) < 0.04:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    rel = pts - mean
    _eig, evec = np.linalg.eigh(np.cov(rel.T))
    axis = evec[:, int(np.argmax(_eig))]
    axis = axis / (float(np.hypot(axis[0], axis[1])) + 1e-9)
    perp = np.array([-axis[1], axis[0]], dtype=np.float64)
    u = rel @ axis
    v = rel @ perp
    u0, u1 = np.percentile(u, [4.0, 96.0])
    v0, v1 = np.percentile(v, [6.0, 94.0])
    span = float(u1 - u0)
    thick = float(v1 - v0)
    if span < 0.55 * width or thick < 8.0 or span < 2.2 * thick:
        return None
    u_c = 0.5 * (float(u0) + float(u1))
    v_c = 0.5 * (float(v0) + float(v1))
    radius = 0.5 * thick
    half_straight = max(0.0, 0.5 * span - radius)
    center = mean + u_c * axis + v_c * perp
    margin = max(8.0, 0.1 * thick)
    walk = int(max(28, radius + margin + 6))
    rays: list[tuple[np.ndarray, np.ndarray]] = []
    # The bright-core stadium runs into the rounded end. Stop the straight
    # before that bend so a vertical ray cannot jump onto the other side.
    straight_half = max(0.0, half_straight - max(12.0, 0.4 * radius))
    straight_count = max(20, int(round(straight_half)))
    for sign in (1.0, -1.0):
        for index in range(straight_count):
            along = -straight_half + 2.0 * straight_half * index / max(1, straight_count - 1)
            origin = center + axis * along + perp * sign * (radius + margin)
            rays.append((origin, -perp * sign))
    grouped: list[list[list[float | None]]] = []
    grouped_rays: list[list[tuple[np.ndarray, np.ndarray]]] = []
    cursor = 0
    for count in (straight_count, straight_count):
        grouped_rays.append(rays[cursor : cursor + count])
        grouped.append([[] for _ in range(4)])
        cursor += count
    for group, group_rays in zip(grouped, grouped_rays):
        for origin, direction in group_rays:
            samples = np.empty(walk, np.float32)
            for step in range(walk):
                x = float(origin[0]) + float(direction[0]) * step
                y = float(origin[1]) + float(direction[1]) * step
                xi, yi = int(round(x)), int(round(y))
                if xi < 0 or yi < 0 or xi >= width or yi >= height:
                    samples[step] = 0.0
                else:
                    samples[step] = float(gray[yi, xi])
            edges = _profile_edges(samples) or (-1, -1, -1, -1)
            for bucket, edge in zip(group, edges):
                bucket.append(None if edge < 2 or edge >= walk - 1 else float(edge))
    drawn: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    placed: list[list[list[tuple[float, float] | None]]] = []
    for group, group_rays in zip(grouped, grouped_rays):
        edge_points: list[list[tuple[float, float] | None]] = []
        for edge_index in range(4):
            smoothed = _median_indexes(group[edge_index])
            points: list[tuple[float, float] | None] = []
            for (origin, direction), distance in zip(group_rays, smoothed):
                if distance is None:
                    points.append(None)
                    continue
                x = float(origin[0]) + float(direction[0]) * distance
                y = float(origin[1]) + float(direction[1]) * distance
                if x < 2 or y < 2 or x >= width - 2 or y >= height - 2:
                    points.append(None)
                else:
                    points.append((x, y))
            edge_points.append(points)
        placed.append(edge_points)
    for edge_index, color in enumerate(EDGE_COLORS):
        for edge_points in placed:
            drawn.extend(
                (cnt, color) for cnt in _chains(edge_points[edge_index], max_gap=8.0, min_len=8)
            )
    if not drawn:
        return None
    return drawn


def _stadium_template() -> np.ndarray:
    radius = 20
    length = 90
    points: list[list[int]] = []
    for degree in range(90, 270, 8):
        angle = np.deg2rad(degree)
        points.append([int(radius + radius * np.cos(angle)), int(radius + radius * np.sin(angle))])
    for degree in range(-90, 90, 8):
        angle = np.deg2rad(degree)
        points.append(
            [int(length - radius + radius * np.cos(angle)), int(radius + radius * np.sin(angle))]
        )
    return np.asarray(points, np.int32).reshape(-1, 1, 2)


def _odd(value: int) -> int:
    k = max(1, int(value))
    return k if k % 2 else k + 1


class CropBench:
    def __init__(self) -> None:
        data = np.fromfile(SOURCE, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(SOURCE)
        self.src = image
        self.height, self.width = image.shape[:2]
        self.zoom: float | None = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag: tuple[float, float] | None = None
        self._pan_start: tuple[int, int, float, float] | None = None
        self.rect: tuple[int, int, int, int] | None = None
        self.crop: np.ndarray | None = None
        self.found: list[np.ndarray] = []
        self._contour_binary: np.ndarray | None = None
        self.crop_zoom = 4.0
        self._overview_photo: ImageTk.PhotoImage | None = None
        self._crop_photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("框选原图")
        self.root.geometry("1280x920")
        self.status = tk.StringVar(value=f"原图 {self.width}×{self.height}    在左侧按住左键拖一个长方形")

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Button(top, text="验证识别", command=self.verify).pack(side=tk.RIGHT)
        ttk.Button(top, text="保存这块原图", command=self.save_crop).pack(side=tk.RIGHT, padx=(0, 8))

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

        self.crop_canvas = tk.Canvas(right, bg="#111111", highlightthickness=0, height=460)
        self.crop_canvas.pack(fill=tk.BOTH, expand=True)
        self.crop_canvas.bind("<Configure>", lambda _e: self.paint_crop())
        self.crop_canvas.bind("<MouseWheel>", self._crop_wheel)

        book = ttk.Notebook(right)
        book.pack(fill=tk.X, pady=(8, 0))
        filters = ttk.Frame(book, padding=8)
        color_tab = ttk.Frame(book, padding=8)
        shape_tab = ttk.Frame(book, padding=8)
        book.add(filters, text="预处理")
        book.add(color_tab, text="颜色分割")
        book.add(shape_tab, text="轮廓")

        self.use_gray = tk.BooleanVar(value=False)
        self.use_clahe = tk.BooleanVar(value=False)
        self.median = tk.IntVar(value=0)
        self.gauss = tk.IntVar(value=0)
        self.thresh = tk.IntVar(value=0)
        self.morph = tk.StringVar(value="无")
        self.morph_k = tk.IntVar(value=3)
        self.color_target = tk.StringVar(value="无")
        self.bit_op = tk.StringVar(value="无")
        self.h_lo = tk.IntVar(value=8)
        self.h_hi = tk.IntVar(value=36)
        self.s_lo = tk.IntVar(value=25)
        self.v_lo = tk.IntVar(value=110)
        self.v_hi = tk.IntVar(value=255)
        self.find_contour = tk.BooleanVar(value=False)
        self.draw_contour = tk.BooleanVar(value=False)
        self.draw_rect = tk.BooleanVar(value=False)
        self.draw_min_rect = tk.BooleanVar(value=False)
        self.draw_ellipse = tk.BooleanVar(value=False)
        self.draw_poly = tk.BooleanVar(value=False)
        self.draw_hull = tk.BooleanVar(value=False)
        self.draw_match = tk.BooleanVar(value=False)
        self.min_area = tk.IntVar(value=80)
        self.poly_eps = tk.IntVar(value=2)

        row1 = ttk.Frame(filters)
        row1.pack(fill=tk.X)
        ttk.Checkbutton(row1, text="灰度", variable=self.use_gray, command=self.paint_crop).pack(side=tk.LEFT)
        ttk.Checkbutton(row1, text="CLAHE", variable=self.use_clahe, command=self.paint_crop).pack(side=tk.LEFT, padx=8)
        ttk.Label(row1, text="形态").pack(side=tk.LEFT, padx=(8, 2))
        morph = ttk.Combobox(row1, textvariable=self.morph, width=6, state="readonly", values=("无", "开运算", "闭运算"))
        morph.pack(side=tk.LEFT)
        morph.bind("<<ComboboxSelected>>", lambda _e: self.paint_crop())

        self._slider(filters, "中值", self.median, 0, 9)
        self._slider(filters, "高斯", self.gauss, 0, 15)
        self._slider(filters, "阈值", self.thresh, 0, 255)
        self._slider(filters, "形态核", self.morph_k, 1, 9)

        row_c = ttk.Frame(color_tab)
        row_c.pack(fill=tk.X)
        ttk.Label(row_c, text="目标").pack(side=tk.LEFT)
        target = ttk.Combobox(
            row_c, textvariable=self.color_target, width=6, state="readonly", values=("无", "金色", "间隙")
        )
        target.pack(side=tk.LEFT, padx=(4, 0))
        target.bind("<<ComboboxSelected>>", self._set_color_target)
        ttk.Label(row_c, text="位运算").pack(side=tk.LEFT, padx=(12, 2))
        bits = ttk.Combobox(row_c, textvariable=self.bit_op, width=4, state="readonly", values=("无", "与", "非", "或"))
        bits.pack(side=tk.LEFT)
        bits.bind("<<ComboboxSelected>>", lambda _e: self.paint_crop())
        self._slider(color_tab, "H 低", self.h_lo, 0, 180)
        self._slider(color_tab, "H 高", self.h_hi, 0, 180)
        self._slider(color_tab, "S 低", self.s_lo, 0, 255)
        self._slider(color_tab, "V 低", self.v_lo, 0, 255)
        self._slider(color_tab, "V 高", self.v_hi, 0, 255)

        row_s = ttk.Frame(shape_tab)
        row_s.pack(fill=tk.X)
        row_s2 = ttk.Frame(shape_tab)
        row_s2.pack(fill=tk.X)
        for row, items in (
            (row_s, (("找轮廓", self.find_contour), ("画轮廓", self.draw_contour), ("外接矩形", self.draw_rect), ("最小矩形", self.draw_min_rect))),
            (row_s2, (("椭圆", self.draw_ellipse), ("多边形", self.draw_poly), ("凸包", self.draw_hull), ("形状匹配", self.draw_match))),
        ):
            for text, var in items:
                ttk.Checkbutton(row, text=text, variable=var, command=self.paint_crop).pack(side=tk.LEFT, padx=(0, 8))
        self._slider(shape_tab, "最小面积", self.min_area, 0, 4000)
        self._slider(shape_tab, "逼近%", self.poly_eps, 1, 20)
        self.shape_info = tk.StringVar(value="勾上「找轮廓」后，这里显示面积、周长和形状匹配。")
        ttk.Label(shape_tab, textvariable=self.shape_info, wraplength=640).pack(fill=tk.X, pady=(4, 0))

        hint = ttk.Label(
            right,
            text="左键拖框。滚轮放大左侧便于对准，右键拖动。右侧滚轮放大这块原图，放大用最近邻，不造新像素。阈值 0 表示关闭。",
            wraplength=420,
        )
        hint.pack(fill=tk.X, pady=(6, 0))

    def _slider(self, parent: ttk.Frame, label: str, var: tk.IntVar, lo: int, hi: int) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=8).pack(side=tk.LEFT)
        ttk.Scale(
            row,
            from_=lo,
            to=hi,
            variable=var,
            command=lambda _v: self.paint_crop(),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _fit_zoom(self) -> float:
        cw = max(1, self.view.winfo_width())
        ch = max(1, self.view.winfo_height())
        return min(cw / self.width, ch / self.height)

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
        cw = self.view.winfo_width()
        ch = self.view.winfo_height()
        sx0 = max(0.0, (0 - x0) / zoom)
        sy0 = max(0.0, (0 - y0) / zoom)
        sx1 = min(float(self.width), (cw - x0) / zoom)
        sy1 = min(float(self.height), (ch - y0) / zoom)
        self.view.delete("all")
        if sx1 <= sx0 or sy1 <= sy0:
            return
        fx0, fy0 = int(sx0), int(sy0)
        fx1, fy1 = int(np.ceil(sx1)), int(np.ceil(sy1))
        crop = self.src[fy0:fy1, fx0:fx1]
        out_w = max(1, int(round(crop.shape[1] * zoom)))
        out_h = max(1, int(round(crop.shape[0] * zoom)))
        interp = cv2.INTER_NEAREST if zoom >= 1.0 else cv2.INTER_AREA
        shown = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        self._overview_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.create_image(int(x0 + fx0 * zoom), int(y0 + fy0 * zoom), anchor=tk.NW, image=self._overview_photo)
        self._draw_rect()

    def _draw_rect(self) -> None:
        self.view.delete("rect")
        if self.rect is None:
            return
        x0, y0 = self._origin()
        zoom = self._zoom()
        a, b, c, d = self.rect
        self.view.create_rectangle(
            x0 + a * zoom,
            y0 + b * zoom,
            x0 + c * zoom,
            y0 + d * zoom,
            outline="#50d250",
            width=2,
            tags="rect",
        )

    def _rect_down(self, event: tk.Event) -> None:
        self._drag = self._to_src(event.x, event.y)

    def _rect_move(self, event: tk.Event) -> None:
        if self._drag is None:
            return
        x0, y0 = self._drag
        x1, y1 = self._to_src(event.x, event.y)
        self.rect = self._clamp_rect(x0, y0, x1, y1)
        self._draw_rect()

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
        self.found = []
        self.crop_zoom = 4.0
        self._set_status()
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
        cw = max(1, self.view.winfo_width())
        ch = max(1, self.view.winfo_height())
        self.pan_x = event.x - src_x * new - (cw - self.width * new) / 2.0
        self.pan_y = event.y - src_y * new - (ch - self.height * new) / 2.0
        self.paint_view()

    def _crop_wheel(self, event: tk.Event) -> None:
        if self.crop is None or event.delta == 0:
            return
        self.crop_zoom = min(32.0, max(0.25, self.crop_zoom * (1.15 if event.delta > 0 else 1 / 1.15)))
        self._set_status()
        self.paint_crop()

    def _apply(self, image: np.ndarray) -> np.ndarray:
        out = image
        if self.use_gray.get() or self.use_clahe.get() or int(self.thresh.get()) > 0:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
            if self.use_clahe.get():
                gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            out = gray if self.use_gray.get() or int(self.thresh.get()) > 0 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        k = _odd(int(round(float(self.median.get()))))
        if k >= 3:
            out = cv2.medianBlur(out, k)
        g = _odd(int(round(float(self.gauss.get()))))
        if g >= 3:
            out = cv2.GaussianBlur(out, (g, g), 0)
        level = int(round(float(self.thresh.get())))
        if level > 0:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
            _t, out = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
        name = self.morph.get()
        if name != "无":
            mk = _odd(int(round(float(self.morph_k.get()))))
            mk = max(3, mk)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            op = cv2.MORPH_OPEN if name == "开运算" else cv2.MORPH_CLOSE
            out = cv2.morphologyEx(out, op, kernel)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        self._contour_binary = None
        if self.color_target.get() != "无":
            mask = self._gold_mask(out)
            op = self.bit_op.get()
            if op == "非":
                mask = cv2.bitwise_not(mask)
                op = "与"
            self._contour_binary = mask
            out = self._paint_gold(out, mask, op)
        return out

    def _gold_mask(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lo_h = min(int(self.h_lo.get()), int(self.h_hi.get()))
        hi_h = max(int(self.h_lo.get()), int(self.h_hi.get()))
        lo_v = min(int(self.v_lo.get()), int(self.v_hi.get()))
        hi_v = max(int(self.v_lo.get()), int(self.v_hi.get()))
        lower = np.array([lo_h, int(self.s_lo.get()), lo_v], np.uint8)
        upper = np.array([hi_h, 255, hi_v], np.uint8)
        return cv2.inRange(hsv, lower, upper)

    def _set_color_target(self, _event: object = None) -> None:
        name = self.color_target.get()
        if name == "金色":
            self.h_lo.set(8)
            self.h_hi.set(36)
            self.s_lo.set(25)
            self.v_lo.set(110)
            self.v_hi.set(255)
        elif name == "间隙":
            self.h_lo.set(28)
            self.h_hi.set(52)
            self.s_lo.set(40)
            self.v_lo.set(40)
            self.v_hi.set(110)
        self.paint_crop()

    def _paint_gold(self, image: np.ndarray, mask: np.ndarray, op: str) -> np.ndarray:
        if op == "无":
            tint = np.zeros_like(image)
            tint[:] = (0, 180, 0) if self.color_target.get() == "间隙" else (0, 220, 255)
            blended = cv2.addWeighted(image, 0.55, tint, 0.45, 0)
            painted = image.copy()
            painted[mask > 0] = blended[mask > 0]
            return painted
        if op == "与":
            return cv2.bitwise_and(image, image, mask=mask)
        return cv2.bitwise_or(image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    def _shape_overlay(self, image: np.ndarray) -> np.ndarray:
        if not self.find_contour.get() or self.crop is None:
            return image
        if self._contour_binary is not None:
            binary = self._contour_binary
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(np.unique(gray)) <= 4:
                _level, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            else:
                _level, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.min_area.get())
        kept = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if not kept:
            self.shape_info.set("没有面积够大的轮廓。把最小面积调低，或先做阈值 / 抠金色。")
            return image
        kept.sort(key=cv2.contourArea, reverse=True)
        largest = kept[0]
        area = cv2.contourArea(largest)
        length = cv2.arcLength(largest, True)
        text = f"轮廓 {len(kept)} 个，最大面积 {area:.0f}，周长 {length:.0f}"
        shown = image.copy()
        if self.draw_contour.get():
            cv2.drawContours(shown, kept, -1, (80, 220, 80), 1, cv2.LINE_8)
        if self.draw_rect.get():
            for cnt in kept[:8]:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(shown, (x, y), (x + w, y + h), (255, 220, 0), 1, cv2.LINE_8)
        if self.draw_min_rect.get():
            for cnt in kept[:8]:
                box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.int32)
                cv2.polylines(shown, [box], True, (0, 255, 255), 1, cv2.LINE_8)
        if self.draw_ellipse.get():
            for cnt in kept[:8]:
                if len(cnt) < 5:
                    continue
                try:
                    cv2.ellipse(shown, cv2.fitEllipse(cnt), (0, 140, 255), 1, cv2.LINE_8)
                except cv2.error:
                    continue
        if self.draw_poly.get():
            ratio = max(1, int(self.poly_eps.get())) / 100.0
            for cnt in kept[:8]:
                poly = cv2.approxPolyDP(cnt, ratio * cv2.arcLength(cnt, True), True)
                cv2.polylines(shown, [poly], True, (255, 0, 255), 1, cv2.LINE_8)
        if self.draw_hull.get():
            hulls = [cv2.convexHull(cnt) for cnt in kept[:8]]
            cv2.drawContours(shown, hulls, -1, (255, 180, 60), 1, cv2.LINE_8)
        if self.draw_match.get():
            template = _stadium_template()
            best_cnt = None
            best_score = None
            for cnt in kept:
                if len(cnt) < 5:
                    continue
                score = cv2.matchShapes(template, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_cnt = cnt
            if best_cnt is not None and best_score is not None:
                cv2.drawContours(shown, [best_cnt], -1, (255, 255, 255), 1, cv2.LINE_8)
                text += f"，最像跑道的匹配值 {best_score:.3f}（越小越像）"
        self.shape_info.set(text)
        return shown

    def paint_crop(self) -> None:
        self.crop_canvas.delete("all")
        if self.crop is None or self.crop_canvas.winfo_width() < 2:
            self.crop_canvas.create_text(
                max(1, self.crop_canvas.winfo_width()) // 2,
                max(1, self.crop_canvas.winfo_height()) // 2,
                text="在左侧拖一个长方形",
                fill="#dddddd",
                font=("Microsoft YaHei", 14),
            )
            return
        shown = self._shape_overlay(self._apply(self.crop))
        if self.found:
            shown = shown.copy()
            for cnt, color in self.found:
                cv2.polylines(shown, [cnt], True, color, 1, cv2.LINE_8)
        zoom = self.crop_zoom
        out_w = max(1, int(round(shown.shape[1] * zoom)))
        out_h = max(1, int(round(shown.shape[0] * zoom)))
        interp = cv2.INTER_NEAREST if zoom >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(shown, (out_w, out_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._crop_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.crop_canvas.create_image(8, 8, anchor=tk.NW, image=self._crop_photo)

    def _set_status(self) -> None:
        if self.crop is None or self.rect is None:
            return
        x, y, x2, y2 = self.rect
        kind = "最近邻放大" if self.crop_zoom >= 1.0 else "缩小显示，数据仍是原图"
        self.status.set(
            f"原图 {self.width}×{self.height}    "
            f"选区 ({x},{y})  {x2 - x}×{y2 - y} 原图像素    "
            f"右侧 {self.crop_zoom:.1f} 倍，{kind}"
        )

    def verify(self) -> None:
        if self.crop is None:
            messagebox.showinfo("提示", "先拖一个长方形，框住一颗电离室。")
            return
        rings = find_chamber_rings(self.crop)
        self.found = rings
        if not rings:
            self.status.set(
                f"这块原图 {self.crop.shape[1]}×{self.crop.shape[0]}："
                "CLAHE 之后没有数到细长的铜环。框要包住整颗电离室的亮铜。"
            )
        else:
            largest = max(int(cv2.contourArea(cnt)) for cnt, _color in rings)
            self.status.set(
                f"这块原图 {self.crop.shape[1]}×{self.crop.shape[0]}："
                f"CLAHE 后找到 {len(rings)} 条铜环轮廓，最大面积 {largest}。"
                "绿线画在铜的亮边上。"
            )
        self.paint_crop()

    def save_crop(self) -> None:
        if self.crop is None:
            messagebox.showinfo("提示", "先拖一个长方形。")
            return
        path = ROOT / "samples" / "crop_raw.png"
        ok, buf = cv2.imencode(".png", self.crop)
        if not ok:
            messagebox.showerror("保存失败", str(path))
            return
        path.write_bytes(buf.tobytes())
        messagebox.showinfo("已保存原图块", f"{path}\n{self.crop.shape[1]}×{self.crop.shape[0]}，未做分析。")

    def mainloop(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        CropBench().mainloop()
    except FileNotFoundError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("找不到原图", str(exc))


if __name__ == "__main__":
    main()
