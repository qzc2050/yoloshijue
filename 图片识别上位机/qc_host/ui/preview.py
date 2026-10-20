# -*- coding: utf-8 -*-
"""Live preview window. Target: ~3 fps full-sensor GigE stream."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from qc_host.devices.camera import Camera
from qc_host.devices.light import CoaxialLight
from qc_host.devices.motion import MotionStage
from qc_host.vision.frontlight import inspect_frontlight

PREVIEW_MAX_SIDE = 1600


class PreviewApp:
    BG = "#101820"
    PANEL = "#17232d"
    PANEL_ALT = "#1d2c38"
    TEXT = "#e8f0f5"
    MUTED = "#9eb1bf"
    ACCENT = "#33b5e5"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("质检上位机 · 分析")
        self.root.geometry("1440x860")
        self.root.minsize(1100, 680)
        self.root.configure(bg=self.BG)

        self.camera = Camera()
        self.motion = MotionStage()
        self.light = CoaxialLight()

        self.ip_var = tk.StringVar(value=Camera.DEFAULT_IP)
        self.camera_info_var = tk.StringVar(value="相机未连接")
        self.motion_var = tk.StringVar(value=self.motion.status_text())
        self.light_var = tk.StringVar(value=self.light.status_text())
        self.status_var = tk.StringVar(value="请先在 MVS 中停止采集，再连接相机。")
        self.hud_var = tk.StringVar(value="fps —    曝光 —    增益 —    均值 —    过曝 —")
        self.show_inspect_var = tk.BooleanVar(value=False)
        self.inspect_summary_var = tk.StringVar(value="尚未分析。正面打光、对实后点「分析当前帧」。")
        self._inspect_bgr: Optional[np.ndarray] = None
        self._inspect_full: Optional[np.ndarray] = None
        self._inspect_layout: Optional[np.ndarray] = None
        self._inspect_chambers: list = []

        self.connect_btn: Optional[ttk.Button] = None
        self.preview_btn: Optional[ttk.Button] = None
        self.canvas: Optional[tk.Canvas] = None
        self.preview_tk: Optional[ImageTk.PhotoImage] = None
        self._preview_bgr: Optional[np.ndarray] = None
        self._full_bgr: Optional[np.ndarray] = None
        self.zoom: Optional[float] = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start: Optional[tuple[int, int]] = None

        self._previewing = False
        self._stop_event = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None
        self._poll_job: Optional[str] = None
        self._lock = threading.Lock()
        self._latest: Optional[tuple[np.ndarray, np.ndarray, dict]] = None
        self._grab_error: Optional[str] = None

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.TEXT, font=("Microsoft YaHei", 10))
        style.configure(
            "Muted.TLabel",
            background=self.PANEL,
            foreground=self.MUTED,
            font=("Microsoft YaHei", 9),
        )
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Microsoft YaHei", 20))
        style.configure(
            "Heading.TLabel",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Microsoft YaHei", 11),
        )
        style.configure(
            "Hud.TLabel",
            background=self.PANEL,
            foreground=self.ACCENT,
            font=("Consolas", 11),
        )
        style.configure(
            "TButton",
            background="#285e76",
            foreground="#ffffff",
            borderwidth=0,
            padding=(12, 7),
            font=("Microsoft YaHei", 9),
        )
        style.map("TButton", background=[("active", "#347b99")])
        style.configure(
            "TCheckbutton",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Microsoft YaHei", 9),
        )
        style.map("TCheckbutton", background=[("active", self.PANEL)])

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="TFrame", padding=(24, 16, 24, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="质检上位机", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="分析  ·  合格 / 渗墨 / 缺墨 / 不能识别",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(16, 0), pady=(8, 0))
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel", anchor=tk.E).pack(
            side=tk.RIGHT, pady=(8, 0)
        )

        body = ttk.Frame(self.root, padding=(24, 0, 24, 20))
        body.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=16)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.configure(width=300)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        stage = ttk.Frame(body, style="Panel.TFrame", padding=12)
        stage.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(stage, style="Panel.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(toolbar, text="相机画面", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="适应窗口", command=self.fit_view).pack(side=tk.RIGHT)
        ttk.Label(
            toolbar,
            text="滚轮缩放  ·  左键拖动  ·  双击适应窗口",
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT, padx=(0, 12))
        ttk.Label(stage, textvariable=self.hud_var, style="Hud.TLabel").pack(anchor=tk.W, pady=(0, 8))
        self.canvas = tk.Canvas(stage, bg="#0a1015", highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._paint())
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_zoom_steps(e, 1))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_zoom_steps(e, -1))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag_view)
        self.canvas.bind("<ButtonRelease-1>", self._end_drag)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.fit_view())

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="海康相机", style="Heading.TLabel").pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="连接前请关闭 MVS 采集。全分辨率预览约 3–6 帧，够对焦和构图。",
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(4, 12))

        ttk.Label(parent, text="相机 IP", style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 4))
        ttk.Entry(parent, textvariable=self.ip_var).pack(fill=tk.X)
        ttk.Label(
            parent,
            textvariable=self.camera_info_var,
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(6, 8))

        ttk.Button(parent, text="刷新设备列表", command=self.refresh_devices).pack(fill=tk.X, pady=(0, 4))
        self.connect_btn = ttk.Button(parent, text="连接相机", command=self.toggle_camera)
        self.connect_btn.pack(fill=tk.X, pady=(0, 4))
        self.preview_btn = ttk.Button(parent, text="开始预览", command=self.toggle_preview)
        self.preview_btn.pack(fill=tk.X)

        ttk.Label(parent, text="分析", style="Heading.TLabel").pack(anchor=tk.W, pady=(22, 6))
        ttk.Label(
            parent,
            text="先抓住外围最明显的跑道形外轮廓，得到电离室后再套模型：外碳油 / 外铜 / FR4 0.3mm / 内铜 / 内碳油。渗墨=间隔进碳；缺墨=内圈碳油露板。",
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(0, 8))
        ttk.Button(parent, text="分析当前帧", command=self.inspect_current).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(parent, text="打开图片分析", command=self.inspect_file).pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(
            parent,
            text="显示分析结果",
            variable=self.show_inspect_var,
            command=self._paint,
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            textvariable=self.inspect_summary_var,
            style="Muted.TLabel",
            wraplength=260,
        ).pack(anchor=tk.W, pady=(6, 0))

        ttk.Label(parent, text="后续工位", style="Heading.TLabel").pack(anchor=tk.W, pady=(22, 6))
        ttk.Label(parent, text="移动平台", style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.motion_var, style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(parent, text="同轴电源", style="Muted.TLabel").pack(anchor=tk.W)
        ttk.Label(parent, textvariable=self.light_var, style="Muted.TLabel").pack(anchor=tk.W)

    def refresh_devices(self) -> None:
        try:
            found = self.camera.enumerate()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("枚举失败", str(exc))
            return
        if not found:
            self.status_var.set("网口上看不到相机。检查网线和 192.168.0.x。")
            return
        first_ip, model, serial = found[0]
        self.ip_var.set(first_ip)
        listing = "；".join(f"{ip} {name}" for ip, name, _sn in found)
        self.status_var.set(f"可见 {len(found)} 台：{listing}")
        self.camera_info_var.set(f"{model}  {serial}  {first_ip}")

    def toggle_camera(self) -> None:
        if self.camera.connected:
            self._stop_preview(join=True)
            self.camera.disconnect()
            self.camera_info_var.set("相机未连接")
            if self.connect_btn is not None:
                self.connect_btn.configure(text="连接相机")
            self.status_var.set("相机已断开。")
            self.hud_var.set("fps —    曝光 —    增益 —    均值 —    过曝 —")
            self._full_bgr = None
            self._preview_bgr = None
            self._inspect_bgr = None
            self._inspect_full = None
            self._inspect_layout = None
            self._inspect_chambers = []
            return
        ip = (self.ip_var.get() or "").strip()
        if not ip:
            messagebox.showinfo("提示", "请填写相机 IP。")
            return
        try:
            info = self.camera.connect(ip)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("连接失败", str(exc))
            self.status_var.set("相机连接失败。")
            return
        self.camera_info_var.set(info)
        if self.connect_btn is not None:
            self.connect_btn.configure(text="断开相机")
        self.status_var.set(f"已连接 {info}。可以开始预览。")

    def toggle_preview(self) -> None:
        if not self.camera.connected:
            messagebox.showinfo("提示", "请先连接相机。")
            return
        if self._previewing:
            self._stop_preview(join=True)
            self.status_var.set("预览已停止。")
            return
        self._stop_event.clear()
        self._grab_error = None
        self._previewing = True
        if self.preview_btn is not None:
            self.preview_btn.configure(text="停止预览")
        self.status_var.set("预览中…")
        self._grab_thread = threading.Thread(target=self._grab_loop, name="qc-preview", daemon=True)
        self._grab_thread.start()
        self._schedule_poll()

    def _stop_preview(self, join: bool = True) -> None:
        self._previewing = False
        self._stop_event.set()
        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except tk.TclError:
                pass
            self._poll_job = None
        thread = self._grab_thread
        self._grab_thread = None
        if join and thread is not None and thread.is_alive():
            thread.join(timeout=3.5)
        if self.preview_btn is not None:
            self.preview_btn.configure(text="开始预览")

    def _schedule_poll(self) -> None:
        if not self._previewing:
            return
        self._poll_job = self.root.after(30, self._poll_frame)

    def _poll_frame(self) -> None:
        if not self._previewing:
            return
        error = self._grab_error
        if error:
            self._on_preview_error(error)
            return
        with self._lock:
            item = self._latest
            self._latest = None
        if item is not None:
            self._show_frame(*item)
        self._schedule_poll()

    def _grab_loop(self) -> None:
        last_tick = time.perf_counter()
        fps_ema = 0.0
        n = 0
        exposure_us = self.camera.exposure_us()
        gain_db = self.camera.gain_db()
        while not self._stop_event.is_set():
            try:
                frame = self.camera.grab_bgr(timeout_ms=4000)
            except Exception as exc:  # noqa: BLE001
                if not self._stop_event.is_set():
                    self._grab_error = str(exc)
                return
            if self._stop_event.is_set():
                return
            now = time.perf_counter()
            inst = 1.0 / max(1e-3, now - last_tick)
            last_tick = now
            fps_ema = inst if fps_ema <= 0.1 else (0.35 * inst + 0.65 * fps_ema)
            src_h, src_w = frame.shape[:2]
            preview = _downscale(frame, PREVIEW_MAX_SIDE)
            n += 1
            if n % 10 == 0:
                exposure_us = self.camera.exposure_us()
                gain_db = self.camera.gain_db()
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY) if preview.ndim == 3 else preview
            stats = {
                "fps": fps_ema,
                "mean": float(gray.mean()),
                "sat": float((gray >= 250).mean() * 100.0),
                "src_w": int(src_w),
                "src_h": int(src_h),
                "exposure_us": exposure_us,
                "gain_db": gain_db,
            }
            with self._lock:
                self._latest = (preview, frame, stats)

    def _show_frame(self, preview: np.ndarray, full: np.ndarray, stats: dict) -> None:
        self._preview_bgr = preview
        self._full_bgr = full
        exp = stats.get("exposure_us")
        gain = stats.get("gain_db")
        exp_txt = f"{exp / 1000:.1f} ms" if exp is not None else "—"
        gain_txt = f"{gain:.1f} dB" if gain is not None else "—"
        note = ""
        if exp is not None and exp > 350_000:
            note = "    曝光偏长，帧率会被压住"
        self.hud_var.set(
            f"fps {stats['fps']:.1f}    "
            f"曝光 {exp_txt}    "
            f"增益 {gain_txt}    "
            f"均值 {stats['mean']:.0f}    "
            f"过曝 {stats['sat']:.2f}%    "
            f"{stats['src_w']}x{stats['src_h']}"
            f"{note}"
        )
        if stats["fps"] >= 2.5:
            self.status_var.set(f"预览中  {stats['fps']:.1f} fps")
        else:
            self.status_var.set(f"预览偏慢  {stats['fps']:.1f} fps（看曝光是否超过 300 ms）")
        self._paint()

    def _on_preview_error(self, message: str) -> None:
        self._stop_preview(join=False)
        self.status_var.set(f"预览失败：{message}")
        messagebox.showerror("预览失败", message)

    def inspect_current(self) -> None:
        frame = self._full_bgr if self._full_bgr is not None else self._preview_bgr
        if frame is None:
            messagebox.showinfo("提示", "请先预览相机，或用「打开图片分析」。")
            return
        self._run_inspect(frame)

    def inspect_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择正面打光图片",
            filetypes=[
                ("图片", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("全部", "*.*"),
            ],
        )
        if not path:
            return
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            messagebox.showerror("打开失败", path)
            return
        self._preview_bgr = _downscale(frame, PREVIEW_MAX_SIDE)
        self._full_bgr = frame
        self.fit_view()
        self._run_inspect(frame)

    def _run_inspect(self, frame: np.ndarray) -> None:
        try:
            result = inspect_frontlight(frame)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("分析失败", str(exc))
            self.inspect_summary_var.set(f"分析失败：{exc}")
            return
        self._inspect_full = frame
        self._inspect_layout = _downscale(frame, PREVIEW_MAX_SIDE)
        self._inspect_chambers = result.chambers
        self._inspect_bgr = result.annotated_bgr
        self.show_inspect_var.set(True)
        self.inspect_summary_var.set(result.summary)
        self.status_var.set(result.summary)
        self._paint()

    def fit_view(self) -> None:
        self.zoom = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._paint()

    def _fit_zoom(self) -> float:
        layout = self._layout_bgr()
        if self.canvas is None or layout is None:
            return 1.0
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        height, width = layout.shape[:2]
        return min(canvas_w / width, canvas_h / height, 1.0)

    def _layout_bgr(self) -> Optional[np.ndarray]:
        if self.show_inspect_var.get() and self._inspect_layout is not None:
            return self._inspect_layout
        return self._preview_bgr

    def _current_zoom(self) -> float:
        return self._fit_zoom() if self.zoom is None else self.zoom

    def _wheel_zoom(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        self._wheel_zoom_steps(event, 1 if event.delta > 0 else -1)

    def _wheel_zoom_steps(self, event: tk.Event, direction: int) -> None:
        if self._layout_bgr() is None or self.canvas is None:
            return
        old = self._current_zoom()
        factor = 1.15 if direction > 0 else 1 / 1.15
        new = min(16.0, max(0.05, old * factor))
        if abs(new - old) < 1e-6:
            return
        height, width = self._layout_bgr().shape[:2]
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        ox = (canvas_w - width * old) / 2.0 + self.pan_x
        oy = (canvas_h - height * old) / 2.0 + self.pan_y
        img_x = (event.x - ox) / old
        img_y = (event.y - oy) / old
        self.zoom = new
        self.pan_x = event.x - img_x * new - (canvas_w - width * new) / 2.0
        self.pan_y = event.y - img_y * new - (canvas_h - height * new) / 2.0
        self._paint()

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        if self.canvas is not None:
            self.canvas.configure(cursor="fleur")

    def _drag_view(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        prev_x, prev_y = self._drag_start
        self.pan_x += event.x - prev_x
        self.pan_y += event.y - prev_y
        self._drag_start = (event.x, event.y)
        self._paint()

    def _bind_wheel(self, _event: tk.Event | None = None) -> None:
        if self.canvas is not None:
            self.canvas.bind_all("<MouseWheel>", self._wheel_zoom)

    def _unbind_wheel(self, _event: tk.Event | None = None) -> None:
        if self.canvas is not None:
            self.canvas.unbind_all("<MouseWheel>")

    def _end_drag(self, _event: tk.Event) -> None:
        self._drag_start = None
        if self.canvas is not None:
            self.canvas.configure(cursor="hand2")

    def _paint_source(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        layout = self._layout_bgr()
        if layout is None:
            return None
        if self.show_inspect_var.get() and self._inspect_full is not None:
            return layout, self._inspect_full
        src = self._full_bgr if self._full_bgr is not None else layout
        return layout, src

    def _paint(self) -> None:
        if self.canvas is None:
            return
        packed = self._paint_source()
        if packed is None:
            return
        layout, src = packed
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        height, width = layout.shape[:2]
        zoom = self._current_zoom()
        x0 = (canvas_w - width * zoom) / 2.0 + self.pan_x
        y0 = (canvas_h - height * zoom) / 2.0 + self.pan_y
        sx0 = max(0.0, (0 - x0) / zoom)
        sy0 = max(0.0, (0 - y0) / zoom)
        sx1 = min(float(width), (canvas_w - x0) / zoom)
        sy1 = min(float(height), (canvas_h - y0) / zoom)
        if sx1 <= sx0 or sy1 <= sy0:
            self.canvas.delete("all")
            return
        scale_x = src.shape[1] / width
        scale_y = src.shape[0] / height
        fx0 = max(0, int(sx0 * scale_x))
        fy0 = max(0, int(sy0 * scale_y))
        fx1 = min(src.shape[1], int(np.ceil(sx1 * scale_x)))
        fy1 = min(src.shape[0], int(np.ceil(sy1 * scale_y)))
        if fx1 <= fx0 or fy1 <= fy0:
            self.canvas.delete("all")
            return
        crop = src[fy0:fy1, fx0:fx1]
        screen_per_src = zoom / scale_x
        out_w = max(1, int(round(crop.shape[1] * screen_per_src)))
        out_h = max(1, int(round(crop.shape[0] * zoom / scale_y)))
        out_w = min(out_w, canvas_w + 8)
        out_h = min(out_h, canvas_h + 8)
        interp = cv2.INTER_AREA if screen_per_src < 0.999 else cv2.INTER_LINEAR
        image = cv2.resize(crop, (out_w, out_h), interpolation=interp)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.preview_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(
            int(x0 + fx0 / scale_x * zoom),
            int(y0 + fy0 / scale_y * zoom),
            anchor=tk.NW,
            image=self.preview_tk,
        )
        self._draw_inspect_overlay(x0, y0, zoom, scale_x, scale_y)

    def _draw_inspect_overlay(self, x0: float, y0: float, zoom: float, scale_x: float, scale_y: float) -> None:
        if self.canvas is None or not self.show_inspect_var.get() or not self._inspect_chambers:
            return
        colors = {
            "合格": "#50d250",
            "渗墨": "#e02828",
            "缺墨": "#ffd000",
            "不能识别": "#ffa000",
        }
        font_n = max(9, min(20, int(round(10 * max(zoom, 0.75)))))
        width = max(2, int(round(2 * max(zoom, 0.8))))
        for ch in self._inspect_chambers:
            color = colors.get(ch.verdict, "#ffffff")
            ring = ch.gap_line
            if ring is not None:
                pts = ring.reshape(-1, 2)
                if len(pts) >= 3:
                    coords = []
                    for px, py in pts:
                        coords.append(x0 + float(px) / scale_x * zoom)
                        coords.append(y0 + float(py) / scale_y * zoom)
                    coords.extend(coords[:2])
                    self.canvas.create_line(
                        *coords,
                        fill=color,
                        width=width,
                        joinstyle=tk.ROUND,
                        capstyle=tk.ROUND,
                    )
            box_src = ch.gap_line if ch.gap_line is not None else ch.outer
            bx, by, _bw, _bh = cv2.boundingRect(box_src)
            self.canvas.create_text(
                x0 + bx / scale_x * zoom,
                y0 + by / scale_y * zoom - 6,
                text=f"#{ch.index} {ch.verdict}",
                fill=color,
                font=("Microsoft YaHei", font_n),
                anchor=tk.SW,
            )
            for mark in ch.marks:
                sx = x0 + mark.x / scale_x * zoom
                sy = y0 + mark.y / scale_y * zoom
                rad = max(4.0, 5.0 * max(zoom, 0.8))
                self.canvas.create_oval(sx - rad, sy - rad, sx + rad, sy + rad, outline=color, width=width)

    def _on_close(self) -> None:
        self._unbind_wheel()
        self._stop_preview(join=True)
        self.camera.disconnect()
        self.root.destroy()


def _downscale(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 0.999:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
