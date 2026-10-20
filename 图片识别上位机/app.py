# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Optional

import cv2
import fitz
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from pdf_image_loader import is_pdf, load_image_file, load_pdf_page
from ring_detector import DetectionOutput, detect_rings
from hikrobot_camera import HikrobotCamera


class InspectionApp:
    BG = "#101820"
    PANEL = "#17232d"
    PANEL_ALT = "#1d2c38"
    TEXT = "#e8f0f5"
    MUTED = "#9eb1bf"
    ACCENT = "#33b5e5"
    WARN = "#ffad42"
    ERROR = "#ff6258"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("铜环视觉检测")
        self.root.geometry("1560x940")
        self.root.minsize(1180, 760)
        self.root.configure(bg=self.BG)

        self.source_path: Optional[str] = None
        self.current_image_bgr: Optional[np.ndarray] = None
        self.loaded_image_key: Optional[tuple] = None
        self.current_detection: Optional[DetectionOutput] = None
        self.preview_tk: Optional[ImageTk.PhotoImage] = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start: Optional[tuple[int, int]] = None
        self.selected_ring_index: Optional[int] = None
        self.filter_var = tk.StringVar(value="all")
        self._detecting = False
        self._progress_win: Optional[tk.Toplevel] = None
        self._progress_bar: Optional[ttk.Progressbar] = None
        self._progress_label_var = tk.StringVar(value="正在检测，请稍候…")
        self.analyze_btn: Optional[ttk.Button] = None
        self.save_btn: Optional[ttk.Button] = None
        self.open_btn: Optional[ttk.Button] = None
        self.camera_connect_btn: Optional[ttk.Button] = None
        self.camera_grab_btn: Optional[ttk.Button] = None
        self.camera_preview_btn: Optional[ttk.Button] = None
        self.live_preview_bgr: Optional[np.ndarray] = None
        self.camera = HikrobotCamera()
        self._previewing = False
        self._preview_job: Optional[str] = None

        self.path_var = tk.StringVar(value="未导入文件")
        self.camera_ip_var = tk.StringVar(value="192.168.0.100")
        self.camera_info_var = tk.StringVar(value="相机未连接")
        self.image_dpi_var = tk.StringVar(value="1200")
        self.threshold_var = tk.StringVar(value="5")
        self.sr_scale_var = tk.StringVar(value="2")
        self.page_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="请连接相机采图，或导入图片 / PDF。")
        self.component_var = tk.StringVar(value="0")
        self.ring_var = tk.StringVar(value="0")
        self.narrow_var = tk.StringVar(value="0")
        self.open_var = tk.StringVar(value="0")
        self.irregular_var = tk.StringVar(value="0")
        self.sort_var = tk.StringVar(value="min_width")

        self._configure_style()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.TEXT, font=("Microsoft YaHei", 10))
        style.configure("Muted.TLabel", background=self.PANEL, foreground=self.MUTED, font=("Microsoft YaHei", 9))
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Microsoft YaHei", 20))
        style.configure("Heading.TLabel", background=self.PANEL, foreground=self.TEXT, font=("Microsoft YaHei", 11))
        style.configure("Metric.TLabel", background=self.PANEL_ALT, foreground=self.TEXT, font=("Microsoft YaHei", 24))
        style.configure("MetricName.TLabel", background=self.PANEL_ALT, foreground=self.MUTED, font=("Microsoft YaHei", 9))
        style.configure("TButton", background="#285e76", foreground="#ffffff", borderwidth=0, padding=(12, 7), font=("Microsoft YaHei", 9))
        style.map("TButton", background=[("active", "#347b99")])
        style.configure("TEntry", fieldbackground="#0e161d", foreground=self.TEXT, insertcolor=self.TEXT, borderwidth=0, padding=6)
        style.configure("TCombobox", fieldbackground="#0e161d", background="#0e161d", foreground=self.TEXT, arrowcolor=self.TEXT)
        style.configure("Treeview", background="#0e161d", fieldbackground="#0e161d", foreground=self.TEXT, rowheight=27, borderwidth=0, font=("Microsoft YaHei", 9))
        style.configure("Treeview.Heading", background="#263846", foreground=self.TEXT, relief="flat", font=("Microsoft YaHei", 9), padding=7)
        style.map("Treeview", background=[("selected", "#22556c")])

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="TFrame", padding=(24, 16, 24, 10))
        header.pack(fill=tk.X)
        ttk.Label(header, text="铜环视觉检测", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(header, text="双跑道环 · 任意角度 · 整板一张图线宽检测", style="Muted.TLabel").pack(side=tk.LEFT, padx=(16, 0), pady=(8, 0))
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel", anchor=tk.E).pack(side=tk.RIGHT, pady=(8, 0))

        body = ttk.Frame(self.root, padding=(24, 0, 24, 20))
        body.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(body, style="Panel.TFrame", padding=16)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        sidebar.configure(width=315)
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        content = ttk.Frame(body)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_metrics(content)
        self._build_preview(content)
        self._build_table(content)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="检测设置", style="Heading.TLabel").pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="相机拍得远时整板像素摊得薄，线宽只能粗测。可连接海康相机采图，或导入扫描件。多相机/分块拍摄能提高线宽精度。",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(3, 16))
        self.open_btn = ttk.Button(parent, text="导入图片或 PDF", command=self.open_file)
        self.open_btn.pack(fill=tk.X)

        ttk.Label(parent, text="海康相机", style="Heading.TLabel").pack(anchor=tk.W, pady=(18, 6))
        ttk.Label(parent, text="相机 IP", style="Muted.TLabel").pack(anchor=tk.W, pady=(0, 4))
        ttk.Entry(parent, textvariable=self.camera_ip_var).pack(fill=tk.X)
        ttk.Label(parent, textvariable=self.camera_info_var, style="Muted.TLabel", wraplength=270).pack(anchor=tk.W, pady=(4, 6))
        self.camera_connect_btn = ttk.Button(parent, text="连接相机", command=self.toggle_camera)
        self.camera_connect_btn.pack(fill=tk.X, pady=(0, 4))
        self.camera_preview_btn = ttk.Button(parent, text="开始预览", command=self.toggle_preview)
        self.camera_preview_btn.pack(fill=tk.X, pady=(0, 4))
        self.camera_grab_btn = ttk.Button(parent, text="采集一帧用于检测", command=self.grab_camera_frame)
        self.camera_grab_btn.pack(fill=tk.X)
        ttk.Label(
            parent,
            text="连接前请在 MVS 中停止采集。采到的图可直接点开始检测。",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(4, 0))

        ttk.Label(parent, text="当前文件", style="Muted.TLabel").pack(anchor=tk.W, pady=(18, 3))
        ttk.Label(parent, textvariable=self.path_var, style="Muted.TLabel", wraplength=270).pack(anchor=tk.W)

        self._field(parent, "图像 DPI", self.image_dpi_var)
        self._field(parent, "环带宽度阈值 (px)", self.threshold_var)
        self._field(parent, "PDF 页码", self.page_var)

        ttk.Label(parent, text="分辨率补偿", style="Muted.TLabel").pack(anchor=tk.W, pady=(15, 4))
        sr_box = ttk.Combobox(
            parent,
            textvariable=self.sr_scale_var,
            values=("关闭", "2", "3"),
            state="readonly",
        )
        sr_box.pack(fill=tk.X)
        ttk.Label(
            parent,
            text="整板一张图请开 2 或 3 倍：先定位，再对每个铜环放大后量线宽。不能凭空创造物理细节，但可做亚像素补偿。",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(4, 0))

        self.analyze_btn = ttk.Button(parent, text="开始检测", command=self.analyze)
        self.analyze_btn.pack(fill=tk.X, pady=(18, 6))
        self.save_btn = ttk.Button(parent, text="保存标注图与 CSV", command=self.save_result)
        self.save_btn.pack(fill=tk.X)

        ttk.Label(parent, text="图例说明", style="Heading.TLabel").pack(anchor=tk.W, pady=(26, 8))
        self._legend(parent, self.ACCENT, "青色", "完整双跑道环")
        self._legend(parent, self.WARN, "橙色", "黑覆盖完全盖住环带（判定溢出）")
        self._legend(parent, self.ERROR, "红色", "环带残余宽度低于阈值（疑似部分覆盖）")
        self._legend(parent, "#c832c8", "紫色", "环带宽窄变化异常")
        ttk.Label(
            parent,
            text="预览支持滚轮缩放与拖拽；单击表格行可定位。内/外环列为残余最窄宽度；覆盖溢出优先于线宽告警。低于阈值的铜线在表中筛“环带过细”。",
            style="Muted.TLabel",
            wraplength=270,
        ).pack(anchor=tk.W, pady=(18, 0))

    def _field(self, parent: ttk.Frame, label: str, value: tk.StringVar) -> None:
        ttk.Label(parent, text=label.upper(), style="Muted.TLabel").pack(anchor=tk.W, pady=(15, 4))
        ttk.Entry(parent, textvariable=value).pack(fill=tk.X)

    def _legend(self, parent: ttk.Frame, color: str, title: str, description: str) -> None:
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, bg=color, width=2, height=1).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(row, text=f"{title}: {description}", style="Muted.TLabel").pack(side=tk.LEFT)

    def _build_metrics(self, parent: ttk.Frame) -> None:
        metrics = ttk.Frame(parent)
        metrics.pack(fill=tk.X, pady=(0, 12))
        self._metric(metrics, "图案总数", self.component_var, 0)
        self._metric(metrics, "已标记图案", self.ring_var, 1)
        self._metric(metrics, "覆盖溢出", self.open_var, 2, self.WARN)
        self._metric(metrics, "环带过细", self.narrow_var, 3, self.ERROR)
        self._metric(metrics, "宽窄异常", self.irregular_var, 4, "#c832c8")
        for column in range(5):
            metrics.columnconfigure(column, weight=1)

    def _metric(self, parent: ttk.Frame, name: str, value: tk.StringVar, column: int, color: str | None = None) -> None:
        card = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        value_label = ttk.Label(card, textvariable=value, style="Metric.TLabel")
        if color:
            value_label.configure(foreground=color)
        value_label.pack(anchor=tk.W)
        ttk.Label(card, text=name, style="MetricName.TLabel").pack(anchor=tk.W, pady=(3, 0))

    def _build_preview(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        toolbar = ttk.Frame(panel, style="Panel.TFrame")
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(toolbar, text="检测图像预览", style="Heading.TLabel").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="缩小", command=lambda: self.change_zoom(0.8)).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(toolbar, text="放大", command=lambda: self.change_zoom(1.25)).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(toolbar, text="适应窗口", command=self.fit_preview).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(panel, bg="#0a1015", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._canvas_resized)
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._drag_preview)
        self.canvas.bind("<MouseWheel>", self._wheel_zoom)

    def _build_table(self, parent: ttk.Frame) -> None:
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        panel.pack(fill=tk.BOTH, expand=False, pady=(12, 0))
        table_header = ttk.Frame(panel, style="Panel.TFrame")
        table_header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(table_header, text="检测结果明细", style="Heading.TLabel").pack(side=tk.LEFT)
        for text, value in [("全部", "all"), ("覆盖溢出", "open"), ("环带过细", "narrow"), ("已配", "paired"), ("宽窄异常", "irregular")]:
            ttk.Radiobutton(table_header, text=text, value=value, variable=self.filter_var, command=self.fill_table).pack(side=tk.RIGHT, padx=(8, 0))

        sort_row = ttk.Frame(panel, style="Panel.TFrame")
        sort_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(sort_row, text="排序:", style="Muted.TLabel").pack(side=tk.LEFT)
        for text, value in [("最小线宽↑", "min_width"), ("内环最窄↑", "inner_width"), ("外环最窄↑", "outer_width")]:
            ttk.Radiobutton(sort_row, text=text, value=value, variable=self.sort_var, command=self.fill_table).pack(side=tk.LEFT, padx=(8, 0))

        columns = (
            "index", "component", "position", "angle", "size",
            "inner_width", "outer_width", "min_width", "min_um",
            "irreg_range", "open", "status",
        )
        self.table = ttk.Treeview(panel, columns=columns, show="headings", height=8)
        headings = [
            "铜环", "组件", "位置", "角度", "外包尺寸",
            "内环最窄 px", "外环最窄 px", "最小线宽 px", "最小 µm",
            "宽度范围 px", "溢出", "状态",
        ]
        widths = [46, 50, 88, 50, 72, 78, 78, 78, 70, 86, 42, 72]
        for column, heading, width in zip(columns, headings, widths):
            self.table.heading(column, text=heading)
            self.table.column(column, width=width, anchor=tk.CENTER)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.table.bind("<<TreeviewSelect>>", self._select_result)
        scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=self.table.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.table.configure(yscrollcommand=scrollbar.set)

    def toggle_camera(self) -> None:
        if self.camera.cam is not None:
            self._stop_preview()
            self.camera.disconnect()
            self.camera_info_var.set("相机未连接")
            if self.camera_connect_btn is not None:
                self.camera_connect_btn.configure(text="连接相机")
            self.status_var.set("相机已断开。")
            return
        ip = (self.camera_ip_var.get() or "").strip()
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
        if self.camera_connect_btn is not None:
            self.camera_connect_btn.configure(text="断开相机")
        self.status_var.set(f"已连接 {info}。可预览或采集一帧。")

    def toggle_preview(self) -> None:
        if self.camera.cam is None:
            messagebox.showinfo("提示", "请先连接相机。")
            return
        if self._previewing:
            self._stop_preview()
            return
        self._previewing = True
        if self.camera_preview_btn is not None:
            self.camera_preview_btn.configure(text="停止预览")
        self.status_var.set("相机预览中…")
        self._schedule_preview()

    def _stop_preview(self) -> None:
        self._previewing = False
        if self._preview_job is not None:
            try:
                self.root.after_cancel(self._preview_job)
            except tk.TclError:
                pass
            self._preview_job = None
        if self.camera_preview_btn is not None:
            self.camera_preview_btn.configure(text="开始预览")

    def _schedule_preview(self) -> None:
        if not self._previewing:
            return
        threading.Thread(target=self._preview_once, daemon=True).start()

    def _preview_once(self) -> None:
        try:
            frame = self.camera.grab_bgr(timeout_ms=2000)
            height, width = frame.shape[:2]
            scale = min(1.0, 1400.0 / max(width, height))
            if scale < 0.999:
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            self.root.after(0, lambda img=frame: self._apply_preview_frame(img))
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, lambda msg=str(exc): self._preview_failed(msg))

    def _apply_preview_frame(self, frame: np.ndarray) -> None:
        if not self._previewing:
            return
        self.live_preview_bgr = frame
        self.current_detection = None
        self.render_preview()
        self._preview_job = self.root.after(80, self._schedule_preview)

    def _preview_failed(self, message: str) -> None:
        self._stop_preview()
        self.status_var.set(f"预览失败：{message}")

    def grab_camera_frame(self) -> None:
        if self.camera.cam is None:
            messagebox.showinfo("提示", "请先连接相机。")
            return
        was_previewing = self._previewing
        self._stop_preview()
        try:
            frame = self.camera.grab_bgr(timeout_ms=4000)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("采图失败", str(exc))
            self.status_var.set("采图失败。")
            if was_previewing:
                self.toggle_preview()
            return
        ip = (self.camera_ip_var.get() or "").strip()
        self.source_path = f"camera://{ip}"
        self.path_var.set(self.camera.info or self.source_path)
        self.current_image_bgr = frame
        self.loaded_image_key = (self.source_path, 1, 0.0, False)
        self.current_detection = None
        self.live_preview_bgr = None
        self.selected_ring_index = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        height, width = frame.shape[:2]
        self.status_var.set(f"已采集 {width}×{height}，可开始检测。")
        self.fit_preview()

    def _on_close(self) -> None:
        self._stop_preview()
        try:
            self.camera.disconnect()
        except Exception:
            pass
        self.root.destroy()

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图片或 PDF 文件",
            filetypes=[("图片或 PDF", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf"), ("所有文件", "*.*")],
        )
        if not path:
            return
        self.source_path = path
        self.path_var.set(path)
        self.current_detection = None
        self.loaded_image_key = None
        self.selected_ring_index = None
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        if is_pdf(path):
            pdf = fitz.open(path)
            try:
                self.status_var.set(f"已导入 PDF，共 {pdf.page_count} 页，当前准备检测第 1 页。")
            finally:
                pdf.close()
        else:
            self.status_var.set("图片已导入，可以开始检测。")
        self.page_var.set("1")
        self.load_current_image()
        self.fit_preview()

    def load_current_image(self, force: bool = False) -> None:
        if not self.source_path or str(self.source_path).startswith("camera:"):
            return
        input_dpi = self._positive_float(self.image_dpi_var.get(), "图像 DPI")
        page = 1
        render_dpi = input_dpi
        if is_pdf(self.source_path):
            page = max(1, int(float(self.page_var.get() or "1")))
            # Stadium rings need enough pixels for template matching; clamp to 400–600.
            render_dpi = min(max(input_dpi, 400.0), 600.0)
        image_key = (self.source_path, page, float(render_dpi), is_pdf(self.source_path))
        if not force and self.loaded_image_key == image_key and self.current_image_bgr is not None:
            return
        if is_pdf(self.source_path):
            image_rgb = load_pdf_page(self.source_path, page - 1, render_dpi)
        else:
            image_rgb = load_image_file(self.source_path)
        self.current_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self.loaded_image_key = image_key

    def _sr_scale(self) -> int:
        raw = (self.sr_scale_var.get() or "1").strip()
        if raw in {"关闭", "1", "off", "Off"}:
            return 1
        try:
            scale = int(float(raw))
        except ValueError:
            return 2
        return max(1, min(4, scale))

    def _positive_float(self, value: str, name: str) -> float:
        try:
            number = float(value.strip())
            if number <= 0:
                raise ValueError
            return number
        except ValueError as exc:
            raise ValueError(f"{name} 必须是大于 0 的数字。") from exc

    def _set_controls_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for button in (
            self.analyze_btn, self.save_btn, self.open_btn,
            self.camera_connect_btn, self.camera_grab_btn, self.camera_preview_btn,
        ):
            if button is not None:
                button.configure(state=state)

    def _show_progress(self, message: str = "正在检测，请稍候…") -> None:
        self._progress_label_var.set(message)
        if self._progress_win is not None and self._progress_win.winfo_exists():
            return

        win = tk.Toplevel(self.root)
        win.title("检测进行中")
        win.configure(bg=self.PANEL)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        # Prevent closing with X while busy.
        win.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(win, style="Panel.TFrame", padding=24)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, textvariable=self._progress_label_var, style="Heading.TLabel").pack(anchor=tk.W)
        ttk.Label(
            frame,
            text="正在识别任意角度双跑道环，并对每个铜环做局部超分量线宽…",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(8, 16))
        bar = ttk.Progressbar(frame, mode="indeterminate", length=320)
        bar.pack(fill=tk.X)
        bar.start(12)

        win.update_idletasks()
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        width, height = 380, 140
        x = root_x + max(0, (root_w - width) // 2)
        y = root_y + max(0, (root_h - height) // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")

        self._progress_win = win
        self._progress_bar = bar

    def _hide_progress(self) -> None:
        win = self._progress_win
        self._progress_win = None
        if win is not None and win.winfo_exists():
            try:
                win.grab_release()
            except tk.TclError:
                pass
            win.destroy()

    def analyze(self) -> None:
        if self.current_image_bgr is None:
            messagebox.showinfo("提示", "请先采集相机画面，或导入图片 / PDF。")
            return
        if self._detecting:
            return
        try:
            input_dpi = self._positive_float(self.image_dpi_var.get(), "图像 DPI")
            threshold_width = self._positive_float(self.threshold_var.get(), "环带宽度阈值 (px)")
            sr_scale = self._sr_scale()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        self._detecting = True
        self._set_controls_busy(True)
        self.status_var.set("正在准备图像…")
        self._show_progress("正在准备图像…")
        self.root.update_idletasks()

        try:
            if self.source_path and not str(self.source_path).startswith("camera:"):
                self.load_current_image()
        except Exception as exc:  # noqa: BLE001
            self._detecting = False
            self._set_controls_busy(False)
            self._hide_progress()
            messagebox.showerror("检测失败", str(exc))
            self.status_var.set("检测失败。")
            return

        if self.current_image_bgr is None:
            self._detecting = False
            self._set_controls_busy(False)
            self._hide_progress()
            return

        image = self.current_image_bgr.copy()
        self._progress_label_var.set("正在识别任意角度双跑道环并局部超分量线宽…")
        self.status_var.set("正在检测中，请稍候…")

        result_box: dict = {"output": None, "error": None}

        def worker() -> None:
            try:
                result_box["output"] = detect_rings(
                    image,
                    input_dpi,
                    threshold_width,
                    sr_scale=sr_scale,
                    allow_small=sr_scale >= 2,
                )
            except Exception as exc:  # noqa: BLE001
                result_box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def poll() -> None:
            if thread.is_alive():
                # Keep progress animation alive and UI responsive.
                self.root.after(80, poll)
                return

            self._hide_progress()
            self._set_controls_busy(False)
            self._detecting = False

            if result_box["error"] is not None:
                messagebox.showerror("检测失败", str(result_box["error"]))
                self.status_var.set("检测失败。")
                return

            output = result_box["output"]
            if output is None:
                self.status_var.set("检测失败。")
                return

            self.current_detection = output
            self.selected_ring_index = None
            self.update_summary()
            self.fill_table()
            self.render_preview()
            narrow = [r for r in output.results if r.is_below_threshold]
            overflow = sum(r.is_broken for r in output.results)
            extra = f" 已用 {output.sr_scale}× 局部补偿。" if output.sr_scale > 1 else ""
            if narrow:
                spots = "、".join(f"#{item.index}({item.x},{item.y})" for item in narrow[:8])
                tail = f" 等共 {len(narrow)} 处" if len(narrow) > 8 else f" 共 {len(narrow)} 处"
                extra += f" 低于阈值：{spots}{tail}。"
            self.status_var.set(
                f"检测完成：共 {len(output.results)} 个图案；"
                f"线宽低于阈值 {len(narrow)}，覆盖溢出 {overflow}。"
                + extra
            )

        self.root.after(80, poll)

    def update_summary(self) -> None:
        output = self.current_detection
        if output is None:
            return
        self.component_var.set(str(len(output.components)))
        self.ring_var.set(str(len(output.results)))
        self.narrow_var.set(str(sum(item.is_below_threshold for item in output.results)))
        self.open_var.set(str(sum(item.is_broken for item in output.results)))
        self.irregular_var.set(str(sum(item.is_irregular_edges for item in output.results)))

    def fill_table(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        if self.current_detection is None:
            return

        selection = self.filter_var.get()
        sort_mode = self.sort_var.get()
        if sort_mode == "inner_width":
            sort_key = lambda item: (item.inner_line_width_px, item.outer_line_width_px, item.index)
        elif sort_mode == "outer_width":
            sort_key = lambda item: (item.outer_line_width_px, item.inner_line_width_px, item.index)
        else:
            sort_key = lambda item: (item.min_line_width_px, item.inner_line_width_px, item.outer_line_width_px, item.index)

        results = sorted(self.current_detection.results, key=sort_key)
        for item in results:
            if selection == "narrow" and not item.is_below_threshold:
                continue
            if selection == "open" and not item.is_broken:
                continue
            if selection == "paired" and item.component_index is None:
                continue
            if selection == "irregular" and not item.is_irregular_edges:
                continue

            status = (
                "覆盖溢出" if item.is_broken
                else "环带过细" if item.is_below_threshold
                else "宽窄异常" if item.is_irregular_edges
                else "合格"
            )
            self.table.insert(
                "",
                tk.END,
                values=(
                    item.index,
                    f"C{item.component_index}" if item.component_index else "-",
                    f"{item.x}, {item.y}",
                    f"{item.angle_deg:.0f}°",
                    f"{item.w} x {item.h}",
                    f"{item.inner_line_width_px:.1f}",
                    f"{item.outer_line_width_px:.1f}",
                    f"{item.min_line_width_px:.1f}",
                    f"{item.min_width_um:.1f}",
                    f"{item.irregularity_min_px:.1f}-{item.irregularity_max_px:.1f}",
                    "是" if item.is_broken else "否",
                    status,
                ),
                tags=(status,),
                iid=str(item.index),
            )
        self.table.tag_configure("覆盖溢出", foreground=self.WARN)
        self.table.tag_configure("环带过细", foreground=self.ERROR)
        self.table.tag_configure("宽窄异常", foreground="#c832c8")
        self.table.tag_configure("合格", foreground=self.TEXT)

    def _select_result(self, _event: tk.Event) -> None:
        selection = self.table.selection()
        if not selection:
            return
        self.selected_ring_index = int(selection[0])
        self.focus_selected_ring()

    def focus_selected_ring(self) -> None:
        if self.current_detection is None or self.selected_ring_index is None:
            return
        result = next(
            (item for item in self.current_detection.results if item.index == self.selected_ring_index),
            None,
        )
        if result is None:
            return
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        image = self._active_preview()
        if image is None:
            return
        self.pan_x = int(image.shape[1] * self.zoom / 2 - (result.x + result.w / 2) * self.zoom)
        self.pan_y = int(image.shape[0] * self.zoom / 2 - (result.y + result.h / 2) * self.zoom)
        self.render_preview()

    def _active_preview(self) -> Optional[np.ndarray]:
        if self.current_detection is not None:
            return self.current_detection.annotated_bgr
        if self._previewing and self.live_preview_bgr is not None:
            return self.live_preview_bgr
        return self.current_image_bgr

    def fit_preview(self) -> None:
        image = self._active_preview()
        if image is None:
            return
        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        self.zoom = min(canvas_w / image.shape[1], canvas_h / image.shape[0], 1.0)
        self.pan_x = 0
        self.pan_y = 0
        self.render_preview()

    def change_zoom(self, factor: float) -> None:
        if self._active_preview() is None:
            return
        self.zoom = min(4.0, max(0.1, self.zoom * factor))
        self.render_preview()

    def _start_drag(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)

    def _drag_preview(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        previous_x, previous_y = self.drag_start
        self.pan_x += event.x - previous_x
        self.pan_y += event.y - previous_y
        self.drag_start = (event.x, event.y)
        self.render_preview()

    def _wheel_zoom(self, event: tk.Event) -> None:
        self.change_zoom(1.15 if event.delta > 0 else 1 / 1.15)

    def _canvas_resized(self, _event: tk.Event) -> None:
        self.render_preview()

    def render_preview(self) -> None:
        image = self._active_preview()
        if image is None:
            return
        width = max(1, int(image.shape[1] * self.zoom))
        height = max(1, int(image.shape[0] * self.zoom))
        display = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.preview_tk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        x = (self.canvas.winfo_width() - width) // 2 + self.pan_x
        y = (self.canvas.winfo_height() - height) // 2 + self.pan_y
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.preview_tk)
        if self.current_detection is not None and self.selected_ring_index is not None:
            selected = next(
                (item for item in self.current_detection.results if item.index == self.selected_ring_index),
                None,
            )
            if selected is not None:
                x0 = x + int(selected.x * self.zoom)
                y0 = y + int(selected.y * self.zoom)
                x1 = x + int((selected.x + selected.w) * self.zoom)
                y1 = y + int((selected.y + selected.h) * self.zoom)
                self.canvas.create_rectangle(x0 - 5, y0 - 5, x1 + 5, y1 + 5, outline="#fff32b", width=3)
                self.canvas.create_text(x0, max(12, y0 - 12), text=f"组件 C{selected.component_index}", fill="#fff32b", anchor=tk.SW, font=("Microsoft YaHei", 10))
        self.canvas.configure(scrollregion=(0, 0, max(width, self.canvas.winfo_width()), max(height, self.canvas.winfo_height())))

    def save_result(self) -> None:
        if self.current_detection is None:
            messagebox.showinfo("提示", "请先完成检测再导出结果。")
            return
        initial_dir = str(Path(self.source_path).parent) if self.source_path and not str(self.source_path).startswith("camera:") else str(Path.cwd())
        image_path = filedialog.asksaveasfilename(
            title="保存标注图片",
            initialdir=initial_dir,
            initialfile="铜环检测结果.png",
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")],
        )
        if not image_path:
            return
        cv2.imwrite(image_path, self.current_detection.annotated_bgr)
        csv_path = str(Path(image_path).with_suffix(".csv"))
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "铜环编号", "组件编号", "X", "Y", "中心X", "中心Y", "角度deg",
                "外包宽度 px", "外包高度 px",
                "内环最窄线宽 px", "外环最窄线宽 px", "最小线宽 px",
                "内环线宽 um", "外环线宽 um", "最小线宽 um",
                "内环线宽有效 DPI", "外环线宽有效 DPI",
                "不规则比", "不规则最小值 px", "不规则最大值 px", "不规则标准差 px",
                "环带过细", "覆盖溢出", "宽窄异常", "超分倍数",
            ])
            for item in self.current_detection.results:
                writer.writerow([
                    item.index, item.component_index or "", item.x, item.y,
                    f"{item.cx:.1f}", f"{item.cy:.1f}", f"{item.angle_deg:.1f}",
                    item.w, item.h,
                    f"{item.inner_line_width_px:.1f}", f"{item.outer_line_width_px:.1f}", f"{item.min_line_width_px:.1f}",
                    f"{item.inner_width_um:.1f}", f"{item.outer_width_um:.1f}", f"{item.min_width_um:.1f}",
                    f"{item.inner_effective_dpi:.1f}", f"{item.outer_effective_dpi:.1f}",
                    f"{item.irregularity_ratio:.2f}", f"{item.irregularity_min_px:.1f}", f"{item.irregularity_max_px:.1f}",
                    f"{item.irregularity_std_px:.1f}", item.is_below_threshold, item.is_broken, item.is_irregular_edges,
                    item.sr_scale,
                ])
        messagebox.showinfo("导出完成", f"已保存标注图片与 CSV 结果。\n{image_path}")


def main() -> None:
    root = tk.Tk()
    InspectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
