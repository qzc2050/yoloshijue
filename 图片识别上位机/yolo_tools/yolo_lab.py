# -*- coding: utf-8 -*-
"""YOLO 实验台: crop chambers, label them, train the classifier and test it, all in one window."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import board_views as bv  # noqa: E402
import chamber_data as cd  # noqa: E402
import export_chamber_crops as ec  # noqa: E402
import motion_port  # noqa: E402
import project_config as cfg  # noqa: E402
from qc_host.vision.grab import CameraSession  # noqa: E402
from version import VERSION  # noqa: E402

MACHINE = cfg.machine()
BOARD = cfg.active_board()
CAPTURES = cd.YOLO_ROOT / "captures"
SCAN_ROOT = MACHINE.scan_root
ALIGN_RIGHT_MM = MACHINE.align_offset_mm
GUIDE_BG = "#eef4fb"


class ImageView(tk.Canvas):
    """Shows one BGR image scaled to fit. Zoomable views keep the zoom when the picture updates."""

    def __init__(self, master, max_up: float = 1.0, zoomable: bool = False, **kw) -> None:
        super().__init__(master, bg="#1e1e1e", highlightthickness=0, **kw)
        self.max_up = max_up
        self.zoomable = zoomable
        self.image: np.ndarray | None = None
        self.text = ""
        self.on_click = None
        self._photo = None
        self._scale = 1.0
        self._offset = (0, 0)
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._drag = None
        self._dragged = False
        self._click_job = None
        self._suppress_release = False
        self.align_mark = False
        self.bind("<Configure>", lambda _e: self._draw())
        if zoomable:
            self.configure(cursor="crosshair")
            self.bind("<ButtonPress-1>", self._press)
            self.bind("<B1-Motion>", self._drag_move)
            self.bind("<ButtonRelease-1>", self._release)
            self.bind("<Double-Button-1>", self._reset_zoom_event)
            self.bind("<Button-3>", self._right_zoom)
            self.bind("<MouseWheel>", self._wheel)
            self.bind("<Enter>", self._bind_wheel)
            self.bind("<Leave>", self._unbind_wheel)
        else:
            self.bind("<Button-1>", self._click)

    def show(self, image: np.ndarray | None, text: str = "") -> None:
        if image is None:
            self._zoom = 1.0
            self._pan = [0.0, 0.0]
        self.image, self.text = image, text
        self._draw()

    def zoom_by(self, factor: float, x: float | None = None, y: float | None = None) -> None:
        if not self.zoomable or self.image is None:
            return
        if x is None or y is None:
            x, y = self.winfo_width() / 2, self.winfo_height() / 2
        self._zoom_at(x, y, factor)

    def reset_zoom(self) -> None:
        self._cancel_click_zoom()
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._draw()

    def _fit_scale(self, width: int, height: int) -> float:
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        return min(cw / width, ch / height, self.max_up)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        old = self._zoom
        new = min(16.0, max(0.5, old * factor))
        if abs(new - old) < 1e-6:
            return
        fit = self._fit_scale(width, height)
        old_scale = fit * old
        ox = (cw - width * old_scale) / 2 + self._pan[0]
        oy = (ch - height * old_scale) / 2 + self._pan[1]
        img_x = (x - ox) / old_scale
        img_y = (y - oy) / old_scale
        self._zoom = new
        new_scale = fit * new
        self._pan[0] = x - img_x * new_scale - (cw - width * new_scale) / 2
        self._pan[1] = y - img_y * new_scale - (ch - height * new_scale) / 2
        self._clamp_pan()
        self._draw()

    def _clamp_pan(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        scale = self._fit_scale(width, height) * self._zoom
        disp_w, disp_h = width * scale, height * scale
        limit_x = max(0.0, (disp_w + cw) / 2 - 80)
        limit_y = max(0.0, (disp_h + ch) / 2 - 80)
        self._pan[0] = min(limit_x, max(-limit_x, self._pan[0]))
        self._pan[1] = min(limit_y, max(-limit_y, self._pan[1]))

    def _draw(self) -> None:
        self.delete("all")
        cw, ch = max(1, self.winfo_width()), max(1, self.winfo_height())
        if self.image is None:
            self.create_text(cw // 2, ch // 2, text=self.text, fill="#aaaaaa", font=("Microsoft YaHei UI", 13), width=cw - 40)
            return
        height, width = self.image.shape[:2]
        fit = self._fit_scale(width, height)
        scale = fit if not self.zoomable else fit * self._zoom
        if self.zoomable:
            self._clamp_pan()
            ox = (cw - width * scale) / 2 + self._pan[0]
            oy = (ch - height * scale) / 2 + self._pan[1]
            x0 = max(0, int((0 - ox) / scale))
            y0 = max(0, int((0 - oy) / scale))
            x1 = min(width, int(np.ceil((cw - ox) / scale)))
            y1 = min(height, int(np.ceil((ch - oy) / scale)))
        else:
            ox, oy = (cw - width * scale) / 2, (ch - height * scale) / 2
            x0, y0, x1, y1 = 0, 0, width, height
        self._scale, self._offset = scale, (ox, oy)
        if x1 <= x0 or y1 <= y0:
            return
        crop = self.image[y0:y1, x0:x1]
        dw = max(1, int(round((x1 - x0) * scale)))
        dh = max(1, int(round((y1 - y0) * scale)))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        rgb = cv2.cvtColor(cv2.resize(crop, (dw, dh), interpolation=interp), cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.create_image(ox + x0 * scale, oy + y0 * scale, anchor="nw", image=self._photo)
        if self.zoomable:
            self.create_text(10, 8, anchor="nw", fill="#f2f2f2", font=("Microsoft YaHei UI", 11),
                             text=f"{self._zoom * 100:.0f}%    左键放大    右键缩小    按住拖动    双击看整张    滚轮也可以")
        if self.align_mark:
            self._draw_align_mark(width, height, scale, ox, oy)

    def align_center(self, width: int, height: int) -> tuple[float, float]:
        """Yellow frame sits 20 mm right of the image center so the ring light reaches the left edge."""
        return width / 2 + ALIGN_RIGHT_MM * bv.PX_PER_MM, height / 2

    def focus_align_mark(self) -> None:
        """Zoom onto the shifted frame. A 5x view of the image center would leave it off screen."""
        self._zoom = 5.0
        if self.image is None:
            self._pan = [0.0, 0.0]
            return
        height, width = self.image.shape[:2]
        scale = self._fit_scale(width, height) * self._zoom
        mark_x, mark_y = self.align_center(width, height)
        self._pan[0] = (width / 2 - mark_x) * scale
        self._pan[1] = (height / 2 - mark_y) * scale

    def _draw_align_mark(self, width: int, height: int, scale: float, ox: float, oy: float) -> None:
        """Fixed obround of the first horizontal chamber, 20 mm right of the frame center."""
        # Bright copper ring on chamber_template.png is 514 x 147 px at 44.2 px/mm.
        half_w, half_h = 514 / 2, 147 / 2
        cx, cy = self.align_center(width, height)

        def canvas_xy(px: float, py: float) -> tuple[float, float]:
            return ox + px * scale, oy + py * scale

        left, top = canvas_xy(cx - half_w, cy - half_h)
        right, bottom = canvas_xy(cx + half_w, cy + half_h)
        cap = bottom - top
        radius = cap / 2
        color = "#ffcc00"
        thickness = 2
        # Only the outer semicircles. Full circles cover the copper and are hard to match.
        self.create_arc(left, top, left + cap, bottom, start=90, extent=180, style="arc", outline=color, width=thickness)
        self.create_arc(right - cap, top, right, bottom, start=270, extent=180, style="arc", outline=color, width=thickness)
        self.create_line(left + radius, top, right - radius, top, fill=color, width=thickness)
        self.create_line(left + radius, bottom, right - radius, bottom, fill=color, width=thickness)
        mx, my = canvas_xy(cx, cy)
        arm = 18 * scale
        self.create_line(mx - arm, my, mx + arm, my, fill=color, width=thickness)
        self.create_line(mx, my - arm, mx, my + arm, fill=color, width=thickness)
        self.create_text(mx, top - 8, anchor="s", fill=color, font=("Microsoft YaHei UI", 12),
                         text="黄框在画面中心右边 20 mm，把铜环外沿重合到黄框")

    def _click(self, event) -> None:
        if self.on_click and self.image is not None:
            self.on_click((event.x - self._offset[0]) / self._scale, (event.y - self._offset[1]) / self._scale)

    def _press(self, event) -> None:
        self._drag = (event.x, event.y)
        self._dragged = False

    def _drag_move(self, event) -> None:
        if self._drag is None or self.image is None:
            return
        dx, dy = event.x - self._drag[0], event.y - self._drag[1]
        if abs(dx) + abs(dy) > 5:
            self._dragged = True
            self._cancel_click_zoom()
        if self._dragged:
            self._pan[0] += dx
            self._pan[1] += dy
            self._drag = (event.x, event.y)
            self._clamp_pan()
            self._draw()

    def _release(self, event) -> None:
        if self._suppress_release:
            self._suppress_release = False
            self._drag = None
            return
        if self.image is not None and not self._dragged:
            point = (event.x, event.y)
            self._cancel_click_zoom()
            self._click_job = self.after(220, lambda: self._finish_click(point))
        self._drag = None

    def _finish_click(self, point) -> None:
        self._click_job = None
        if self.image is None:
            return
        self._zoom_at(point[0], point[1], 1.5)

    def _cancel_click_zoom(self) -> None:
        if self._click_job is not None:
            self.after_cancel(self._click_job)
            self._click_job = None

    def _right_zoom(self, event) -> None:
        self._cancel_click_zoom()
        self._zoom_at(event.x, event.y, 1 / 1.5)

    def _reset_zoom_event(self, _event) -> None:
        self._suppress_release = True
        self.reset_zoom()

    def _wheel(self, event) -> None:
        if event.delta == 0 or self.image is None:
            return
        self._cancel_click_zoom()
        self._zoom_at(event.x, event.y, 1.15 if event.delta > 0 else 1 / 1.15)

    def _bind_wheel(self, _event) -> None:
        self.bind_all("<MouseWheel>", self._wheel)

    def _unbind_wheel(self, _event) -> None:
        self.unbind_all("<MouseWheel>")


def guide(parent, text: str) -> tk.Label:
    label = tk.Label(parent, text=text, justify="left", anchor="w", bg=GUIDE_BG, fg="#1a3d6d",
                     font=("Microsoft YaHei UI", 11), padx=12, pady=8, wraplength=1500)
    label.pack(fill="x", padx=8, pady=(8, 4))
    return label


def log_box(parent, height: int = 10, expand: bool = True) -> tk.Text:
    frame = ttk.Frame(parent)
    frame.pack(fill="both" if expand else "x", expand=expand, pady=(6, 0))
    text = tk.Text(frame, height=height, wrap="word", font=("Microsoft YaHei UI", 10), state="disabled")
    bar = ttk.Scrollbar(frame, command=text.yview)
    text.configure(yscrollcommand=bar.set)
    bar.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    return text


def append(text: tk.Text, line: str) -> None:
    text.configure(state="normal")
    text.insert("end", line + "\n")
    text.see("end")
    text.configure(state="disabled")


class YoloLab:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"YOLO 实验台 — 电离室质检 v{VERSION}")
        root.geometry("1500x930")
        try:
            root.state("zoomed")
        except tk.TclError:
            pass
        root.option_add("*Font", ("Microsoft YaHei UI", 11))
        style = ttk.Style()
        style.configure(".", font=("Microsoft YaHei UI", 11))
        style.configure("Big.TButton", font=("Microsoft YaHei UI", 13, "bold"), padding=8)
        style.configure("Treeview", rowheight=24)

        self.q: queue.Queue = queue.Queue()
        self.board = BOARD
        self.side_labels = {side.label: side.id for side in self.board.sides.values()}
        self.train_proc: subprocess.Popen | None = None

        self.session = CameraSession()
        self.motion = motion_port.MotionLink()
        self.tabs: dict[str, ttk.Frame] = {}
        self._cap_serial = -1
        self._cap_shown_at = 0.0
        self._cap_last_shot = 0.0
        self._shooting = False
        self._cap_session_count = 0
        self._closing = False
        self._scanning = False
        self._scan_cancel = threading.Event()

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True)
        self._build_capture_tab()
        self._build_board_tab()
        self._build_overlap_tab()
        self.nb.bind("<<NotebookTabChanged>>", self._tab_changed)
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(100, self._poll)

    # ---------- plumbing ----------
    def _poll(self) -> None:
        try:
            while True:
                self.q.get_nowait()()
        except queue.Empty:
            pass
        self._refresh_capture_view()
        self.root.after(100, self._poll)

    def later(self, fn, *args) -> None:
        self.q.put(lambda: fn(*args))

    def run_bg(self, work, done, failed=None) -> None:
        def body() -> None:
            try:
                result = work()
            except Exception as exc:  # shown to the operator
                detail = traceback.format_exc()
                self.later(failed or self._bg_error, exc, detail)
                return
            self.later(done, result)
        threading.Thread(target=body, daemon=True).start()

    def _bg_error(self, exc: Exception, detail: str) -> None:
        messagebox.showerror("出错了", f"{exc}\n\n{detail[-1500:]}")

    def _current_tab(self):
        try:
            return self.nb.nametowidget(self.nb.select())
        except tk.TclError:
            return None

    def _tab_changed(self, _event=None) -> None:
        tab = self._current_tab()
        if tab is self.tabs.get("capture"):
            self._update_cap_count()
        else:
            view = getattr(self, "cap_view", None)
            if view is not None:
                view._unbind_wheel()

    def _close(self) -> None:
        if getattr(self, "_closing", False):
            os._exit(0)
        if self.train_proc and self.train_proc.poll() is None:
            if not messagebox.askyesno("正在训练", "训练还没结束，关闭窗口会中止训练。确定关闭吗？"):
                return
            self.train_proc.terminate()
        self._closing = True
        # Camera shutdown can wait on this window. Never block here, or the close button hangs.
        threading.Thread(target=lambda: (time.sleep(0.8), os._exit(0)), daemon=True).start()
        try:
            self.motion.close()
        except Exception:
            pass
        try:
            self.session.stop(wait=False)
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

    # ---------- capture ----------
    def _build_capture_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["capture"] = tab
        self.nb.add(tab, text="  拍照  ")

        serial = ttk.Frame(tab)
        serial.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(serial, text="串口").pack(side="left")
        self.port_pick = ttk.Combobox(serial, width=10)
        self.port_pick.pack(side="left", padx=(6, 0))
        ttk.Button(serial, text="刷新", command=self._fill_ports, width=6).pack(side="left", padx=(6, 0))
        self.port_button = ttk.Button(serial, text="打开串口", command=self._toggle_port)
        self.port_button.pack(side="left", padx=(6, 0))
        self.pos_label = ttk.Label(serial, text="位置：未知", width=24, font=("Microsoft YaHei UI", 13, "bold"))
        self.pos_label.pack(side="left", padx=(12, 0))
        self.estop_button = tk.Button(
            serial, text="急停", command=self._emergency_stop, bg="#c62828", fg="white",
            activebackground="#8e0000", activeforeground="white", font=("Microsoft YaHei UI", 13, "bold"),
            width=6, relief="raised", state="disabled",
        )
        self.estop_button.pack(side="left", padx=(12, 0))
        ttk.Label(serial, text="板面").pack(side="left", padx=(16, 0))
        self.side_pick = ttk.Combobox(serial, width=12, state="readonly", values=tuple(self.side_labels))
        self.side_pick.set(next(iter(self.side_labels)))
        self.side_pick.pack(side="left", padx=(6, 0))
        self.scan_button = ttk.Button(serial, text="扫描整板", style="Big.TButton", command=self._start_board_scan)
        self.scan_button.pack(side="left", padx=(18, 0))
        self.scan_stop = ttk.Button(serial, text="停止扫描", command=self._cancel_board_scan, state="disabled")
        self.scan_stop.pack(side="left", padx=6)

        commands = ttk.Frame(tab)
        commands.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(commands, text="命令").pack(side="left")
        self.cmd_entry = ttk.Entry(commands, width=28)
        self.cmd_entry.pack(side="left", padx=(6, 0))
        self.cmd_entry.bind("<Return>", lambda _e: self._send_serial())
        self.send_button = ttk.Button(commands, text="发送", command=self._send_serial)
        self.send_button.pack(side="left", padx=(6, 0))
        self.cmd_buttons = [self.send_button]
        for name in ("pos", "zero", "x+", "x-", "y+", "y-", "z+", "z-", "h"):
            button = ttk.Button(commands, text=name, width=5, command=lambda name=name: self._send_serial(name))
            button.pack(side="left", padx=(6, 0))
            self.cmd_buttons.append(button)
        self.zero_button = ttk.Button(commands, text="设为原点", command=self._set_origin)
        self.zero_button.pack(side="left", padx=(12, 0))
        self.home_button = ttk.Button(commands, text="回到原点", command=self._go_home)
        self.home_button.pack(side="left", padx=(6, 0))

        self.serial_log = log_box(tab, 6, expand=False)

        jog = ttk.Frame(tab)
        jog.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Label(jog, text="每次走").pack(side="left")
        self.jog_step = tk.StringVar(value="0.10")
        for label, value in (("0.05", "0.05"), ("0.10", "0.10"), ("0.20", "0.20"), ("1", "1"), ("5", "5"), ("10", "10"), ("50 mm", "50")):
            ttk.Radiobutton(jog, text=label, variable=self.jog_step, value=value).pack(side="left", padx=(8, 0))
        self.jog_buttons = [self.zero_button, self.home_button, *self.cmd_buttons]
        for text, axis, sign in (("X−", "x", -1), ("X+", "x", 1), ("Y−", "y", -1), ("Y+", "y", 1)):
            button = ttk.Button(jog, text=text, width=5, command=lambda axis=axis, sign=sign: self._nudge(axis, sign))
            button.pack(side="left", padx=(10 if text == "X−" else 4, 0))
            self.jog_buttons.append(button)
        self._jogging = False
        self._fill_ports()

        row = ttk.Frame(tab)
        row.pack(fill="x", padx=8, pady=(6, 0))
        self.cap_connect = ttk.Button(row, text="连接相机", style="Big.TButton", command=self._connect_camera)
        self.cap_connect.pack(side="left")
        self.cap_disconnect = ttk.Button(row, text="断开相机", command=self._disconnect_camera)
        self.cap_disconnect.pack(side="left", padx=6)
        self.align_mark = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="对齐第一颗", variable=self.align_mark,
                        command=self._toggle_align).pack(side="left", padx=(16, 0))
        self.cap_status = ttk.Label(row, text="相机未连接")
        self.cap_status.pack(side="left", padx=10)
        ttk.Button(row, text="训练模型", style="Big.TButton", command=self._open_train_model).pack(side="right")

        self.cap_bright = ttk.Label(tab, text="", font=("Microsoft YaHei UI", 11))
        self.cap_bright.pack(anchor="w", padx=12, pady=(4, 0))

        self.cap_view = ImageView(tab, zoomable=True)
        self.cap_view.pack(fill="both", expand=True, padx=8, pady=6)
        self.cap_view.show(None, "")

        stored = ttk.Frame(tab)
        stored.pack(fill="x", padx=8, pady=(0, 6))
        stored_row = ttk.Frame(stored)
        stored_row.pack(fill="x")
        ttk.Label(stored_row, text="板号/备注").pack(side="left")
        self.cap_note = ttk.Entry(stored_row, width=24)
        self.cap_note.pack(side="left", padx=(6, 0))
        self.cap_shoot = ttk.Button(stored_row, text="拍一张", command=self._shoot)
        self.cap_shoot.pack(side="left", padx=(8, 0))
        ttk.Button(stored_row, text="打开照片文件夹", command=self._open_capture_folder).pack(side="left", padx=(8, 0))
        self.cap_count = ttk.Label(stored_row, text="")
        self.cap_count.pack(side="left", padx=(12, 0))
        self.cap_log = log_box(stored, 5, expand=False)
        self._set_cap_buttons("off")
        self._update_cap_count()

    def _set_cap_buttons(self, mode: str) -> None:
        connect = "disabled" if mode in ("connecting", "on", "saving") else "normal"
        disconnect = "normal" if mode in ("on", "saving") else "disabled"
        shoot = "normal" if mode == "on" else "disabled"
        self.cap_connect.configure(state=connect)
        self.cap_disconnect.configure(state=disconnect)
        self.cap_shoot.configure(state=shoot)

    def _toggle_align(self) -> None:
        on = bool(self.align_mark.get())
        self.cap_view.align_mark = on
        if on:
            self.cap_view.focus_align_mark()
            self.cap_status.configure(
                text=f"黄框在画面中心右边 {ALIGN_RIGHT_MM:g} mm。先摆电路板让铜环外沿贴上黄框，再用 0.05 或 0.10 mm 微调。"
            )
        self.cap_view._draw()

    def _frame_corr(self, before: np.ndarray, after: np.ndarray) -> float:
        def norm(image: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (480, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
            return (small - small.mean()) / (float(small.std()) + 1e-6)

        return float((norm(before) * norm(after)).mean())

    def _wait_settled_frame(self) -> np.ndarray:
        time.sleep(1.0)
        serial0, _, _, _ = self.session.peek()
        deadline = time.perf_counter() + 4.0
        while time.perf_counter() < deadline:
            serial, frame, _, _ = self.session.peek()
            if frame is not None and serial >= serial0 + 2:
                return frame.copy()
            time.sleep(0.05)
        raise OSError("到位后没有新画面，扫描已停")

    def _save_scan_frame(self, path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise OSError(f"保存失败 {path.name}")
        buf.tofile(str(path))

    def _set_scan_busy(self, busy: bool) -> None:
        self._scanning = busy
        self.scan_button.configure(state="disabled" if busy else "normal")
        self.side_pick.configure(state="disabled" if busy else "readonly")
        self.scan_stop.configure(state="normal" if busy else "disabled")
        self.port_button.configure(state="disabled" if busy else "normal")
        for button in self.jog_buttons:
            button.configure(state="disabled" if busy else "normal")
        if not busy and self.session.connected:
            self.cap_shoot.configure(state="normal")

    def _cancel_board_scan(self) -> None:
        if self._scanning:
            self._scan_cancel.set()
            self.cap_status.configure(text="这一点走完并拍完后停止，不再去下一点。")

    def _selected_side(self) -> str:
        return self.side_labels.get(self.side_pick.get(), next(iter(self.side_labels)))

    def _start_board_scan(self) -> None:
        if self._scanning or self._jogging:
            return
        if not self.motion.is_open:
            self.cap_status.configure(text="先点「打开串口」，再扫描整板。")
            return
        if not self.session.connected:
            self.cap_status.configure(text="先点「连接相机」，再扫描整板。")
            return
        side = self._selected_side()
        side_profile = self.board.side(side)
        points = side_profile.scan_points
        if not points:
            self.cap_status.configure(text=f"{self.board.name} 的 {side_profile.label} 没有扫描点，检查板型配置。")
            return
        point_count = len(points)
        first = "D 列第一颗" if side == "back" else "第一颗"
        if not messagebox.askyesno(
            "扫描整板",
            f"这次扫 {self.board.name} {side_profile.label}。平台按 {point_count} 个点自己走，每点拍一张，"
            f"大约 {side_profile.scan_minutes} 分钟。\n"
            f"原点是黄框对准{first}后设下的位置，画面中心在这颗左边 {ALIGN_RIGHT_MM:g} mm。"
            "只在当前位置离原点 2 mm 以内才出发。\n"
            "照片单独放在这次的文件夹里。确定开始？",
        ):
            return
        note = self.cap_note.get().strip()
        settings = self.session.settings
        camera = {}
        if settings is not None:
            camera = {
                "exposure_us": settings.exposure_us,
                "gain_db": settings.gain_db,
                "white_balance_auto": settings.white_balance_auto,
            }
        self._scan_cancel.clear()
        self._set_scan_busy(True)
        self.cap_shoot.configure(state="disabled")
        self.cap_status.configure(text="正在核对原点…")
        append(self.cap_log, "开始扫描整板：先读位置，确认在原点附近才移动。")

        def work() -> dict:
            pos = self.motion.read_xy()
            if pos[0] < -2 or pos[1] < -2 or pos[0] > 2 or pos[1] > 2:
                raise OSError(f"当前位置 x={pos[0]:.3f} y={pos[1]:.3f} 不在原点附近，没有移动")
            folder = SCAN_ROOT / time.strftime(f"{self.board.id}_{side}_%Y%m%d_%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "board.txt").write_text(self.board.id, encoding="utf-8")
            (folder / "side.txt").write_text(side, encoding="utf-8")
            meta = {
                "board": self.board.id,
                "board_name": self.board.name,
                "side": side,
                "side_label": side_profile.label,
                "board_note": note,
                "software": VERSION,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "origin": {"x": round(pos[0], 3), "y": round(pos[1], 3), "source": "operator_zero"},
                "align": {"offset_mm": ALIGN_RIGHT_MM},
                "camera": camera,
                "machine": {"px_per_mm": MACHINE.px_per_mm, "scan_points": point_count, "port": MACHINE.port},
            }
            log_path = folder / "scan_log.txt"
            previous: np.ndarray | None = None
            saved = 0

            def log(line: str) -> None:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(time.strftime("%H:%M:%S ") + line + "\n")

            def write_meta(status: str) -> None:
                payload = {**meta, "status": status, "photos": saved,
                           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                (folder / "scan_meta.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )

            log(
                f"start board={self.board.id} side={side} software={VERSION} "
                f"exposure={camera.get('exposure_us')} gain={camera.get('gain_db')} "
                f"x={pos[0]:.3f} y={pos[1]:.3f} note={note}"
            )
            write_meta("running")
            self.later(append, self.cap_log, f"原点附近 x={pos[0]:.3f} y={pos[1]:.3f}，照片放到 {folder}")
            try:
                for index, (x_mm, y_mm) in enumerate(points, start=1):
                    if self._scan_cancel.is_set():
                        log(f"stopped after {saved}")
                        write_meta("stopped")
                        return {"folder": folder, "count": saved, "done": False}
                    self.later(
                        lambda text: self.cap_status.configure(text=text),
                        f"正在去第 {index}/{point_count} 点，X {x_mm} mm，Y {y_mm} mm",
                    )
                    log(f"move {index}/{point_count} x={x_mm} y={y_mm}")
                    pos = self.motion.move_abs(float(x_mm), float(y_mm), pos[0], pos[1])
                    self.later(self._show_pos, pos)
                    log(f"arrived x={pos[0]:.3f} y={pos[1]:.3f}")
                    image = self._wait_settled_frame()
                    path = folder / f"line_{index:02d}_x{x_mm:g}_y{y_mm:g}.jpg"
                    if previous is not None:
                        corr = self._frame_corr(previous, image)
                        log(f"corr {corr:.3f}")
                        if corr > 0.95:
                            raise OSError(f"第 {index} 张和上一张几乎一样（相关 {corr:.3f}），扫描已停在这一点，没有继续走")
                    self._save_scan_frame(path, image)
                    previous = image
                    saved += 1
                    log(f"saved {path.name}")
                    self.later(append, self.cap_log, f"已拍 {index}/{point_count}　{path.name}")
            except Exception:
                write_meta("error")
                raise
            log(f"scan done {point_count}")
            write_meta("done")
            return {"folder": folder, "count": saved, "done": True}

        def done(result: dict) -> None:
            self._set_scan_busy(False)
            folder = result["folder"]
            count = result["count"]
            if result["done"]:
                self.cap_status.configure(text=f"{point_count} 张已保存，正在按光绘编号识别…")
                append(self.cap_log, f"扫描完成，{point_count} 张在 {folder}。开始识别 {self.board.name} {side_profile.label}。")
                self.nb.select(self.tabs["board"])
                self.board_panel.identify(Path(folder), lambda payload, exc, folder=folder: self._scan_identified(folder, payload, exc))
            else:
                self.cap_status.configure(text=f"扫描已停止，已拍 {count} 张，在 {folder}")
                append(self.cap_log, f"扫描停止，已拍 {count} 张，在 {folder}")

        def failed(exc: Exception, detail: str) -> None:
            self._set_scan_busy(False)
            self._refresh_pos()
            if isinstance(exc, motion_port.MotionStopped):
                self.cap_status.configure(text=f"扫描已急停。{exc}。这次的照片不完整，不要拿去识别。")
                append(self.cap_log, f"扫描急停：{exc}")
                return
            self.cap_status.configure(text="扫描停了，平台留在当前位置")
            append(self.cap_log, f"扫描停了：{exc}")
            self._bg_error(exc, detail)

        self.run_bg(work, done, failed)

    def _scan_identified(self, folder: Path, payload: dict | None, exc: Exception | None) -> None:
        if exc is not None or payload is None:
            self.cap_status.configure(text=f"照片已保存在 {folder}，识别没有完成")
            append(self.cap_log, f"识别没有完成：{exc}")
            messagebox.showerror("识别没有完成", f"扫描照片已经保存在\n{folder}\n\n{exc}")
            return
        total = payload.get("gerber") or 0
        text = f"识别 {payload['matched']}/{total}，缺 {payload['missing']}，多余框 {payload['extras']}"
        self.cap_status.configure(text=text)
        append(self.cap_log, text)
        messagebox.showinfo("识别完成", f"{text}\n误差中位 {payload['median_error_mm']} mm\n{folder}")

    def _fill_ports(self) -> None:
        if self.motion.is_open:
            return
        ports = motion_port.list_ports()
        self.port_pick["values"] = ports
        current = self.port_pick.get().strip()
        if current in ports:
            return
        if "COM7" in ports:
            self.port_pick.set("COM7")
        elif ports:
            self.port_pick.set(ports[0])

    def _debug(self, line: str) -> None:
        append(self.serial_log, line)

    def _send_serial(self, line: str | None = None) -> None:
        if line is None:
            line = self.cmd_entry.get()
        line = line.strip()
        if not line:
            return
        if line.lower() in ("s", "stop"):
            self._debug(f">> {line}")
            self._emergency_stop()
            return
        if self._jogging or self._scanning:
            self._debug("正在动，这条命令没发")
            return
        if not self.motion.is_open:
            self._debug("串口没开")
            return
        self._debug(f">> {line}")
        self.cmd_entry.delete(0, "end")

        def work() -> str:
            return self.motion.transact(line)

        def done(reply: str) -> None:
            self._debug(reply or "(无回应)")
            found = motion_port.DONE_POS.findall(reply) or motion_port.POS.findall(reply)
            if found:
                self._show_pos((float(found[-1][0]), float(found[-1][1])))
            else:
                self._refresh_pos()

        def failed(exc: Exception, detail: str) -> None:
            self._debug(str(exc))
            self._refresh_pos()
            if not isinstance(exc, motion_port.MotionStopped):
                self._bg_error(exc, detail)

        self._stage_job(f"命令 {line}", work, lambda _pos: f"命令完成 {line}", "命令没有完成", on_done=done, on_failed=failed)

    def _toggle_port(self) -> None:
        if self._jogging or self._scanning:
            return
        opening = not self.motion.is_open
        if opening:
            try:
                self.motion.set_port(self.port_pick.get())
            except OSError as exc:
                self.cap_status.configure(text=str(exc))
                self._debug(str(exc))
                return
        self._jogging = True
        self.port_button.configure(state="disabled")
        self.cap_status.configure(text="正在打开串口…" if opening else "正在关闭串口…")

        def work() -> str:
            if opening:
                return self.motion.connect()
            self.motion.close()
            return ""

        def done(message: str) -> None:
            self._jogging = False
            self.port_button.configure(state="normal")
            if self.motion.is_open:
                self.port_button.configure(text="关闭串口")
                self.estop_button.configure(state="normal")
                self.port_pick.configure(state="disabled")
                self.cap_status.configure(text=f"串口已打开 {self.motion.port}")
                self._debug(f"打开 {self.motion.port}  {message}")
                append(self.cap_log, f"串口已打开 {message}")
                found = motion_port.POS.search(message)
                if found:
                    self._show_pos((float(found.group(1)), float(found.group(2))))
            else:
                self.port_button.configure(text="打开串口")
                self.port_pick.configure(state="normal")
                self.estop_button.configure(state="disabled")
                self._show_pos(None)
                self.cap_status.configure(text="串口已关闭")
                self._debug("串口已关闭")
                append(self.cap_log, "串口已关闭。")

        def failed(exc: Exception, detail: str) -> None:
            self._jogging = False
            self.port_button.configure(state="normal", text="打开串口" if not self.motion.is_open else "关闭串口")
            self.estop_button.configure(state="normal" if self.motion.is_open else "disabled")
            self.cap_status.configure(text="串口没有打开")
            self._bg_error(exc, detail)

        self.run_bg(work, done, failed)

    def _show_pos(self, pos: tuple[float, float] | None) -> None:
        if pos is None:
            self.pos_label.configure(text="位置：未知")
        else:
            self.pos_label.configure(text=f"X {pos[0]:.3f}   Y {pos[1]:.3f}")

    def _stage_job(self, busy_text: str, work, done_text, fail_text: str, on_done=None, on_failed=None) -> None:
        """Run one stage move in the background with the move buttons locked."""
        if self._jogging or self._scanning:
            return
        if not self.motion.is_open:
            self.cap_status.configure(text="先点「打开串口」。")
            return
        self._jogging = True
        self.port_button.configure(state="disabled")
        for button in self.jog_buttons:
            button.configure(state="disabled")
        self.cap_status.configure(text=busy_text)

        def unlock() -> None:
            self._jogging = False
            self.port_button.configure(state="normal")
            for button in self.jog_buttons:
                button.configure(state="normal")

        def done(result) -> None:
            unlock()
            if on_done is not None:
                on_done(result)
                return
            self._show_pos(result)
            text = done_text(result)
            self.cap_status.configure(text=text)
            append(self.cap_log, text)

        def failed(exc: Exception, detail: str) -> None:
            unlock()
            if on_failed is not None:
                on_failed(exc, detail)
                return
            self._refresh_pos()
            if isinstance(exc, motion_port.MotionStopped):
                self.cap_status.configure(text=str(exc))
                self._debug(str(exc))
                append(self.cap_log, str(exc))
                return
            self.cap_status.configure(text=fail_text)
            self._bg_error(exc, detail)

        self.run_bg(work, done, failed)

    def _refresh_pos(self) -> None:
        if not self.motion.is_open:
            self._show_pos(None)
            return

        def work() -> tuple[float, float]:
            return self.motion.read_xy()

        self.run_bg(work, self._show_pos, lambda _exc, _detail: self._show_pos(None))

    def _nudge(self, axis: str, sign: int) -> None:
        step = float(self.jog_step.get()) * sign
        self._stage_job(
            f"正在走 {axis.upper()} {step:+.2f} mm…",
            lambda: self.motion.nudge(axis, step),
            lambda pos: f"{axis.upper()} {step:+.2f} mm 已到位",
            "没有动",
        )

    def _set_origin(self) -> None:
        if self._jogging or self._scanning:
            return
        if not self.motion.is_open:
            self.cap_status.configure(text="先点「打开串口」。")
            return
        if not messagebox.askyesno(
            "设为原点",
            "黄框的十字已经和第一颗电离室的中心重合了吗？黄框在画面中心右边 20 mm。\n\n"
            "设为原点后，平台现在的位置记为 X 0、Y 0。平台不会动。",
        ):
            return
        self._stage_job(
            "正在设原点…",
            self.motion.zero,
            lambda pos: "已设为原点 X 0、Y 0。现在可以扫描整板。",
            "原点没有设上",
        )

    def _go_home(self) -> None:
        if self._jogging or self._scanning:
            return
        if not self.motion.is_open:
            self.cap_status.configure(text="先点「打开串口」。")
            return
        if not messagebox.askyesno(
            "回到原点",
            "平台将走回 X 0、Y 0，先走 X 再走 Y，最远大约 1 分钟。\n\n"
            "断过电或急停过的话，原点已经不准，回来后要重新对齐第一颗。\n"
            "路上不要把手伸进去。确定？",
        ):
            return
        self._stage_job(
            "正在回原点…有异常按「急停」",
            lambda: self.motion.go_to(0.0, 0.0),
            lambda pos: "已回到原点。打开「对齐第一颗」核对十字是否还在第一颗中心。",
            "没有回到原点",
        )

    def _emergency_stop(self) -> None:
        if not self.motion.is_open:
            return
        self._scan_cancel.set()
        self.cap_status.configure(text="急停已发出")
        append(self.cap_log, "急停")
        threading.Thread(target=self.motion.request_stop, daemon=True).start()

    def _capture_total(self) -> int:
        if not CAPTURES.exists():
            return 0
        return sum(1 for path in CAPTURES.iterdir() if path.suffix.lower() in ec.IMAGE_EXT)

    def _update_cap_count(self) -> None:
        self.cap_count.configure(text=f"这次已拍 {self._cap_session_count} 张　　文件夹里一共 {self._capture_total()} 张")

    def _capture_path(self, note: str) -> Path:
        cleaned = "".join(ch for ch in note.strip() if ch not in '\\/:*?"<>|').strip()[:40]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        stem = f"{stamp}_{millis:03d}" + (f"_{cleaned}" if cleaned else "")
        return CAPTURES / f"{stem}.jpg"

    def _light_word(self, mean: float) -> str:
        if mean < 45:
            return "偏暗，这种也要拍一些"
        if mean > 190:
            return "偏亮，留意发白过曝"
        return "亮度正常"

    def _connect_camera(self) -> None:
        if self.session.connected:
            return
        self._set_cap_buttons("connecting")
        self.cap_status.configure(text="正在连接相机…")

        def failed(exc: Exception, detail: str) -> None:
            self._set_cap_buttons("off")
            self.cap_status.configure(text="连接失败")
            messagebox.showerror("连接相机失败", f"{exc}\n\n请确认：相机已上电、网线插好，并且 MVS 已经关闭。\n\n{detail[-800:]}")

        self.run_bg(self.session.start, self._camera_connected, failed)

    def _camera_connected(self, info: str) -> None:
        self._set_cap_buttons("on")
        self.cap_status.configure(text=f"已连接：{info}。挪好平台后点「拍一张」，或按空格。")
        append(self.cap_log, f"相机已连接：{info}")

    def _disconnect_camera(self) -> None:
        if self._scanning:
            self.cap_status.configure(text="扫描还没结束，先点「停止扫描」。")
            return
        self._set_cap_buttons("connecting")
        self.cap_status.configure(text="正在断开…")

        def done(_result) -> None:
            self._cap_serial = -1
            self._set_cap_buttons("off")
            self.cap_status.configure(text="相机已断开。")
            self.cap_bright.configure(text="连接后这里显示亮度。偏暗也可以拍。")
            self.cap_view.show(None, "相机已断开。点「连接相机」再拍。")
            append(self.cap_log, "相机已断开。")

        def failed(exc: Exception, detail: str) -> None:
            self._set_cap_buttons("off")
            self._bg_error(exc, detail)

        self.run_bg(self.session.stop, done, failed)

    def _shoot(self) -> None:
        if self._shooting or self._scanning or not self.session.connected:
            return
        if time.perf_counter() - self._cap_last_shot < 0.4:
            return
        shot = self.session.take()
        if shot is None:
            self.cap_status.configure(text="画面还没出来，等一秒再拍。")
            return
        image, center, _overall = shot
        note = self.cap_note.get()
        self._cap_last_shot = time.perf_counter()
        self._shooting = True
        self._set_cap_buttons("saving")
        self.cap_status.configure(text="正在保存…")

        def work():
            path = self._capture_path(note)
            cd.write_image(path, image)
            return path, center

        def done(result) -> None:
            path, bright = result
            self._shooting = False
            self._cap_session_count += 1
            self._set_cap_buttons("on" if self.session.connected else "off")
            self._update_cap_count()
            self.cap_status.configure(text=f"已保存 {path.name}。可以继续挪平台再拍。")
            append(self.cap_log, f"已保存 {path.name}　中间亮度 {bright:.0f}（{self._light_word(bright)}）")

        def failed(exc: Exception, detail: str) -> None:
            self._shooting = False
            self._set_cap_buttons("on" if self.session.connected else "off")
            self.cap_status.configure(text="保存失败")
            self._bg_error(exc, detail)

        self.run_bg(work, done, failed)

    def _refresh_capture_view(self) -> None:
        if self._current_tab() is not self.tabs.get("capture"):
            return
        error = self.session.consume_error()
        if error:
            self._shooting = False
            self._cap_serial = -1
            self._set_cap_buttons("off")
            self.cap_status.configure(text="相机中断了")
            self.cap_view.show(None, "相机中断了。检查网线后，再点「连接相机」。")
            append(self.cap_log, f"相机中断：{error}")
            return
        if not self.session.connected:
            return
        serial, frame, center, overall = self.session.peek()
        now = time.perf_counter()
        if frame is None or serial == self._cap_serial or now - self._cap_shown_at < 0.2:
            return
        self._cap_serial = serial
        self._cap_shown_at = now
        self.cap_view.show(frame)
        self.cap_bright.configure(
            text=f"画面中间亮度 {center:.0f}（{self._light_word(center)}）　　整张平均 {overall:.0f}"
        )

    def _open_train_model(self) -> None:
        window = getattr(self, "_train_window", None)
        if window is not None and window.winfo_exists():
            window.lift()
            window.focus_force()
            return
        from train_model import open_train_window

        self._train_window = open_train_window(self)

    def _open_capture_folder(self) -> None:
        CAPTURES.mkdir(parents=True, exist_ok=True)
        os.startfile(CAPTURES)  # noqa: S606

    # ---------- board views ----------
    def _build_board_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["board"] = tab
        self.nb.add(tab, text="  整板对照  ")
        self.board_panel = bv.GerberBoardPanel(tab)
        self.board_panel.pack(fill="both", expand=True)

    def _build_overlap_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["overlap"] = tab
        self.nb.add(tab, text="  两张核对  ")
        self.overlap_panel = bv.OverlapPanel(tab)
        self.overlap_panel.pack(fill="both", expand=True)

def main() -> None:
    root = tk.Tk()
    try:
        YoloLab(root)
    except Exception:
        messagebox.showerror("启动失败", traceback.format_exc())
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
