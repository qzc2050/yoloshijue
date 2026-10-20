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
from tkinter import filedialog, messagebox, ttk

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
import hole_assist as ha  # noqa: E402
from qc_host.vision.grab import CameraSession  # noqa: E402

CAPTURES = cd.YOLO_ROOT / "captures"
SCAN_ROOT = Path(r"E:\xiangmu\shibie\qc_motion\scan_yolo")
BEAMCHECK_POINTS = (
    (5, 50), (5, 100), (5, 150), (5, 200),
    (61, 200), (61, 150), (61, 100), (61, 50),
    (117, 0), (117, 50), (117, 100), (117, 150), (117, 200), (117, 250),
    (173, 200), (173, 150), (173, 100), (173, 50),
    (229, 50), (229, 100), (229, 150), (229, 200),
    (285, 200), (285, 150), (285, 100), (285, 50),
)
ALIGN_RIGHT_MM = 20.0
UNSURE = 0.6
BOX_COLORS = {"hege": (80, 200, 80), "shenmo": (60, 60, 235), "quemo": (0, 210, 255), "kanbuqing": (200, 120, 40),
              "unsure": (230, 80, 230), None: (170, 170, 170)}
TREE_COLORS = {"hege": "#2e7d32", "shenmo": "#c62828", "quemo": "#b28704", "kanbuqing": "#e65100",
               "unsure": "#8e24aa", None: "#555555"}
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
        root.title("YOLO 实验台 — 电离室质检")
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
        self.template = ec.load_template()
        self.train_proc: subprocess.Popen | None = None
        self.model = None
        self.model_key = None

        self._keys_down: set[str] = set()
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
        root.bind("<KeyPress>", self._key)
        root.bind("<KeyRelease>", self._key_up)
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
        if tab is self.tabs.get("label"):
            self._reload_labels()
        elif tab is self.tabs.get("train"):
            self._refresh_train_counts()
        elif tab is self.tabs.get("test"):
            self._refresh_model_label()
        elif tab is self.tabs.get("capture"):
            self._update_cap_count()
        else:
            view = getattr(self, "cap_view", None)
            if view is not None:
                view._unbind_wheel()

    def _key(self, event) -> None:
        if event.keysym in self._keys_down:
            return
        self._keys_down.add(event.keysym)
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return
        tab = self._current_tab()
        if tab is self.tabs.get("capture"):
            return
        if tab is not self.tabs.get("label"):
            return
        keys = {"1": "hege", "2": "shenmo", "3": "quemo", "4": "kanbuqing"}
        if event.char in keys:
            self._set_label(keys[event.char])
        elif event.keysym in ("Right", "space"):
            self._step(1)
        elif event.keysym == "Left":
            self._step(-1)

    def _key_up(self, event) -> None:
        self._keys_down.discard(event.keysym)

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
        self.side_pick = ttk.Combobox(serial, width=12, state="readonly", values=("xbeam 正面", "xbeam 背面"))
        self.side_pick.set("xbeam 正面")
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
        self.cap_note = ttk.Entry(stored, width=24)
        self.cap_shoot = ttk.Button(stored, text="拍一张", command=self._shoot)
        self.cap_count = ttk.Label(stored, text="")
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
            self.cap_status.configure(text="黄框在画面中心右边 20 mm。先摆电路板让铜环外沿贴上黄框，再用 0.05 或 0.10 mm 微调。")
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
        return "back" if self.side_pick.get() == "xbeam 背面" else "front"

    def _start_board_scan(self) -> None:
        if self._scanning or self._jogging:
            return
        if not self.motion.is_open:
            self.cap_status.configure(text="先点「打开串口」，再扫描整板。")
            return
        if not self.session.connected:
            self.cap_status.configure(text="先点「连接相机」，再扫描整板。")
            return
        point_count = len(BEAMCHECK_POINTS)
        if point_count != 26:
            self.cap_status.configure(text="扫描点不是 26 个，没有移动。")
            return
        side = self._selected_side()
        side_name = "背面" if side == "back" else "正面"
        first = "D 列第一颗" if side == "back" else "第一颗"
        if not messagebox.askyesno(
            "扫描整板",
            f"这次扫 xbeam {side_name}。平台按 26 个点自己走，每点拍一张，大约 4 分钟。\n"
            f"原点是黄框对准{first}后设下的位置，画面中心在这颗左边 20 mm。只在当前位置离原点 2 mm 以内才出发。\n"
            "照片单独放在这次的文件夹里。确定开始？",
        ):
            return
        self._scan_cancel.clear()
        self._set_scan_busy(True)
        self.cap_shoot.configure(state="disabled")
        self.cap_status.configure(text="正在核对原点…")
        append(self.cap_log, "开始扫描整板：先读位置，确认在原点附近才移动。")

        def work() -> dict:
            pos = self.motion.read_xy()
            if pos[0] < -2 or pos[1] < -2 or pos[0] > 2 or pos[1] > 2:
                raise OSError(f"当前位置 x={pos[0]:.3f} y={pos[1]:.3f} 不在原点附近，没有移动")
            folder = SCAN_ROOT / time.strftime(f"beamcheck_{side}_%Y%m%d_%H%M%S")
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "side.txt").write_text(side, encoding="utf-8")
            log_path = folder / "scan_log.txt"
            previous: np.ndarray | None = None
            saved = 0

            def log(line: str) -> None:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(time.strftime("%H:%M:%S ") + line + "\n")

            log(f"start x={pos[0]:.3f} y={pos[1]:.3f}")
            self.later(append, self.cap_log, f"原点附近 x={pos[0]:.3f} y={pos[1]:.3f}，照片放到 {folder}")
            for index, (x_mm, y_mm) in enumerate(BEAMCHECK_POINTS, start=1):
                if self._scan_cancel.is_set():
                    log(f"stopped after {saved}")
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
                path = folder / f"line_{index:02d}_x{x_mm}_y{y_mm}.jpg"
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
            log(f"scan done {point_count}")
            return {"folder": folder, "count": saved, "done": True}

        def done(result: dict) -> None:
            self._set_scan_busy(False)
            folder = result["folder"]
            count = result["count"]
            if result["done"]:
                self.cap_status.configure(text=f"{point_count} 张已保存，正在按光绘编号识别…")
                append(self.cap_log, f"扫描完成，{point_count} 张在 {folder}。开始识别 xbeam {side_name}。")
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
            messagebox.showerror("识别没有完成", f"{len(BEAMCHECK_POINTS)} 张已经保存在\n{folder}\n\n{exc}")
            return
        total = payload.get("gerber") or 399
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

    def _send_captures_to_crop(self) -> None:
        if self._capture_total() == 0:
            messagebox.showinfo("提示", "还没有拍过照片。先连接相机，拍几张。")
            return
        folder = str(CAPTURES)
        if folder not in self.crop_list.get(0, "end"):
            self.crop_list.insert("end", folder)
        self.nb.select(self.tabs["crop"])
        append(self.crop_log, f"已加入刚拍的照片文件夹：{folder}。点左边的「开始裁图」，只会分析还没裁过的新照片。")
        if not any((cd.RAW / "_overview").glob("*.jpg")):
            self.overview_view.show(None, f"照片已放到左边列表（共 {self._capture_total()} 张）。\n点左边的「开始裁图」，裁完后这里才会画出找到的电离室。")

    # ---------- step 1: crop ----------
    def _build_crop_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["crop"] = tab
        self.nb.add(tab, text="  第 1 步：裁图  ")
        guide(tab, "第 1 步：把相机拍的整张照片，变成一颗颗电离室小图。\n"
                   "从拍照页点「拿去裁图」后，照片文件夹已经在左边列表里。再点左边的「开始裁图」。"
                   "已经裁过的照片会跳过，只分析新拍的。裁出来的小图保存到 "
                   f"{cd.RAW}。\n右边一开始是空的，裁完才会画出找到的电离室。框的颜色是现有规则程序的判定，只作参考。"
                   "没框到的一般是在画面边缘、太暗，或只拍到一半。")
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body, width=460)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Button(row, text="选择图片…", command=self._pick_files).pack(side="left")
        ttk.Button(row, text="选择文件夹…", command=self._pick_folder).pack(side="left", padx=4)
        ttk.Button(row, text="清空列表", command=lambda: self.crop_list.delete(0, "end")).pack(side="left")
        self.crop_list = tk.Listbox(left, height=10)
        self.crop_list.pack(fill="x", pady=6)

        opts = ttk.LabelFrame(left, text="设置（一般不用改）")
        opts.pack(fill="x", pady=4)
        self.threshold = tk.DoubleVar(value=0.5)
        ttk.Label(opts, text="找电离室的严格程度（越低找得越多，但可能把不是电离室的也框进来）").pack(anchor="w", padx=6)
        srow = ttk.Frame(opts)
        srow.pack(fill="x", padx=6)
        ttk.Scale(srow, from_=0.3, to=0.8, variable=self.threshold,
                  command=lambda _v: self.threshold_text.configure(text=f"{self.threshold.get():.2f}")).pack(side="left", fill="x", expand=True)
        self.threshold_text = ttk.Label(srow, text="0.50", width=5)
        self.threshold_text.pack(side="left")
        self.use_rule = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="同时跑一遍现有规则程序，按它的判定预先分好（每张图多花约 3 秒）",
                        variable=self.use_rule).pack(anchor="w", padx=6, pady=4)

        self.crop_btn = ttk.Button(left, text="开始裁图", style="Big.TButton", command=self._start_crop)
        self.crop_btn.pack(fill="x", pady=6)
        self.crop_log = log_box(left, 12)
        cd.RAW.mkdir(parents=True, exist_ok=True)
        self._use_capture_folder_if_list_empty()

        top = ttk.Frame(right)
        top.pack(fill="x")
        ttk.Label(top, text="查看找到的电离室：").pack(side="left")
        self.overview_pick = ttk.Combobox(top, state="readonly", width=60)
        self.overview_pick.pack(side="left")
        self.overview_pick.bind("<<ComboboxSelected>>", lambda _e: self._show_overview(self.overview_pick.get()))
        self.overview_view = ImageView(right)
        self.overview_view.pack(fill="both", expand=True, pady=(6, 0))
        self._refresh_overviews()

    def _pick_files(self) -> None:
        for f in filedialog.askopenfilenames(title="选择原图", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]):
            self.crop_list.insert("end", f)

    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择放原图的文件夹")
        if folder:
            self.crop_list.insert("end", folder)

    def _refresh_overviews(self, select_last: bool = False) -> None:
        folder = cd.RAW / "_overview"
        names = sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []
        self.overview_pick["values"] = names
        if names:
            pick = names[-1] if select_last or not self.overview_pick.get() else self.overview_pick.get()
            self.overview_pick.set(pick)
            self._show_overview(pick)
        else:
            self.overview_view.show(None, "右边这里要等裁完才有图。\n左边列表里已经有拍照文件夹的话，点左边的「开始裁图」。")

    def _show_overview(self, name: str) -> None:
        self.overview_view.show(cd.read_image(cd.RAW / "_overview" / name))

    def _use_capture_folder_if_list_empty(self) -> None:
        if self.crop_list.size() == 0 and self._capture_total() > 0:
            folder = str(CAPTURES)
            self.crop_list.insert("end", folder)
            append(self.crop_log, f"已放上拍照文件夹：{folder}。点「开始裁图」只处理还没裁过的新照片。")

    def _already_cropped(self, path: Path) -> bool:
        tag = f"{path.parent.name}_{path.stem}"
        return (cd.RAW / "_overview" / f"{tag}.jpg").exists()

    def _start_crop(self) -> None:
        self._use_capture_folder_if_list_empty()
        items = list(self.crop_list.get(0, "end"))
        files = ec.collect_images(items)
        if not files:
            messagebox.showinfo("提示", "先点「选择图片」或「选择文件夹」，把原图加进列表。拍照的原图在 E:\\yolo\\captures。")
            return
        pending = [path for path in files if not self._already_cropped(path)]
        skipped = len(files) - len(pending)
        if not pending:
            messagebox.showinfo("提示", f"这 {len(files)} 张都裁过了，不用再分析。去「第 2 步：标注」看还没标的。新拍的照片再来裁。")
            return
        self.crop_btn.configure(state="disabled", text="裁图中…")
        if skipped:
            append(self.crop_log, f"跳过已裁过的 {skipped} 张，开始处理新的 {len(pending)} 张…")
        else:
            append(self.crop_log, f"开始处理 {len(pending)} 张图…")
        files = pending
        threshold, use_rule = self.threshold.get(), self.use_rule.get()

        def log(line: str) -> None:
            self.later(append, self.crop_log, line.replace("done ", "完成 ").replace("chambers", "电离室").replace(
                "skip ", "跳过 ").replace("(not a full-resolution frame)", "（不是相机原始分辨率的图）"))

        self.run_bg(lambda: ec.export_files(files, cd.RAW, self.template, threshold, use_rule, log),
                    self._crop_done, self._crop_failed)

    def _crop_done(self, counts: dict) -> None:
        self.crop_btn.configure(state="normal", text="开始裁图")
        total = sum(counts.values())
        detail = "，".join(f"{cd.RULE_NAMES[k]} {v}" for k, v in counts.items() if v)
        append(self.crop_log, f"全部完成：共裁出 {total} 颗（规则判定：{detail}）。下一步去「第 2 步：标注」。")
        self._refresh_overviews(select_last=True)

    def _crop_failed(self, exc: Exception, detail: str) -> None:
        self.crop_btn.configure(state="normal", text="开始裁图")
        self._bg_error(exc, detail)

    # ---------- step 2: label ----------
    FILTERS = ["未标注", "全部", "已标：合格", "已标：渗墨", "已标：缺墨", "已标：看不清",
               "规则判：渗墨", "规则判：缺墨", "规则判：不能识别", "规则没找到"]

    def _build_label_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["label"] = tab
        self.nb.add(tab, text="  第 2 步：标注  ")
        guide(tab, "第 2 步：告诉电脑每颗电离室是什么。看大图，点按钮或按键盘数字键："
                   "1 合格　2 渗墨　3 缺墨　4 看不清（灯太暗、对焦糊、在画面边上）。标完自动跳到下一张，← → 键前后翻，随时可以回头改。\n"
                   "判定标准：渗墨 = 两圈铜环之间那条细缝（0.3 mm 隔离层）里进了黑；"
                   "缺墨 = 内圈铜环里面、该是黑色的地方露出白色或绿色；铜环本身被黑盖住、金边不整齐，标合格。"
                   "灯太暗、糊了、或者只拍到边上，标看不清。拿不准的先跳过（→ 键），不要乱标。\n"
                   "圆孔画框会在图上画出四条边：红点是疑似渗墨，品红是疑似缺墨。它只认现在这一种尺寸。"
                   "用肉眼看时，去掉右上角的「显示描线」。建议对了点「采用圆孔结果」，不对就按数字键。")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8)
        ttk.Label(top, text="显示：").pack(side="left")
        self.label_filter = ttk.Combobox(top, values=self.FILTERS, state="readonly", width=16)
        self.label_filter.set("未标注")
        self.label_filter.pack(side="left")
        self.label_filter.bind("<<ComboboxSelected>>", lambda _e: self._reload_labels())
        ttk.Button(top, text="刷新", command=self._reload_labels).pack(side="left", padx=6)
        self.label_stats = ttk.Label(top, text="", font=("Microsoft YaHei UI", 11, "bold"))
        self.label_stats.pack(side="left", padx=16)
        self.show_frame = tk.BooleanVar(value=True)
        tk.Checkbutton(top, text="显示描线", variable=self.show_frame, command=self._toggle_assist_view,
                       font=("Microsoft YaHei UI", 12, "bold"), fg="#1a3d6d",
                       activeforeground="#1a3d6d").pack(side="right", padx=8)

        self.label_view = ImageView(tab, max_up=3.0, zoomable=True)
        self.label_view.pack(fill="both", expand=True, padx=8, pady=6)
        self.label_info = ttk.Label(tab, text="", font=("Microsoft YaHei UI", 11))
        self.label_info.pack(fill="x", padx=12)

        assist = ttk.LabelFrame(tab, text="圆孔画框辅助（只认现在这一种电离室尺寸）")
        assist.pack(fill="x", padx=8, pady=(0, 4))
        head = ttk.Frame(assist)
        head.pack(fill="x", padx=6, pady=(4, 0))
        self.assist_title = tk.Label(head, text="打开一张小图后，圆孔画框会自动分析。", anchor="w", justify="left",
                                     font=("Microsoft YaHei UI", 11, "bold"), wraplength=980, fg="#1a3d6d")
        self.assist_title.pack(side="left", fill="x", expand=True)
        self.assist_text = tk.Text(assist, height=5, wrap="word", font=("Microsoft YaHei UI", 10), state="disabled")
        self.assist_text.pack(fill="x", padx=6, pady=4)
        arow = ttk.Frame(assist)
        arow.pack(fill="x", padx=6, pady=(0, 6))
        self.assist_apply = ttk.Button(arow, text="采用圆孔结果", command=self._apply_assist, state="disabled")
        self.assist_apply.pack(side="left")
        self.assist_batch = ttk.Button(arow, text="未标注的都按建议先标上", command=self._batch_assist)
        self.assist_batch.pack(side="left", padx=8)
        ttk.Label(arow, text="点图片放大，右键缩小，双击看整张。").pack(side="left", padx=8)

        buttons = ttk.Frame(tab)
        buttons.pack(pady=8)
        ttk.Button(buttons, text="← 上一张", command=lambda: self._step(-1)).pack(side="left", padx=6)
        for key, (label, name) in enumerate([("hege", "合格"), ("shenmo", "渗墨"), ("quemo", "缺墨"),
                                             ("kanbuqing", "看不清")], start=1):
            ttk.Button(buttons, text=f"{key}  {name}", style="Big.TButton", width=12,
                       command=lambda lb=label: self._set_label(lb)).pack(side="left", padx=6)
        ttk.Button(buttons, text="下一张 →", command=lambda: self._step(1)).pack(side="left", padx=6)

        self.index_rows: dict = {}
        self.labels: dict = {}
        self.items: list[str] = []
        self.pos = 0
        self._assist_token = 0
        self._assist_label: str | None = None
        self._assist_plain: np.ndarray | None = None
        self._assist_vis: np.ndarray | None = None
        self._assist_busy = False

    def _filtered(self) -> list[str]:
        mode = self.label_filter.get()
        names = {v: k for k, v in cd.LABEL_NAMES.items()}
        rule = {v: k for k, v in cd.RULE_NAMES.items()}
        out = []
        for crop in sorted(self.index_rows):
            folder = crop.split("/")[0]
            label = self.labels.get(crop)
            if mode == "未标注" and label is None \
                    or mode == "全部" \
                    or mode.startswith("已标：") and label == names.get(mode[3:]) \
                    or mode.startswith("规则判：") and folder == rule.get(mode[4:]) \
                    or mode == "规则没找到" and folder == "weidingwei":
                out.append(crop)
        return out

    def _reload_labels(self) -> None:
        current = self.items[self.pos] if self.items and self.pos < len(self.items) else None
        self.index_rows = cd.read_index()
        self.labels = cd.read_labels()
        self.items = self._filtered()
        self.pos = self.items.index(current) if current in self.items else 0
        self._show_label_item()

    def _update_stats(self) -> None:
        counts = cd.label_counts(self.labels)
        done = sum(counts.values())
        self.label_stats.configure(text=f"共 {len(self.index_rows)} 颗，已标 {done} 颗："
                                        + "　".join(f"{cd.LABEL_NAMES[k]} {v}" for k, v in counts.items()))

    def _show_label_item(self) -> None:
        self._update_stats()
        self._assist_token += 1
        self._assist_label = None
        self._assist_plain = None
        self._assist_vis = None
        if not self._assist_busy:
            self.assist_apply.configure(state="disabled")
        if not self.items:
            msg = "这一类里没有图了。" if self.index_rows else "还没有小图。先去「第 1 步：裁图」。"
            if self.index_rows and self.label_filter.get() == "未标注":
                msg = "全部标完了！可以去「第 3 步：训练」。也可以在上面「显示」里选别的类别回头检查。"
            self.label_view.show(None, msg)
            self.label_info.configure(text="")
            self._set_assist("圆孔画框：还没有可分析的小图。", "", None)
            return
        if self.pos >= len(self.items):
            self.label_view.show(None, "这一类看完了。上面「显示」可以换一类检查，或去「第 3 步：训练」。")
            self.label_info.configure(text="")
            self._set_assist("圆孔画框：这一类看完了。", "", None)
            return
        self.pos = max(0, self.pos)
        crop = self.items[self.pos]
        row = self.index_rows.get(crop, {})
        image = cd.read_image(cd.RAW / crop)
        label = self.labels.get(crop)
        done = sum(cd.label_counts(self.labels).values())
        self.label_info.configure(
            text=f"已标 {done} 颗，这一类还剩 {len(self.items)} 张　　当前：{cd.LABEL_NAMES.get(label, '未标注')}　　"
                 f"来源：{Path(row.get('source', '')).name} 第 {row.get('chamber', '?')} 颗　　{Path(crop).name}")
        if image is None:
            self.label_view.show(None, "这张小图读不出来。")
            self._set_assist("这张小图读不出来。", "", None)
            return
        self._assist_plain = image
        self.label_view.reset_zoom()
        self.label_view.show(image)
        token = self._assist_token
        self._set_assist("圆孔画框正在分析这颗…", "", None)
        self.run_bg(lambda img=image: ha.analyse(img),
                    lambda result, t=token, c=crop: self._assist_done(t, c, result))

    def _set_label(self, label: str) -> None:
        if not self.items or self.pos >= len(self.items):
            return
        crop = self.items[self.pos]
        self.labels[crop] = label
        cd.write_labels(self.labels)
        if self.label_filter.get() == "未标注":
            self.items.pop(self.pos)
            if self.items and self.pos >= len(self.items):
                self.pos = 0
        else:
            self.pos += 1
        self._show_label_item()

    def _step(self, delta: int) -> None:
        if self.items:
            self.pos = max(0, min(len(self.items) - 1, self.pos + delta))
            self._show_label_item()

    def _set_assist(self, title: str, detail: str, label: str | None) -> None:
        colors = {"hege": "#2e7d32", "shenmo": "#c62828", "quemo": "#b28704", "kanbuqing": "#e65100"}
        self.assist_title.configure(text=title, fg=colors.get(label, "#1a3d6d"))
        self.assist_text.configure(state="normal")
        self.assist_text.delete("1.0", "end")
        if detail:
            self.assist_text.insert("1.0", detail)
        self.assist_text.configure(state="disabled")

    def _assist_done(self, token: int, crop: str, result: ha.Assist) -> None:
        if token != self._assist_token or self._closing:
            return
        if not self.items or self.pos >= len(self.items) or self.items[self.pos] != crop:
            return
        self._assist_label = result.label
        self._assist_vis = result.overlay
        self._set_assist(result.title, result.detail, result.label)
        if result.label and not self._assist_busy:
            self.assist_apply.configure(state="normal")
        if self.show_frame.get():
            self.label_view.show(result.overlay)

    def _toggle_assist_view(self) -> None:
        image = self._assist_vis if self.show_frame.get() and self._assist_vis is not None else self._assist_plain
        if image is not None:
            self.label_view.show(image)

    def _apply_assist(self) -> None:
        if self._assist_label:
            self._set_label(self._assist_label)

    def _assist_progress(self, done: int, total: int) -> None:
        if self._closing or not self._assist_busy:
            return
        self._set_assist(f"圆孔画框正在先标未标注的：{done}/{total}", "标完后你可以逐张改。框没套准的会留空。", None)

    def _batch_assist(self) -> None:
        if self._assist_busy:
            return
        pending = [crop for crop in sorted(self.index_rows) if not self.labels.get(crop)]
        if not pending:
            messagebox.showinfo("圆孔画框", "没有未标注的小图。")
            return
        if not messagebox.askyesno(
            "按圆孔建议先标",
            f"用圆孔画框给 {len(pending)} 张还没标注的小图先写上建议。\n"
            "框没套准、两种缺陷都有的会跳过，留给你看。写上之后还可以改。继续吗？",
        ):
            return
        self._assist_busy = True
        self._assist_token += 1
        self.assist_apply.configure(state="disabled")
        self.assist_batch.configure(state="disabled")
        total = len(pending)

        def work():
            updates: dict[str, str] = {}
            counts: dict[str, int] = {}
            skipped = 0
            for i, crop in enumerate(pending, 1):
                image = cd.read_image(cd.RAW / crop)
                result = ha.analyse(image) if image is not None else None
                if result is None or result.label is None:
                    skipped += 1
                else:
                    updates[crop] = result.label
                    counts[result.label] = counts.get(result.label, 0) + 1
                if i == 1 or i % 5 == 0 or i == total:
                    self.later(self._assist_progress, i, total)
            return updates, counts, skipped

        def done(pack) -> None:
            updates, counts, skipped = pack
            applied = 0
            for crop, label in updates.items():
                if self.labels.get(crop):
                    continue
                self.labels[crop] = label
                applied += 1
            if applied:
                cd.write_labels(self.labels)
            self._assist_busy = False
            self.assist_batch.configure(state="normal")
            summary = "，".join(f"{cd.LABEL_NAMES[k]} {counts.get(k, 0)}" for k in cd.CLASSES if counts.get(k))
            self._reload_labels()
            messagebox.showinfo("圆孔画框", f"写上了 {applied} 张（{summary or '没有'}）。跳过 {skipped} 张，请你自己看。")

        def failed(exc: Exception, detail: str) -> None:
            self._assist_busy = False
            self.assist_batch.configure(state="normal")
            self._bg_error(exc, detail)

        self.run_bg(work, done, failed)

    # ---------- step 3: train ----------
    def _build_train_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["train"] = tab
        self.nb.add(tab, text="  第 3 步：训练  ")
        guide(tab, "第 3 步：让电脑从你标好的样本里自己学会分辨。样本越多、标得越准，学得越好。\n"
                   f"要求：至少两类、每类至少 {cd.MIN_PER_CLASS} 张才能开始（想要可用的效果，每类建议 50 张以上）。"
                   "程序会自动留出约 20% 的样本不给它学，学完后用这部分来考它，看它判得准不准。\n"
                   "训练在这台电脑的 CPU 上进行：几百张图大约 10–40 分钟。期间可以做别的事，但不要关掉这个窗口。")
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body, width=480)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.train_counts = ttk.Label(left, text="", font=("Microsoft YaHei UI", 12, "bold"), wraplength=460)
        self.train_counts.pack(anchor="w", pady=(0, 8))
        opts = ttk.LabelFrame(left, text="设置（第一次用默认值就行）")
        opts.pack(fill="x")
        self.model_size = tk.StringVar(value="n")
        ttk.Radiobutton(opts, text="小模型 n：训练快、检测快（推荐先用这个）", value="n", variable=self.model_size).pack(anchor="w", padx=6)
        ttk.Radiobutton(opts, text="中模型 s：可能更准一点，训练慢 2–3 倍", value="s", variable=self.model_size).pack(anchor="w", padx=6)
        erow = ttk.Frame(opts)
        erow.pack(fill="x", padx=6, pady=4)
        ttk.Label(erow, text="学几遍（训练轮数）：").pack(side="left")
        self.epochs = tk.IntVar(value=30)
        ttk.Spinbox(erow, from_=5, to=200, increment=5, textvariable=self.epochs, width=6).pack(side="left")
        ttk.Label(erow, text="  样本少时 30 够了").pack(side="left")
        irow = ttk.Frame(opts)
        irow.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(irow, text="图片尺寸：").pack(side="left")
        self.imgsz = ttk.Combobox(irow, values=["320", "416", "512", "640"], state="readonly", width=6)
        self.imgsz.set("512")
        self.imgsz.pack(side="left")
        ttk.Label(irow, text="  越大越能看清 0.3 mm 细缝，也越慢").pack(side="left")

        brow = ttk.Frame(left)
        brow.pack(fill="x", pady=10)
        self.train_btn = ttk.Button(brow, text="开始训练", style="Big.TButton", command=self._start_train)
        self.train_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(brow, text="停止", command=self._stop_train, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))
        self.train_progress = ttk.Progressbar(left, maximum=1.0)
        self.train_progress.pack(fill="x")
        self.train_stage = ttk.Label(left, text="", wraplength=460)
        self.train_stage.pack(anchor="w", pady=4)
        self.train_result = tk.Label(left, text="", justify="left", anchor="nw", wraplength=460,
                                     font=("Microsoft YaHei UI", 11), fg="#1a3d6d")
        self.train_result.pack(fill="both", expand=True)

        ttk.Label(right, text="训练过程：").pack(anchor="w")
        self.train_log = log_box(right, 20)
        self._refresh_train_counts()
        self._show_last_result()

    def _refresh_train_counts(self) -> None:
        counts = cd.label_counts(cd.read_labels())
        parts = [f"{cd.CLASSES[k]} {counts.get(k, 0)} 张" for k in cd.CLASSES]
        ready = sum(counts.get(k, 0) >= cd.MIN_PER_CLASS for k in cd.CLASSES) >= 2
        self.train_counts.configure(text="已标注：" + "，".join(parts) + ("\n可以开始训练。" if ready else
                                    f"\n还不能训练：至少两类、每类 {cd.MIN_PER_CLASS} 张以上。"),
                                    foreground="#2e7d32" if ready else "#c62828")

    def _start_train(self) -> None:
        self.train_btn.configure(state="disabled")
        self.train_result.configure(text="")
        self.train_progress["value"] = 0
        self.train_stage.configure(text="整理样本…")
        append(self.train_log, "—— " + time.strftime("%H:%M:%S") + " 开始 ——")
        self.run_bg(cd.build_dataset, self._dataset_ready, self._dataset_failed)

    def _dataset_failed(self, exc: Exception, _detail: str) -> None:
        self.train_btn.configure(state="normal")
        self.train_stage.configure(text="")
        messagebox.showwarning("还不能训练", str(exc))

    def _dataset_ready(self, summary: dict) -> None:
        for split, name in (("train", "给它学"), ("val", "留着考它")):
            append(self.train_log, f"{name}：" + "，".join(f"{cd.CLASSES[k]} {v} 张" for k, v in summary[split].items()))
        if summary["skipped"]:
            append(self.train_log, "样本太少、这次先不学：" + "，".join(f"{cd.CLASSES[k]} {v} 张" for k, v in summary["skipped"].items()))
        args = [sys.executable, str(HERE / "train_classifier.py"), "--model", self.model_size.get(),
                "--epochs", str(int(self.epochs.get())), "--imgsz", self.imgsz.get()]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TEMP": r"E:\tmp", "TMP": r"E:\tmp"}
        Path(r"E:\tmp").mkdir(exist_ok=True)
        self.train_proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=str(ROOT), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._read_train, args=(self.train_proc,), daemon=True).start()

    def _read_train(self, proc: subprocess.Popen) -> None:
        cd.RUNS.mkdir(parents=True, exist_ok=True)
        tail: list[str] = []
        with (cd.RUNS / "train_log.txt").open("w", encoding="utf-8") as log:
            for line in proc.stdout:
                line = line.rstrip()
                log.write(line + "\n")
                if line.startswith("@@"):
                    tag, _, payload = line[2:].partition(" ")
                    self.later(self._train_msg, tag, json.loads(payload) if payload else None)
                elif line.strip():
                    tail = (tail + [line])[-15:]
        code = proc.wait()
        self.later(self._train_exit, code, tail)

    def _train_msg(self, tag: str, data) -> None:
        if tag == "STAGE":
            self.train_stage.configure(text=data)
            append(self.train_log, data)
        elif tag == "EPOCH":
            if data["epoch"] > data["epochs"]:
                return
            self.train_progress["value"] = data["epoch"] / data["epochs"]
            self.train_stage.configure(text=f"正在学第 {data['epoch']} / {data['epochs']} 遍")
            append(self.train_log, f"第 {data['epoch']:>3} 遍：误差 {data['loss']:.3f}　考试准确率 {data['acc'] * 100:.1f}%")
        elif tag == "DONE":
            self._show_result(data)
            self.model = None

    def _train_exit(self, code: int, tail: list[str]) -> None:
        self.train_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        stopped = self.train_proc is None
        self.train_proc = None
        if code != 0 and not stopped:
            self.train_stage.configure(text="训练出错了")
            append(self.train_log, "训练出错，最后几行信息：\n" + "\n".join(tail) + f"\n完整记录：{cd.RUNS / 'train_log.txt'}")
        elif code == 0:
            self.train_progress["value"] = 1.0

    def _stop_train(self) -> None:
        if self.train_proc and self.train_proc.poll() is None:
            proc, self.train_proc = self.train_proc, None
            proc.terminate()
            self.train_stage.configure(text="已停止")
            append(self.train_log, "已手动停止。")

    def _show_last_result(self) -> None:
        info = cd.read_latest()
        if info:
            self._show_result(info)

    def _show_result(self, info: dict) -> None:
        val = info.get("val", {})
        total, right = val.get("total", 0), val.get("right", 0)
        lines = [f"最近一次训练：{info.get('trained_at', '')}（{info.get('model', '')}，图片尺寸 {info.get('imgsz')}）",
                 f"考试结果：{total} 张里判对 {right} 张，准确率 {right / total * 100:.1f}%" if total else "考试结果：无",
                 f"漏检（渗墨、缺墨或看不清却判成合格）：{val.get('missed', 0)} 张　← 质检最要紧的指标，越少越好",
                 f"误报（合格却判成别的）：{val.get('false_alarm', 0)} 张"]
        wrong = val.get("wrong", [])
        if wrong:
            lines.append("判错的：")
            lines += [f"  {w['file']}：实际 {cd.CLASSES.get(w['truth'], w['truth'])}，判成 {cd.CLASSES.get(w['pred'], w['pred'])}"
                      f"（把握 {w['conf'] * 100:.0f}%）" for w in wrong[:12]]
            if len(wrong) > 12:
                lines.append(f"  …还有 {len(wrong) - 12} 张")
        lines.append("\n下一步：去「第 4 步：测试」拿新图片试试。")
        self.train_result.configure(text="\n".join(lines))

    # ---------- step 4: test ----------
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

    def _build_test_tab(self) -> None:
        tab = ttk.Frame(self.nb)
        self.tabs["test"] = tab
        self.nb.add(tab, text="  第 4 步：测试  ")
        guide(tab, "第 4 步：用训练好的模型检查新图片。点「打开图片」选一张原图，或「从相机拍一张」（先在 MVS 里停止采集）。\n"
                   "程序会找出每颗电离室并给出判定：绿 = 合格，红 = 渗墨，黄 = 缺墨，橙 = 看不清，"
                   "紫 = 模型拿不准（把握不到 60%，建议人工看），灰 = 还没有模型、只显示找到的位置。"
                   "点右边列表的某一行，或直接点图上的框，下方会放大显示那一颗。")
        top = ttk.Frame(tab)
        top.pack(fill="x", padx=8)
        ttk.Button(top, text="打开图片…", style="Big.TButton", command=self._open_test_image).pack(side="left")
        ttk.Button(top, text="从相机拍一张", style="Big.TButton", command=self._grab_test_image).pack(side="left", padx=6)
        self.save_grabs = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text=f"拍的照片自动保存到 {CAPTURES}（以后可拿去第 1 步裁图）",
                        variable=self.save_grabs).pack(side="left", padx=8)
        mrow = ttk.Frame(tab)
        mrow.pack(fill="x", padx=8, pady=4)
        self.model_label = ttk.Label(mrow, text="")
        self.model_label.pack(side="left")
        ttk.Button(mrow, text="换一个模型…", command=self._pick_model).pack(side="left", padx=8)
        ttk.Label(mrow, text="用什么跑：").pack(side="left", padx=(16, 0))
        self.device = ttk.Combobox(mrow, values=["核显（快，推荐）", "CPU"], state="readonly", width=14)
        self.device.set("核显（快，推荐）")
        self.device.pack(side="left")
        self.device.bind("<<ComboboxSelected>>", lambda _e: setattr(self, "model", None))
        self.model_path: str | None = None
        self.model_imgsz = 512

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        side = ttk.Frame(body, width=620)
        side.pack(side="right", fill="y", padx=(8, 0))
        side.pack_propagate(False)
        self.test_view = ImageView(body)
        self.test_view.on_click = self._click_test_image
        self.test_view.pack(side="left", fill="both", expand=True)
        self.test_summary = ttk.Label(side, text="", font=("Microsoft YaHei UI", 12, "bold"), wraplength=600)
        self.test_summary.pack(anchor="w")
        self.test_tree = ttk.Treeview(side, columns=("n", "verdict", "conf"), show="headings", height=12)
        for col, name, width in (("n", "编号", 60), ("verdict", "判定", 120), ("conf", "把握", 90)):
            self.test_tree.heading(col, text=name)
            self.test_tree.column(col, width=width, anchor="center")
        for key, color in TREE_COLORS.items():
            self.test_tree.tag_configure(str(key), foreground=color)
        self.test_tree.pack(fill="both", expand=True, pady=4)
        self.test_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_test_crop())
        self.test_crop = ImageView(side, max_up=2.0, height=260)
        self.test_crop.pack(fill="x")
        self.test_view.show(None, "点上面的「打开图片」或「从相机拍一张」开始。")
        self.test_crop.show(None, "点列表里的一行放大看")
        self.test_hits: list = []
        self.test_crops: list = []
        self.test_results: list = []
        self._refresh_model_label()

    def _refresh_model_label(self) -> None:
        if self.model_path is None:
            info = cd.read_latest()
            if info:
                ov = Path(info["openvino"])
                self.model_path = str(ov if ov.exists() else info["pt"])
                self.model_imgsz = int(info.get("imgsz", 512))
        if self.model_path:
            self.model_label.configure(text=f"当前模型：{self.model_path}（图片尺寸 {self.model_imgsz}）", foreground="#2e7d32")
        else:
            self.model_label.configure(text="当前模型：还没有（先完成第 3 步训练）。现在只会显示找到的电离室。", foreground="#c62828")

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(title="选模型：best.pt，或 best_openvino_model 文件夹里的 .xml",
                                          initialdir=str(cd.RUNS), filetypes=[("模型", "*.pt *.xml")])
        if not path:
            return
        p = Path(path)
        self.model_path = str(p.parent if p.suffix == ".xml" else p)
        self.model_imgsz = model_imgsz(Path(self.model_path)) or self.model_imgsz
        self.model = None
        self._refresh_model_label()

    def _open_test_image(self) -> None:
        path = filedialog.askopenfilename(title="选一张原图", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.test_summary.configure(text="读取中…")
            self.run_bg(lambda: cd.read_image(Path(path)), self._run_test)

    def _grab_test_image(self) -> None:
        self.test_summary.configure(text="正在拍照…")
        save = self.save_grabs.get()
        held = self.session.take() if self.session.connected else None

        def work():
            if held is not None:
                image = held[0]
            else:
                from qc_host.vision.grab import grab_frame
                image, _settings = grab_frame()
            if image is None:
                raise RuntimeError("相机没有返回图像。")
            if save:
                cd.write_image(self._capture_path(""), image)
            return image

        def failed(exc, detail):
            self.test_summary.configure(text="拍照失败")
            messagebox.showerror("拍照失败", f"{exc}\n\n请确认：相机已连接；MVS 里已经停止采集或关闭了 MVS。\n\n{detail[-800:]}")

        self.run_bg(work, self._run_test, failed)

    def _load_model(self, use_gpu: bool):
        from ultralytics import YOLO
        device = "intel:gpu" if use_gpu else "intel:cpu"
        if not Path(self.model_path).is_dir():
            device = "cpu"
        key = (self.model_path, device)
        if self.model is None or self.model_key != key:
            self.model = YOLO(self.model_path, task="classify")
            self.model_key = key
        return self.model, device

    def _run_test(self, image: np.ndarray | None) -> None:
        if image is None:
            messagebox.showerror("出错了", "图片读不出来。")
            return
        if max(image.shape[:2]) < ec.MIN_SIDE:
            messagebox.showwarning("提示", "这张图分辨率太低，需要相机原始分辨率（5472×3648）的原图。")
            return
        self.test_summary.configure(text="找电离室、判定中…（第一次用模型要加载几秒）")
        use_model = bool(self.model_path)
        use_gpu = self.device.get().startswith("核显")
        imgsz = self.model_imgsz

        def work():
            started = time.perf_counter()
            hits = ec.find_chambers(image, self.template, 0.5)
            crops = [ec.crop_chamber(image, h) for h in hits]
            results = [None] * len(hits)
            if use_model and crops:
                model, device = self._load_model(use_gpu)
                results = []
                # The exported model has a fixed batch of one.
                for crop in crops:
                    p = model.predict(cd.square(crop), imgsz=imgsz, device=device, verbose=False)[0]
                    results.append((p.names[int(p.probs.top1)], float(p.probs.top1conf)))
            return image, hits, crops, results, time.perf_counter() - started

        self.run_bg(work, self._show_test, self._test_failed)

    def _test_failed(self, exc, detail) -> None:
        self.test_summary.configure(text="出错了")
        self._bg_error(exc, detail)

    def _show_test(self, payload) -> None:
        image, hits, crops, results, seconds = payload
        self.test_hits, self.test_crops, self.test_results = hits, crops, results
        scale = 1800.0 / max(image.shape[:2])
        vis = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.test_tree.delete(*self.test_tree.get_children())
        counts: dict = {}
        for i, (hit, res) in enumerate(zip(hits, results), start=1):
            if res is None:
                key, verdict, conf = None, "未判定", ""
            else:
                name, p = res
                key = "unsure" if p < UNSURE else name
                verdict = cd.CLASSES.get(name, name) + ("（拿不准）" if p < UNSURE else "")
                conf = f"{p * 100:.0f}%"
            counts[verdict] = counts.get(verdict, 0) + 1
            p0 = (int(hit.x * scale), int(hit.y * scale))
            p1 = (int((hit.x + hit.w) * scale), int((hit.y + hit.h) * scale))
            cv2.rectangle(vis, p0, p1, BOX_COLORS[key], 3)
            cv2.putText(vis, str(i), (p0[0] + 3, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            self.test_tree.insert("", "end", iid=str(i - 1), values=(i, verdict, conf), tags=(str(key),))
        self.test_vis_scale = scale
        self.test_view.show(vis)
        summary = f"找到 {len(hits)} 颗电离室，用时 {seconds:.1f} 秒。" + "　".join(f"{k} {v}" for k, v in counts.items())
        if not hits:
            summary = "一颗电离室都没找到。确认是正面光拍的整板原图，电离室在画面里且清楚。"
        self.test_summary.configure(text=summary)
        self.test_crop.show(None, "点列表里的一行放大看")

    def _show_test_crop(self) -> None:
        sel = self.test_tree.selection()
        if sel:
            i = int(sel[0])
            self.test_crop.show(self.test_crops[i])

    def _click_test_image(self, x: float, y: float) -> None:
        if not self.test_hits:
            return
        sx, sy = x / self.test_vis_scale, y / self.test_vis_scale
        for i, h in enumerate(self.test_hits):
            if h.x <= sx <= h.x + h.w and h.y <= sy <= h.y + h.h:
                self.test_tree.selection_set(str(i))
                self.test_tree.see(str(i))
                return


def model_imgsz(path: Path) -> int | None:
    """Training image size of a best.pt file or an exported OpenVINO folder."""
    try:
        if path.is_dir():
            import yaml
            meta = yaml.safe_load((path / "metadata.yaml").read_text(encoding="utf-8"))
            size = meta.get("imgsz")
        else:
            import torch
            size = torch.load(str(path), map_location="cpu", weights_only=False).get("train_args", {}).get("imgsz")
        return int(size[0] if isinstance(size, (list, tuple)) else size)
    except Exception:
        return None


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
