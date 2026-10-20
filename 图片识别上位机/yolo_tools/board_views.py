# -*- coding: utf-8 -*-
"""Number chambers from the motor origin and show every scan photo that contains one."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import tkinter as tk
from collections import OrderedDict
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

import export_chamber_crops as ec

PX_PER_MM = 44.2
MERGE_MM = 2.5
ROW_MM = 3.0
CACHE = Path(r"E:\yolo\runs\board_view_index.json")
POSE = re.compile(r"_x(-?\d+)_y(-?\d+)$")


def scan_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "qc_motion" / "scan"


def _pose(path: Path) -> tuple[int, int] | None:
    match = POSE.search(path.stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _board_mm(mx: int, my: int, width: int, height: int, px: float, py: float) -> tuple[float, float]:
    """The scene moves up when the camera moves X+ and left when it moves Y+."""
    return mx + (py - height / 2) / PX_PER_MM, my + (px - width / 2) / PX_PER_MM


def build_index(folder: Path, template: np.ndarray, threshold: float = 0.5) -> dict:
    files = sorted(p for p in folder.glob("scan_*.jpg") if _pose(p))
    detections = []
    for path in files:
        pose = _pose(path)
        image = ec._read(path)
        if image is None or pose is None:
            continue
        height, width = image.shape[:2]
        mx, my = pose
        for hit in ec.find_chambers(image, template, threshold):
            cx, cy = hit.center
            bx, by = _board_mm(mx, my, width, height, cx, cy)
            detections.append({
                "file": str(path),
                "mx": mx,
                "my": my,
                "x": int(hit.x),
                "y": int(hit.y),
                "w": int(hit.w),
                "h": int(hit.h),
                "vertical": bool(hit.vertical),
                "score": round(float(hit.score), 3),
                "bx": bx,
                "by": by,
            })
    detections.sort(key=lambda item: -item["score"])
    clusters: list[dict] = []
    for det in detections:
        chosen = None
        nearest = MERGE_MM
        for cluster in clusters:
            distance = float(np.hypot(det["bx"] - cluster["x"], det["by"] - cluster["y"]))
            if distance < nearest:
                nearest = distance
                chosen = cluster
        if chosen is None:
            clusters.append({"x": det["bx"], "y": det["by"], "n": 1, "views": [det]})
            continue
        chosen["views"].append(det)
        chosen["n"] += 1
        chosen["x"] = sum(view["bx"] for view in chosen["views"]) / chosen["n"]
        chosen["y"] = sum(view["by"] for view in chosen["views"]) / chosen["n"]
    clusters.sort(key=lambda cluster: (round(cluster["x"] / ROW_MM), cluster["y"]))
    chambers = []
    for number, cluster in enumerate(clusters, start=1):
        views = sorted(cluster["views"], key=lambda view: -view["score"])
        for view in views:
            view.pop("bx", None)
            view.pop("by", None)
        chambers.append({
            "id": number,
            "x": round(cluster["x"], 1),
            "y": round(cluster["y"], 1),
            "views": views,
        })
    return {"px_per_mm": PX_PER_MM, "merge_mm": MERGE_MM, "chambers": chambers}


def load_index() -> dict | None:
    if not CACHE.is_file():
        return None
    return json.loads(CACHE.read_text(encoding="utf-8"))


def save_index(data: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class BoardViewPanel(ttk.Frame):
    """List chambers numbered from the motor origin, and every photo of the selected one."""

    def __init__(self, master) -> None:
        super().__init__(master)
        self.template = ec.load_template()
        self.data: dict | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._busy = False

        guide = ttk.Label(
            self,
            text="从电机 0、0 起给电离室编号。点左边的编号，右边列出这颗在哪几张扫描照片里，每张是一个拍摄位置。",
            background="#eef4fb",
            padding=8,
            justify="left",
        )
        guide.pack(fill="x")

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Button(bar, text="从扫描照片建立编号", command=self._start).pack(side="left")
        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=8)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body, width=360)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        columns = ("id", "x", "y", "n")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="编号")
        self.tree.heading("x", text="X mm")
        self.tree.heading("y", text="Y mm")
        self.tree.heading("n", text="张数")
        self.tree.column("id", width=60, anchor="e")
        self.tree.column("x", width=80, anchor="e")
        self.tree.column("y", width=80, anchor="e")
        self.tree.column("n", width=50, anchor="e")
        scroll = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected())

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.canvas = tk.Canvas(right, bg="#1e1e1e", highlightthickness=0)
        hbar = ttk.Scrollbar(right, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=hbar.set)
        self.canvas.pack(fill="both", expand=True)
        hbar.pack(fill="x")
        self.view_row = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.view_row, anchor="nw")
        self.view_row.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        cached = load_index()
        if cached:
            self._fill(cached)
        else:
            self.status.set(f"还没有编号。扫描照片在 {scan_dir()}")

    def _start(self) -> None:
        if self._busy:
            return
        folder = scan_dir()
        if not folder.is_dir():
            self.status.set(f"找不到扫描照片：{folder}")
            return
        self._busy = True
        self.status.set("正在逐张找电离室，大约需要半分钟…")

        def work() -> None:
            try:
                data = build_index(folder, self.template)
                save_index(data)
                self.after(0, lambda: self._fill(data))
            except Exception as exc:
                message = str(exc)
                self.after(0, lambda message=message: self.status.set(f"建立失败：{message}"))
            finally:
                self._busy = False

        threading.Thread(target=work, daemon=True).start()

    def _fill(self, data: dict) -> None:
        self.data = data
        self.tree.delete(*self.tree.get_children())
        for chamber in data["chambers"]:
            self.tree.insert("", "end", iid=str(chamber["id"]), values=(
                chamber["id"], f"{chamber['x']:.1f}", f"{chamber['y']:.1f}", len(chamber["views"]),
            ))
        multi = sum(1 for chamber in data["chambers"] if len(chamber["views"]) > 1)
        self.status.set(f"共 {len(data['chambers'])} 颗，其中 {multi} 颗出现在多张照片里。点编号查看。")
        if data["chambers"]:
            self.tree.selection_set("1")
            self.tree.focus("1")
            self._show_selected()

    def _show_selected(self) -> None:
        for child in self.view_row.winfo_children():
            child.destroy()
        self._photos.clear()
        if not self.data:
            return
        selected = self.tree.selection()
        if not selected:
            return
        chamber = next(item for item in self.data["chambers"] if item["id"] == int(selected[0]))
        for view in chamber["views"]:
            image = ec._read(Path(view["file"]))
            if image is None:
                continue
            hit = ec.Hit(view["score"], view["vertical"], view["x"], view["y"], view["w"], view["h"])
            crop = ec.crop_chamber(image, hit)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            picture = Image.fromarray(rgb)
            picture.thumbnail((320, 240))
            photo = ImageTk.PhotoImage(picture)
            self._photos.append(photo)
            card = ttk.Frame(self.view_row)
            card.pack(side="left", padx=6, pady=6)
            ttk.Label(card, image=photo).pack()
            name = Path(view["file"]).name
            ttk.Label(card, text=f"电机 {view['mx']} , {view['my']}\n{name}\n匹配 {view['score']:.2f}").pack()
        self.canvas.xview_moveto(0)


SCAN_YOLO = Path(__file__).resolve().parents[2] / "qc_motion" / "scan_yolo"
KIND_NAME = {"horizontal": "横放", "vertical": "竖放", "short": "短的"}
KIND_COLOR = {"horizontal": "#e53935", "vertical": "#1e88e5", "short": "#43a047"}
LABEL_COLOR = {"hege": "#2e7d32", "shenmo": "#c62828", "quemo": "#f9a825", "kanbuqing": "#e65100", "mixed": "#6a1b9a"}
LABEL_NAME = {"hege": "合格", "shenmo": "渗墨", "quemo": "缺墨", "kanbuqing": "看不清", "mixed": "多种"}
VERDICT_COLOR = {"hege": "#2e7d32", "shenmo": "#c62828", "quemo": "#f9a825", "kanbuqing": "#e65100", "unsure": "#8e24aa"}
UNSURE = 0.6
BOARD_ROTATION_DEG = -2.027


def _verdict_key(name: str, conf: float) -> str:
    return "unsure" if conf < UNSURE else name


def _verdict_text(name: str, conf: float) -> str:
    if conf < UNSURE:
        return f"拿不准 {conf * 100:.0f}%"
    return f"{LABEL_NAME.get(name, name)} {conf * 100:.0f}%"


def load_verdicts(folder: Path) -> dict[int, dict]:
    path = folder / "model.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(key): value for key, value in data.get("chambers", {}).items()}


def classify_chambers(folder: Path, chambers: list[dict], on_progress=None) -> dict[int, dict]:
    """Classify each chamber from its clearest view and write folder/model.json."""
    import chamber_data as cd
    from ultralytics import YOLO

    info = cd.read_latest()
    if not info:
        raise RuntimeError("还没有训练好的模型。")
    weights = Path(info["openvino"])
    if not weights.exists():
        weights = Path(info["pt"])
    if not weights.exists():
        raise RuntimeError(f"模型文件不在：{weights}")
    imgsz = int(info.get("imgsz", 512))
    model = YOLO(str(weights), task="classify")
    device = "intel:gpu" if weights.is_dir() else "cpu"
    jobs = []
    for chamber in chambers:
        views = chamber.get("views") or []
        if not views:
            continue
        jobs.append((int(chamber["id"]), max(views, key=lambda view: view.get("score", 0))))
    by_file: dict[str, list[tuple[int, dict]]] = {}
    for chamber_id, view in jobs:
        by_file.setdefault(view["file"], []).append((chamber_id, view))
    results: dict[int, dict] = {}
    done = 0

    def predict(crop: np.ndarray):
        nonlocal device
        try:
            return model.predict(crop, imgsz=imgsz, device=device, verbose=False)[0]
        except Exception:
            if device == "cpu":
                raise
            device = "cpu"
            return model.predict(crop, imgsz=imgsz, device=device, verbose=False)[0]

    for name, items in by_file.items():
        image = ec._read(folder / name)
        if image is None:
            continue
        for chamber_id, view in items:
            hit = ec.Hit(float(view.get("score", 0)), bool(view.get("vertical")), int(view["x"]), int(view["y"]), int(view["w"]), int(view["h"]))
            crop = ec.crop_chamber(image, hit)
            if crop.size == 0:
                continue
            pred = predict(cd.square(crop))
            results[chamber_id] = {
                "name": pred.names[int(pred.probs.top1)],
                "conf": round(float(pred.probs.top1conf), 3),
                "file": view["file"],
            }
            done += 1
            if on_progress is not None and (done == len(jobs) or done % 20 == 0):
                on_progress(done, len(jobs))
    payload = {"model": str(weights), "imgsz": imgsz, "chambers": {str(key): value for key, value in results.items()}}
    (folder / "model.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return results


def _rotation_for(side: str) -> float:
    """Front columns lean one way. Mirroring Y for the back flip reverses that lean."""
    if side == "back":
        return -BOARD_ROTATION_DEG
    return BOARD_ROTATION_DEG


def _upright_point(machine_x: float, machine_y: float, origin_x: float, origin_y: float, rotation_deg: float) -> tuple[float, float]:
    """Take off the machine rotation so the five columns stand vertical."""
    phi = math.radians(-rotation_deg)
    cosine, sine = math.cos(phi), math.sin(phi)
    dx, dy = machine_x - origin_x, machine_y - origin_y
    return origin_x + cosine * dx - sine * dy, origin_y + sine * dx + cosine * dy


def assign_display_numbers(chambers: list[dict], side: str = "front") -> list[dict]:
    """Five horizontal columns top to bottom, then shorts, then the vertical bar left to right."""
    if not chambers:
        return []
    origin_x = sum(chamber["x"] for chamber in chambers) / len(chambers)
    origin_y = sum(chamber["y"] for chamber in chambers) / len(chambers)
    rotation = _rotation_for(side)
    horizontals: list[dict] = []
    shorts: list[dict] = []
    verticals: list[dict] = []
    for chamber in chambers:
        upright_x, upright_y = _upright_point(chamber["x"], chamber["y"], origin_x, origin_y, rotation)
        chamber["_ux"], chamber["_uy"] = upright_x, upright_y
        if chamber["kind"] == "horizontal":
            horizontals.append(chamber)
        elif chamber["kind"] == "short":
            shorts.append(chamber)
        else:
            verticals.append(chamber)
    horizontals.sort(key=lambda chamber: chamber["_uy"])
    columns: list[list[dict]] = []
    for chamber in horizontals:
        if not columns or chamber["_uy"] - columns[-1][0]["_uy"] > 20:
            columns.append([chamber])
        else:
            columns[-1].append(chamber)
    ordered: list[dict] = []
    for column in columns:
        column.sort(key=lambda chamber: chamber["_ux"])
        ordered.extend(column)
    shorts.sort(key=lambda chamber: (round(chamber["_ux"]), chamber["_uy"]))
    ordered.extend(shorts)
    verticals.sort(key=lambda chamber: chamber["_uy"])
    ordered.extend(verticals)
    for number, chamber in enumerate(ordered, start=1):
        chamber["n"] = number
    return ordered


def _group_module():
    folder = Path(__file__).resolve().parents[2] / "qc_motion"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import _group_line as group_line

    return group_line


def _scan_folders() -> list[Path]:
    """Newest photo folder first, whether the name is beamcheck or pathcheck."""
    if not SCAN_YOLO.is_dir():
        return []
    folders = [path for path in SCAN_YOLO.iterdir() if path.is_dir() and any(path.glob("line_*.jpg"))]
    folders.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return folders


def latest_grouped_scan() -> tuple[Path, dict] | None:
    grouped = [path for path in _scan_folders() if (path / "groups.json").is_file()]
    if not grouped:
        return None
    folder = grouped[0]
    return folder, json.loads((folder / "groups.json").read_text(encoding="utf-8"))


_mouse_down = False
_mouse_watch = False


def _note_mouse_down(_event=None) -> None:
    global _mouse_down
    _mouse_down = True


def _note_mouse_up(_event=None) -> None:
    global _mouse_down
    _mouse_down = False


def _bind_mouse_watch(widget: tk.Misc) -> None:
    global _mouse_watch
    if _mouse_watch:
        return
    _mouse_watch = True
    widget.bind_all("<ButtonPress-1>", _note_mouse_down, add="+")
    widget.bind_all("<ButtonRelease-1>", _note_mouse_up, add="+")


class GerberBoardPanel(ttk.Frame):
    """Show the 399 Gerber ids for one scan. The motor offset is measured again each time."""

    def __init__(self, master, embedded: bool = False) -> None:
        super().__init__(master)
        _bind_mouse_watch(self)
        self.embedded = embedded
        self.folder: Path | None = None
        self.data: dict | None = None
        self.labels: dict[int, str] = {}
        self.verdicts: dict[int, dict] = {}
        self.label_text: dict[int, str] = {}
        self.view_name = None
        self.view_picked = None
        self.pick_view = None
        self.zoom_image = None
        self.on_thumbs = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._busy = False
        self._shown_n: int | None = None
        self._thumbs_ready = False
        self._thumb_token = 0
        self._image_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._cache_lock = threading.Lock()

        guide = ttk.Label(
            self,
            text="扫描结束后按光绘编号识别。点左边编号，坐标图上会标出这一颗；点坐标图上的点，左边跳到对应编号。点下面的照片会放大。",
            background="#eef4fb",
            padding=8,
            justify="left",
            wraplength=1100,
        )
        if not embedded:
            guide.pack(fill="x")

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        self.identify_button = ttk.Button(bar, text="识别已有扫描", command=self._identify_latest)
        self.analyze_button = ttk.Button(bar, text="用模型分析", command=self._analyze)
        if not embedded:
            self.identify_button.pack(side="left")
            self.analyze_button.pack(side="left", padx=(8, 0))
        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=8)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body, width=560 if embedded else 520)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        columns = ("id", "kind", "xy", "n", "label") if embedded else ("id", "kind", "err", "n", "judge")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="编号")
        self.tree.heading("kind", text="外形")
        self.tree.column("id", width=60, anchor="e")
        self.tree.column("kind", width=70, anchor="center")
        if embedded:
            self.tree.heading("xy", text="坐标")
            self.tree.heading("n", text="张数")
            self.tree.heading("label", text="标签")
            self.tree.column("xy", width=130, anchor="e")
            self.tree.column("n", width=50, anchor="e")
            self.tree.column("label", width=70, anchor="center")
        else:
            self.tree.heading("err", text="误差 mm")
            self.tree.heading("n", text="张数")
            self.tree.heading("judge", text="判定")
            self.tree.column("err", width=80, anchor="e")
            self.tree.column("n", width=50, anchor="e")
            self.tree.column("judge", width=120, anchor="center")
        self.tree.tag_configure("missing", foreground="#c62828")
        for key, color in VERDICT_COLOR.items():
            self.tree.tag_configure(key, foreground=color)
        scroll = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected())

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        thumbs = ttk.Frame(right, height=230)
        thumbs.pack(side="bottom", fill="x")
        thumbs.pack_propagate(False)
        self.map = tk.Canvas(right, bg="white", highlightthickness=0, cursor="hand2")
        self.map.pack(fill="both", expand=True)
        self.map.bind("<Configure>", lambda _e: self._draw_map())
        self.map.bind("<Button-1>", self._on_map_click)
        self._markers: list[tuple[float, float, int]] = []
        self._zoom_win: tk.Toplevel | None = None
        self._zoom_photo = None
        self._zoom_pending: tuple[np.ndarray, str] | None = None
        self._zoom_job = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self.canvas = tk.Canvas(thumbs, bg="#1e1e1e", highlightthickness=0)
        hbar = ttk.Scrollbar(thumbs, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=hbar.set)
        self.canvas.pack(fill="both", expand=True)
        hbar.pack(fill="x")
        self.view_row = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.view_row, anchor="nw")
        self.view_row.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        cached = latest_grouped_scan()
        if cached:
            self.show(*cached)
        else:
            self.status.set("还没有识别结果。扫完整板后这里会按所选板面列出。")

    def _identify_latest(self) -> None:
        folders = _scan_folders()
        if not folders:
            self.status.set("还没有扫描文件夹。")
            return
        self.identify(folders[0])

    def identify(self, folder: Path, on_done=None) -> None:
        if self._busy:
            return
        self._busy = True
        self.identify_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        self.status.set(f"正在按光绘编号识别 {folder.name}，大约半分钟…")

        def work() -> None:
            try:
                payload = _group_module().group_scan(folder)
            except Exception as exc:
                self.after(0, lambda exc=exc: self._failed(exc, on_done))
                return
            self.after(0, lambda payload=payload: self._finished(folder, payload, on_done))

        threading.Thread(target=work, daemon=True).start()

    def _failed(self, exc: Exception, on_done) -> None:
        self._busy = False
        self.identify_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        self.status.set(f"识别没有完成：{exc}")
        if on_done:
            on_done(None, exc)

    def _finished(self, folder: Path, payload: dict, on_done) -> None:
        self._busy = False
        self.identify_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        self.show(folder, payload)
        if on_done:
            on_done(payload, None)

    def show(self, folder: Path, payload: dict) -> None:
        self.folder = folder
        self.data = payload
        self.verdicts = load_verdicts(folder)
        self._shown_n = None
        self._thumbs_ready = False
        self._thumb_token += 1
        if payload["chambers"]:
            self._origin_x = sum(chamber["x"] for chamber in payload["chambers"]) / len(payload["chambers"])
            self._origin_y = sum(chamber["y"] for chamber in payload["chambers"]) / len(payload["chambers"])
        ordered = self._assign_numbers()
        self.tree.delete(*self.tree.get_children())
        first_missing = None
        for chamber in ordered:
            missing = not chamber["views"]
            if missing and first_missing is None:
                first_missing = str(chamber["n"])
            error = "" if missing else f"{chamber['error_mm']:.2f}"
            kind = KIND_NAME.get(chamber["kind"], chamber["kind"])
            if self.embedded:
                values = (
                    chamber["n"], kind, f"{chamber['x']:.1f}, {chamber['y']:.1f}", len(chamber["views"]),
                    self.label_text.get(chamber.get("id"), LABEL_NAME.get(self.labels.get(chamber.get("id")), "")),
                )
            else:
                verdict = self.verdicts.get(chamber.get("id"))
                judge = _verdict_text(verdict["name"], float(verdict["conf"])) if verdict else ""
                values = (chamber["n"], kind, error, len(chamber["views"]), judge)
            tag = "missing" if missing else ""
            if not missing and not self.embedded:
                verdict = self.verdicts.get(chamber.get("id"))
                if verdict:
                    tag = _verdict_key(verdict["name"], float(verdict["conf"]))
            self.tree.insert("", "end", iid=str(chamber["n"]), tags=(tag,) if tag else (), values=values)
        total = payload.get("gerber") or len(payload.get("chambers") or [])
        side_name = "背面" if payload.get("side") == "back" else "正面"
        judged = self._verdict_summary()
        self.status.set(
            f"xbeam {side_name} {payload['matched']}/{total} 已编号，缺 {payload['missing']}，多余框 {payload['extras']}，"
            f"误差中位 {payload['median_error_mm']} mm。{judged}{folder.name}"
        )
        pick = first_missing or (str(ordered[0]["n"]) if ordered else "")
        if pick and self.tree.exists(pick):
            self.tree.selection_set(pick)
            self.tree.focus(pick)
            self.tree.see(pick)

    def _verdict_summary(self) -> str:
        if not self.verdicts:
            return ""
        counts: dict[str, int] = {}
        for item in self.verdicts.values():
            key = _verdict_key(item["name"], float(item["conf"]))
            title = "拿不准" if key == "unsure" else LABEL_NAME.get(item["name"], item["name"])
            counts[title] = counts.get(title, 0) + 1
        text = "，".join(f"{name} {count}" for name, count in counts.items())
        note = "这是正面训练的模型，背面结果只供对照。" if (self.data or {}).get("side") == "back" else ""
        return f"模型：{text}。{note} "

    def _analyze(self) -> None:
        if self._busy or self.folder is None or not self.data:
            self.status.set("先有一整板的编号，再做模型分析。")
            return
        self._busy = True
        self.analyze_button.configure(state="disabled")
        self.identify_button.configure(state="disabled")
        folder = self.folder
        chambers = self.data["chambers"]
        self.status.set(f"正在用模型分析 {folder.name}…")

        def progress(done: int, total: int) -> None:
            self.after(0, lambda done=done, total=total: self.status.set(f"模型分析 {done}/{total}"))

        def work() -> None:
            try:
                results = classify_chambers(folder, chambers, progress)
            except Exception as exc:
                self.after(0, lambda exc=exc: self._analyze_failed(exc))
                return
            self.after(0, lambda results=results: self._analyze_done(results))

        threading.Thread(target=work, daemon=True).start()

    def _analyze_failed(self, exc: Exception) -> None:
        self._busy = False
        self.analyze_button.configure(state="normal")
        self.identify_button.configure(state="normal")
        self.status.set(f"模型分析没有完成：{exc}")

    def _analyze_done(self, results: dict[int, dict]) -> None:
        self._busy = False
        self.analyze_button.configure(state="normal")
        self.identify_button.configure(state="normal")
        self.verdicts = results
        if self.folder is not None and self.data is not None:
            self.show(self.folder, self.data)
            return
        self._draw_map()

    def _assign_numbers(self) -> list[dict]:
        """Five horizontal columns top to bottom, then shorts, then the vertical bar left to right."""
        return assign_display_numbers(self.data["chambers"], self.data.get("side", "front"))

    def _upright(self, machine_x: float, machine_y: float) -> tuple[float, float]:
        """Take off the machine rotation so the five columns stand vertical."""
        side = (self.data or {}).get("side", "front")
        return _upright_point(machine_x, machine_y, self._origin_x, self._origin_y, _rotation_for(side))

    def _plot_box(self) -> tuple[float, float, float, float, float, float, float, float] | None:
        if not self.data:
            return None
        width, height = self.map.winfo_width(), self.map.winfo_height()
        if width < 40 or height < 40:
            return None
        points = [self._upright(chamber["x"], chamber["y"]) for chamber in self.data["chambers"]]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        pad = 16.0
        x0, x1 = min(0.0, min(xs) - pad), max(xs) + pad
        y0, y1 = min(0.0, min(ys) - pad), max(ys) + pad
        avail_w, avail_h = width - 110, height - 120
        scale = min(avail_w / max(y1 - y0, 1.0), avail_h / max(x1 - x0, 1.0))
        used_w, used_h = (y1 - y0) * scale, (x1 - x0) * scale
        left = 84 + (avail_w - used_w) / 2
        top = 64 + (avail_h - used_h) / 2
        return x0, x1, y0, y1, left, left + used_w, top, top + used_h

    def _to_screen(self, machine_x: float, machine_y: float, box) -> tuple[float, float]:
        x0, x1, y0, y1, left, right, top, bottom = box
        sx = left + (machine_y - y0) / (y1 - y0) * (right - left)
        sy = top + (machine_x - x0) / (x1 - x0) * (bottom - top)
        return sx, sy

    def _draw_map(self) -> None:
        self.map.delete("all")
        self._markers = []
        if self.data:
            self._origin_x = sum(chamber["x"] for chamber in self.data["chambers"]) / len(self.data["chambers"])
            self._origin_y = sum(chamber["y"] for chamber in self.data["chambers"]) / len(self.data["chambers"])
        box = self._plot_box()
        if box is None or not self.data:
            return
        _x0, _x1, _y0, _y1, left, right, top, bottom = box
        self.map.create_rectangle(left, top, right, bottom, outline="#e65100", width=2)
        machine_x0, machine_x1, machine_y0, machine_y1 = box[0], box[1], box[2], box[3]
        for value in (-50, 0, 50, 100, 150, 200, 250):
            if machine_y0 <= value <= machine_y1:
                sx, _sy = self._to_screen(machine_x0, value, box)
                self.map.create_line(sx, top, sx, bottom, fill="#eeeeee")
                self.map.create_text(sx, bottom + 14, text=f"{value:.0f}", fill="#616161", font=("Microsoft YaHei UI", 9))
        for value in (0, 100, 200, 300):
            if machine_x0 <= value <= machine_x1:
                _sx, sy = self._to_screen(value, machine_y0, box)
                self.map.create_line(left, sy, right, sy, fill="#eeeeee")
                self.map.create_text(left - 8, sy, text=f"{value:.0f}", anchor="e", fill="#616161", font=("Microsoft YaHei UI", 9))
        self.map.create_text((left + right) / 2, bottom + 28, text="Y 沿横梁 (mm)", fill="#424242", font=("Microsoft YaHei UI", 10))
        self.map.create_text(18, (top + bottom) / 2, text="X 朝绿电路 (mm)", angle=90, fill="#424242", font=("Microsoft YaHei UI", 10))
        selected = self.tree.selection()
        selected_id = int(selected[0]) if selected else None
        chosen = None
        for chamber in self.data["chambers"]:
            if "n" not in chamber:
                continue
            sx, sy = self._to_screen(*self._upright(chamber["x"], chamber["y"]), box)
            self._markers.append((sx, sy, chamber["n"]))
            if chamber["n"] == selected_id:
                chosen = (sx, sy, chamber)
                continue
            labeled = LABEL_COLOR.get(self.labels.get(chamber.get("id")))
            verdict = self.verdicts.get(chamber.get("id"))
            if not chamber["views"]:
                color = "#9e9e9e"
            elif labeled:
                color = labeled
            elif verdict:
                color = VERDICT_COLOR.get(_verdict_key(verdict["name"], float(verdict["conf"])), "#757575")
            else:
                color = KIND_COLOR.get(chamber["kind"], "#757575")
            self.map.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill=color, outline="")
        if chosen:
            sx, sy, chamber = chosen
            self.map.create_oval(sx - 9, sy - 9, sx + 9, sy + 9, outline="#f9a825", width=3)
            self.map.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill="#f9a825", outline="#f9a825")
            anchor = "w" if sx < right - 36 else "e"
            self.map.create_text(sx + (12 if anchor == "w" else -12), sy, text=str(chamber["n"]), anchor=anchor,
                                 fill="#e65100", font=("Microsoft YaHei UI", 12, "bold"))
        if self.verdicts:
            legend = [("合格", VERDICT_COLOR["hege"]), ("渗墨", VERDICT_COLOR["shenmo"]), ("缺墨", VERDICT_COLOR["quemo"]),
                      ("看不清", VERDICT_COLOR["kanbuqing"]), ("拿不准", VERDICT_COLOR["unsure"])]
        else:
            legend = [("横放", KIND_COLOR["horizontal"]), ("竖放", KIND_COLOR["vertical"]), ("短的", KIND_COLOR["short"])]
        lx = right - 8
        for index, (name, color) in enumerate(legend):
            ly = top + 8 + index * 16
            self.map.create_oval(lx - 70, ly - 4, lx - 62, ly + 4, fill=color, outline=color)
            self.map.create_text(lx - 56, ly, text=name, anchor="w", fill="#424242", font=("Microsoft YaHei UI", 9))
        self._draw_axis_mark(*self._to_screen(0.0, 0.0, box), left, top)

    def _draw_axis_mark(self, origin_x: float, origin_y: float, left: float, top: float) -> None:
        """Cross sits on real (0, 0). Direction arrows stay in the margin so they do not cover the left chambers."""
        self.map.create_line(origin_x - 8, origin_y, origin_x + 8, origin_y, fill="#e65100", width=2)
        self.map.create_line(origin_x, origin_y - 8, origin_x, origin_y + 8, fill="#e65100", width=2)
        self.map.create_oval(origin_x - 5, origin_y - 5, origin_x + 5, origin_y + 5, fill="#fff3e0", outline="#e65100", width=2)
        self.map.create_text(origin_x + 10, origin_y - 12, text="0, 0", anchor="sw", fill="#e65100", font=("Microsoft YaHei UI", 12, "bold"))
        self.map.create_line(left, top - 28, left + 86, top - 28, fill="#e65100", width=2, arrow=tk.LAST, arrowshape=(12, 14, 5))
        self.map.create_text(left + 92, top - 28, text="Y 沿横梁", anchor="w", fill="#e65100", font=("Microsoft YaHei UI", 11, "bold"))
        self.map.create_line(left - 36, top, left - 36, top + 86, fill="#e65100", width=2, arrow=tk.LAST, arrowshape=(12, 14, 5))
        self.map.create_text(left - 36, top + 94, text="X 朝绿电路", anchor="n", fill="#e65100", font=("Microsoft YaHei UI", 11, "bold"))

    def _on_map_click(self, event) -> None:
        if not self._markers:
            return
        nearest = min(self._markers, key=lambda item: (item[0] - event.x) ** 2 + (item[1] - event.y) ** 2)
        if (nearest[0] - event.x) ** 2 + (nearest[1] - event.y) ** 2 > 16 ** 2:
            return
        iid = str(nearest[2])
        if not self.tree.exists(iid):
            return
        self.tree.selection_set(iid)
        self.tree.focus(iid)
        self.tree.see(iid)

    def _zoom_window(self) -> tk.Toplevel | None:
        window = self._zoom_win
        if window is None:
            return None
        try:
            if window.winfo_exists():
                return window
        except tk.TclError:
            pass
        self._zoom_win = None
        self._zoom_photo = None
        return None

    def _close_zoom(self) -> None:
        if self._zoom_job is not None:
            try:
                self.after_cancel(self._zoom_job)
            except tk.TclError:
                pass
            self._zoom_job = None
        self._zoom_pending = None
        window = self._zoom_win
        self._zoom_win = None
        self._zoom_photo = None
        if window is None:
            return
        try:
            if window.winfo_exists():
                window.destroy()
        except tk.TclError:
            pass

    def _present_zoom(self, window: tk.Toplevel, *, created: bool) -> None:
        """Only restore a window that was actually hidden. lift() during a click freezes Tk on Windows."""
        try:
            if created or not window.winfo_viewable():
                window.deiconify()
        except tk.TclError:
            self._close_zoom()

    def _move_zoom_on_screen(self, window: tk.Toplevel, *, first: bool, width: int, height: int) -> None:
        try:
            screen_w, screen_h = window.winfo_screenwidth(), window.winfo_screenheight()
            width, height = max(width, 1), max(height, 1)
            if first:
                host = self.winfo_toplevel()
                x = host.winfo_rootx() + host.winfo_width() + 12
                y = host.winfo_rooty() + 40
                if x + width > screen_w - 8:
                    x = host.winfo_rootx() + 80
            else:
                x, y = window.winfo_rootx(), window.winfo_rooty()
                if 0 <= x <= screen_w - 80 and 0 <= y <= screen_h - 80:
                    return
            x = min(max(0, x), max(0, screen_w - min(width, screen_w)))
            y = min(max(0, y), max(0, screen_h - 80))
            window.geometry(f"+{int(x)}+{int(y)}")
        except tk.TclError:
            pass

    def _zoom_crop(self, image: np.ndarray, title: str) -> None:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3:
            return
        self._zoom_pending = (image, title)
        if self._zoom_job is not None:
            return
        self._zoom_job = self.after(60, self._flush_zoom)

    def _flush_zoom(self) -> None:
        self._zoom_job = None
        if not self.winfo_exists():
            return
        if _mouse_down:
            self._zoom_job = self.after(60, self._flush_zoom)
            return
        pending = self._zoom_pending
        self._zoom_pending = None
        if pending is None:
            return
        image, title = pending
        window = self._zoom_window()
        created = window is None
        if created:
            window = tk.Toplevel(self.winfo_toplevel())
            window.title(title)
            window.transient(self.winfo_toplevel())
            window.protocol("WM_DELETE_WINDOW", self._close_zoom)
            label = tk.Label(window, bd=0, bg="black")
            label.pack()
            window._label = label
            self._zoom_win = window
        window.title(title)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        long_side = max(picture.size)
        limit = max(480, int(min(window.winfo_screenwidth(), window.winfo_screenheight()) * 0.75))
        scale = min(3.0, limit / long_side)
        if abs(scale - 1.0) > 0.01:
            picture = picture.resize(
                (max(1, int(picture.width * scale)), max(1, int(picture.height * scale))),
                Image.Resampling.BILINEAR,
            )
        photo = ImageTk.PhotoImage(picture, master=window)
        window._label.configure(image=photo)
        window._photo = photo
        self._zoom_photo = photo
        self._present_zoom(window, created=created)
        self._move_zoom_on_screen(window, first=created, width=picture.width, height=picture.height)
        if self._zoom_pending is not None and self._zoom_job is None:
            self._zoom_job = self.after(60, self._flush_zoom)

    def follow_zoom(self) -> None:
        """Keep an open enlargement on the yellow-framed photo when the selection moves."""
        window = self._zoom_window()
        if window is None or not self.view_picked:
            return
        for card in self.view_row.winfo_children():
            view = getattr(card, "_view", None)
            crop = getattr(card, "_crop", None)
            if view is None or crop is None or not self.view_picked(view):
                continue
            title = f"编号 {self._shown_n}　电机 {view['mx']} , {view['my']}　{view['file']}"
            self._zoom_crop(self._zoom_source(view, crop), title)
            return

    def _cached_frame(self, name: str) -> np.ndarray | None:
        with self._cache_lock:
            if name in self._image_cache:
                self._image_cache.move_to_end(name)
                return self._image_cache[name]
        image = ec._read(self.folder / name) if self.folder is not None else None
        with self._cache_lock:
            self._image_cache[name] = image
            while len(self._image_cache) > 4:
                self._image_cache.popitem(last=False)
        return image

    def _show_selected(self) -> None:
        if not self.data or self.folder is None:
            return
        selected = self.tree.selection()
        if not selected:
            return
        number = int(selected[0])
        chamber = next(item for item in self.data["chambers"] if item["n"] == number)
        if self._shown_n == number:
            self._draw_map()
            if self._thumbs_ready:
                self.refresh_pick()
            return
        self._shown_n = number
        self._thumbs_ready = False
        self._thumb_token += 1
        token = self._thumb_token
        self._draw_map()
        for child in self.view_row.winfo_children():
            child.destroy()
        self._photos.clear()
        if not chamber["views"]:
            ttk.Label(self.view_row, text=f"编号 {chamber['n']} 没有照片", foreground="#ffcdd2").pack(padx=12, pady=12)
            return
        ttk.Label(self.view_row, text="正在取这几张…", foreground="#eceff1").pack(padx=12, pady=12)
        views = list(chamber["views"])

        def work() -> list[tuple[dict, np.ndarray]] | None:
            crops = []
            for view in views:
                if token != self._thumb_token:
                    return None
                image = self._cached_frame(view["file"])
                if image is None:
                    continue
                hit = ec.Hit(view["score"], view["vertical"], view["x"], view["y"], view["w"], view["h"])
                crops.append((view, ec.crop_chamber(image, hit)))
            return crops

        def finish() -> None:
            try:
                crops = work()
            except Exception:
                crops = []
            self.after(0, lambda: self._place_thumbs(token, crops or []))

        threading.Thread(target=finish, daemon=True).start()

    def _place_thumbs(self, token: int, crops: list[tuple[dict, np.ndarray]]) -> None:
        if token != self._thumb_token:
            return
        for child in self.view_row.winfo_children():
            child.destroy()
        self._photos.clear()
        if not crops:
            ttk.Label(self.view_row, text="这几张没有读出来", foreground="#ffcdd2").pack(padx=12, pady=12)
            return
        for view, crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            picture = Image.fromarray(rgb)
            picture.thumbnail((320, 240))
            photo = ImageTk.PhotoImage(picture)
            self._photos.append(photo)
            picked = bool(self.view_picked and self.view_picked(view))
            card = tk.Frame(
                self.view_row, bg="#1e1e1e", highlightthickness=4,
                highlightbackground="#f9a825" if picked else "#1e1e1e",
            )
            card.pack(side="left", padx=6, pady=6)
            title = f"编号 {self._shown_n}　电机 {view['mx']} , {view['my']}　{view['file']}"
            picture_label = ttk.Label(card, image=photo, cursor="hand2")
            picture_label.pack()
            if self.embedded and self.pick_view:
                card._view = view
                card._crop = crop.copy()
                card._picture = picture_label
                picture_label.bind("<Button-1>", lambda _e, view=view: self.pick_view(view))
                picture_label.bind(
                    "<Double-Button-1>",
                    lambda _e, view=view, crop=crop.copy(), title=title: self._zoom_crop(self._zoom_source(view, crop), title),
                )
                name = self.view_name(view) if self.view_name else ""
                caption = ttk.Label(card, text=f"电机 {view['mx']} , {view['my']}\n{view['file']}\n{name}　双击放大")
                caption.pack()
                card._caption = caption
            else:
                picture_label.bind("<Button-1>", lambda _e, crop=crop.copy(), title=title: self._zoom_crop(crop, title))
                ttk.Label(card, text=f"电机 {view['mx']} , {view['my']}\n{view['file']}\n匹配 {view['score']:.2f}　点击放大").pack()
        self._thumbs_ready = True
        self.canvas.xview_moveto(0)
        if self.on_thumbs is not None:
            self.on_thumbs()

    def _zoom_source(self, view: dict, crop: np.ndarray) -> np.ndarray:
        if self.zoom_image is None:
            return crop
        return self.zoom_image(view, crop)

    def crop_of(self, view: dict) -> np.ndarray | None:
        for card in self.view_row.winfo_children():
            other = getattr(card, "_view", None)
            if other is None:
                continue
            if other["file"] == view["file"] and int(other["x"]) == int(view["x"]) and int(other["y"]) == int(view["y"]):
                crop = getattr(card, "_crop", None)
                return None if crop is None else crop.copy()
        return None

    def set_card_image(self, view: dict, image: np.ndarray) -> None:
        for card in self.view_row.winfo_children():
            other = getattr(card, "_view", None)
            picture = getattr(card, "_picture", None)
            if other is None or picture is None:
                continue
            if other["file"] != view["file"] or int(other["x"]) != int(view["x"]) or int(other["y"]) != int(view["y"]):
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((320, 240))
            photo = ImageTk.PhotoImage(thumb)
            picture.configure(image=photo)
            card._photo = photo
            return

    def refresh_pick(self) -> None:
        """Move the yellow frame without rebuilding the photos, so a double-click can still zoom."""
        for card in self.view_row.winfo_children():
            view = getattr(card, "_view", None)
            if view is None:
                continue
            picked = bool(self.view_picked and self.view_picked(view))
            card.configure(highlightbackground="#f9a825" if picked else "#1e1e1e")
            caption = getattr(card, "_caption", None)
            if caption is not None and self.view_name:
                caption.configure(text=f"电机 {view['mx']} , {view['my']}\n{view['file']}\n{self.view_name(view)}　双击放大")

    def note_labels(self) -> None:
        """Refresh the label column and the coloured points. Display numbers stay as they are."""
        if not self.embedded or not self.data:
            self._draw_map()
            return
        for chamber in self.data["chambers"]:
            iid = str(chamber.get("n"))
            if not self.tree.exists(iid):
                continue
            values = list(self.tree.item(iid, "values"))
            values[-1] = self.label_text.get(chamber.get("id"), LABEL_NAME.get(self.labels.get(chamber.get("id")), ""))
            self.tree.item(iid, values=values)
        self._draw_map()


def _view_labels(payload: dict) -> dict[tuple[str, int, int], tuple[int, str]]:
    assign_display_numbers(payload["chambers"], payload.get("side", "front"))
    labels = {}
    for chamber in payload["chambers"]:
        kind = KIND_NAME.get(chamber["kind"], chamber["kind"])
        for view in chamber["views"]:
            labels[(view["file"], int(view["x"]), int(view["y"]))] = (int(chamber["n"]), kind)
    return labels


def _with_board(dets: list[dict], pose: tuple[int, int], size: tuple[int, int]) -> list[dict]:
    mx, my = pose
    width, height = size
    items = []
    for det in dets:
        cx = det["x"] + det["w"] / 2.0
        cy = det["y"] + det["h"] / 2.0
        bx, by = _board_mm(mx, my, width, height, cx, cy)
        items.append({
            "file": det["file"], "x": int(det["x"]), "y": int(det["y"]), "w": int(det["w"]), "h": int(det["h"]),
            "cx": cx, "cy": cy, "bx": bx, "by": by,
        })
    return items


def _project_box(bx: float, by: float, w: int, h: int, pose: tuple[int, int], size: tuple[int, int]):
    """Where a board point lands in a photo, and whether the whole box still fits."""
    mx, my = pose
    width, height = size
    px = (by - my) * PX_PER_MM + width / 2.0
    py = (bx - mx) * PX_PER_MM + height / 2.0
    x0, y0 = px - w / 2.0, py - h / 2.0
    inside = x0 >= -1 and y0 >= -1 and x0 + w <= width + 1 and y0 + h <= height + 1
    return px, py, (x0, y0, w, h), inside


def _nearest_index(item: dict, others: list[dict]) -> tuple[int | None, float]:
    found = None
    distance = 1e9
    for index, other in enumerate(others):
        gap = math.hypot(item["bx"] - other["bx"], item["by"] - other["by"])
        if gap < distance:
            distance = gap
            found = index
    return found, distance


def build_overlap_rows(
    dets_a: list[dict], pose_a: tuple[int, int], size_a: tuple[int, int],
    dets_b: list[dict], pose_b: tuple[int, int], size_b: tuple[int, int],
    labels: dict[tuple[str, int, int], tuple[int, str]],
) -> list[dict]:
    """Theory is the platform shift. Actual is a box within 2.5 mm on the board."""
    items_a = _with_board(dets_a, pose_a, size_a)
    items_b = _with_board(dets_b, pose_b, size_b)
    nearest_b = [_nearest_index(item, items_b) for item in items_a]
    pairs = []
    for index_a, (_index_b, distance) in enumerate(nearest_b):
        for index_b, item_b in enumerate(items_b):
            distance = math.hypot(items_a[index_a]["bx"] - item_b["bx"], items_a[index_a]["by"] - item_b["by"])
            if distance < MERGE_MM:
                pairs.append((distance, index_a, index_b))
    pairs.sort()
    paired_a: dict[int, tuple[int, float]] = {}
    paired_b: dict[int, tuple[int, float]] = {}
    for distance, index_a, index_b in pairs:
        if index_a in paired_a or index_b in paired_b:
            continue
        paired_a[index_a] = (index_b, distance)
        paired_b[index_b] = (index_a, distance)
    claimed_b = {index_b for index_b, distance in nearest_b if index_b is not None and distance < 3.5}

    def label_of(item: dict) -> tuple[int, str] | None:
        return labels.get((item["file"], item["x"], item["y"]))

    def row_for(item_a: dict | None, item_b: dict | None, paired: bool, near_mm: float | None) -> dict:
        theory = False
        pred_b = pred_a = None
        if item_a is not None:
            _px, _py, pred_b, theory_b = _project_box(item_a["bx"], item_a["by"], item_a["w"], item_a["h"], pose_b, size_b)
            theory = theory or theory_b
        else:
            theory_b = False
        if item_b is not None:
            _px, _py, pred_a, theory_a = _project_box(item_b["bx"], item_b["by"], item_b["w"], item_b["h"], pose_a, size_a)
            if item_a is None:
                theory = theory_a
        mark_a = label_of(item_a) if item_a else None
        mark_b = label_of(item_b) if item_b else None
        if mark_a and mark_b and mark_a[0] != mark_b[0]:
            tag = "clash"
            number = f"{mark_a[0]}/{mark_b[0]}"
            kind = mark_a[1]
        elif mark_a:
            number, kind = str(mark_a[0]), mark_a[1]
            tag = "ok"
        elif mark_b:
            number, kind = str(mark_b[0]), mark_b[1]
            tag = "ok"
        else:
            number, kind, tag = "—", "", "ok"
        pixel_err = None
        if paired and item_a is not None and item_b is not None and pred_b is not None:
            pixel_err = math.hypot(item_b["cx"] - (pred_b[0] + pred_b[2] / 2), item_b["cy"] - (pred_b[1] + pred_b[3] / 2))
        if tag != "clash":
            if theory and paired:
                tag = "ok"
            elif theory and not paired:
                tag = "miss"
            elif paired and not theory:
                tag = "extra"
            else:
                tag = "only"
        return {
            "n": number,
            "kind": kind,
            "theory": "应重叠" if theory else "不应重叠",
            "actual": "两张都有框" if paired else "只有一张",
            "same": "不一样" if tag in ("miss", "extra", "clash") else "一样",
            "dist": None if near_mm is None or near_mm > 80 else round(near_mm, 2),
            "pixel_err": None if pixel_err is None else round(pixel_err, 1),
            "tag": tag,
            "box_a": None if item_a is None else (item_a["x"], item_a["y"], item_a["w"], item_a["h"]),
            "box_b": None if item_b is None else (item_b["x"], item_b["y"], item_b["w"], item_b["h"]),
            "pred_a": pred_a,
            "pred_b": pred_b,
            "ax": None if item_a is None else item_a["bx"],
            "ay": None if item_a is None else item_a["by"],
            "bx": None if item_b is None else item_b["bx"],
            "by": None if item_b is None else item_b["by"],
        }

    rows = []
    for index_a, item_a in enumerate(items_a):
        if index_a in paired_a:
            index_b, distance = paired_a[index_a]
            rows.append(row_for(item_a, items_b[index_b], True, distance))
        else:
            index_b, distance = nearest_b[index_a]
            rows.append(row_for(item_a, None, False, None if index_b is None else distance))
    for index_b, item_b in enumerate(items_b):
        if index_b in paired_b or index_b in claimed_b:
            continue
        _index_a, distance = _nearest_index(item_b, items_a)
        rows.append(row_for(None, item_b, False, None if _index_a is None else distance))
    order = {"clash": 0, "miss": 1, "extra": 2, "ok": 3, "only": 4}
    rows.sort(key=lambda item: (order[item["tag"]], -(item["pixel_err"] or 0), item["n"]))
    return rows


_BOX_COLOR = {
    "ok": (70, 170, 70),
    "miss": (40, 40, 210),
    "extra": (0, 140, 230),
    "clash": (180, 0, 160),
    "only": (150, 150, 150),
}


class OverlapPanel(ttk.Frame):
    """Let the operator pick two photos and compare the motion prediction with the boxes found."""

    def __init__(self, master) -> None:
        super().__init__(master)
        self.folder: Path | None = None
        self.labels: dict[tuple[str, int, int], tuple[int, str]] = {}
        self._file_by_label: dict[str, Path] = {}
        self._detections: dict[str, list[dict]] | None = None
        self.rows: list[dict] = []
        self.image_a: np.ndarray | None = None
        self.image_b: np.ndarray | None = None
        self._photos: list[ImageTk.PhotoImage] = []
        self._zoom_photo: ImageTk.PhotoImage | None = None
        self._zoom_win: tk.Toplevel | None = None
        self._busy = False
        self._resize_job: str | None = None
        self._click_a: list[tuple[float, float, float, float, int]] = []
        self._click_b: list[tuple[float, float, float, float, int]] = []
        self._pose_a = (0, 0)
        self._pose_b = (0, 0)

        guide = ttk.Label(
            self,
            text=(
                "选两张这次扫描的照片，点「对照这两张」，自己看。\n"
                "坐标：把一张照片里框的中心，按两张平台坐标换算到另一张。整框还能落进画面，就是「应重叠」。品红框和十字是算出来的位置。\n"
                "照片：另一张在 2.5 mm 以内有框，就是「两张都有框」。对照列写这两件事是否一样。\n"
                "绿 = 一样且都有框，红 = 坐标说该有但另一张没有，橙 = 两张都有框但坐标说进不了画面，紫 = 两边编号不一致。点一行会在下面放大。\n"
                "板上 X = 平台X + (像素行 − 画面高/2) / 44.2　　板上 Y = 平台Y + (像素列 − 画面宽/2) / 44.2"
            ),
            background="#eef4fb",
            padding=8,
            justify="left",
            wraplength=1100,
        )
        guide.pack(fill="x")

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=4)
        ttk.Label(bar, text="照片 A").pack(side="left")
        self.pick_a = ttk.Combobox(bar, state="readonly", width=42)
        self.pick_a.pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="照片 B").pack(side="left")
        self.pick_b = ttk.Combobox(bar, state="readonly", width=42)
        self.pick_b.pack(side="left", padx=(4, 12))
        self.compare_button = ttk.Button(bar, text="对照这两张", command=self._compare)
        self.compare_button.pack(side="left")
        self.show_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="也列出只在一张里的", variable=self.show_only, command=self._fill_tree).pack(side="left", padx=8)

        self.status = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.status, wraplength=1100, justify="left").pack(fill="x", padx=8)

        zoom = ttk.Frame(self, height=250)
        zoom.pack(side="bottom", fill="x", padx=8, pady=4)
        zoom.pack_propagate(False)
        self.detail = tk.StringVar(value="点左边一行，这里放大这一颗。品红十字是按平台坐标算出来的位置。")
        ttk.Label(zoom, textvariable=self.detail, wraplength=1100, justify="left").pack(fill="x")
        shots = ttk.Frame(zoom)
        shots.pack(fill="both", expand=True)
        self.crop_a = ttk.Label(shots)
        self.crop_a.pack(side="left", padx=6, pady=4)
        self.crop_b = ttk.Label(shots)
        self.crop_b.pack(side="left", padx=6, pady=4)
        self.crop_a.bind("<Button-1>", lambda _e: self._zoom("a"))
        self.crop_b.bind("<Button-1>", lambda _e: self._zoom("b"))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        left = ttk.Frame(body, width=560)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        columns = ("n", "kind", "theory", "actual", "same", "dist", "pixel")
        self.tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        headers = (
            ("n", "编号", 54), ("kind", "外形", 52), ("theory", "坐标", 70), ("actual", "照片", 90),
            ("same", "对照", 64), ("dist", "框距 mm", 70), ("pixel", "十字偏差 px", 90),
        )
        for name, title, width in headers:
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor="center")
        for tag, color in (("ok", "#2e7d32"), ("miss", "#c62828"), ("extra", "#e65100"), ("clash", "#6a1b9a"), ("only", "#616161")):
            self.tree.tag_configure(tag, foreground=color)
        scroll = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected())

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.title_a = ttk.Label(right, text="照片 A")
        self.title_a.pack(anchor="w")
        self.canvas_a = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0, height=220)
        self.canvas_a.pack(fill="both", expand=True)
        self.title_b = ttk.Label(right, text="照片 B")
        self.title_b.pack(anchor="w")
        self.canvas_b = tk.Canvas(right, bg="#1a1a1a", highlightthickness=0, height=220)
        self.canvas_b.pack(fill="both", expand=True)
        self.canvas_a.bind("<Configure>", self._schedule_paint)
        self.canvas_b.bind("<Configure>", self._schedule_paint)
        self.canvas_a.bind("<Button-1>", lambda event: self._click_canvas(event, "a"))
        self.canvas_b.bind("<Button-1>", lambda event: self._click_canvas(event, "b"))
        self._load_files()

    def _load_files(self) -> None:
        cached = latest_grouped_scan()
        if not cached:
            self.status.set("还没有识别结果。先在「整板对照」里识别一次，再来选照片。")
            return
        self.folder, payload = cached
        self.labels = _view_labels(payload)
        self._detections = None
        self._file_by_label = {}
        labels = []
        for path in sorted(self.folder.glob("line_*.jpg")):
            pose = _pose(path)
            if pose is None:
                continue
            label = f"{path.name}    平台 X {pose[0]}    Y {pose[1]}"
            self._file_by_label[label] = path
            labels.append(label)
        self.pick_a["values"] = labels
        self.pick_b["values"] = labels
        if labels:
            self.pick_a.set(labels[0])
            self.pick_b.set(labels[1] if len(labels) > 1 else labels[0])
        self.status.set(f"照片来自 {self.folder.name}。选两张，再点「对照这两张」。程序不会替你下结论，框和十字画在图上。")

    def _compare(self) -> None:
        if self._busy or self.folder is None:
            return
        label_a = self.pick_a.get()
        label_b = self.pick_b.get()
        path_a = self._file_by_label.get(label_a)
        path_b = self._file_by_label.get(label_b)
        if path_a is None or path_b is None:
            self.status.set("先选两张照片。")
            return
        if path_a == path_b:
            self.status.set("请选两张不同的照片。")
            return
        pose_a, pose_b = _pose(path_a), _pose(path_b)
        if pose_a is None or pose_b is None:
            self.status.set("文件名里没有平台坐标，没法按运动换算。")
            return
        self._busy = True
        self.compare_button.configure(state="disabled")
        self.status.set("正在读两张照片…")
        cached = self._detections
        labels = self.labels
        folder = self.folder

        def work() -> None:
            try:
                detections = cached
                if detections is None:
                    cache_path = folder / "detections.json"
                    if not cache_path.is_file():
                        raise RuntimeError("还没有逐张找到的框。先在「整板对照」里点「识别已有扫描」。")
                    grouped: dict[str, list[dict]] = {}
                    for det in json.loads(cache_path.read_text(encoding="utf-8")):
                        grouped.setdefault(det["file"], []).append(det)
                    detections = grouped
                image_a = ec._read(path_a)
                image_b = ec._read(path_b)
                if image_a is None or image_b is None:
                    raise RuntimeError("有一张照片没有读出来。")
                rows = build_overlap_rows(
                    detections.get(path_a.name, []), pose_a, (image_a.shape[1], image_a.shape[0]),
                    detections.get(path_b.name, []), pose_b, (image_b.shape[1], image_b.shape[0]),
                    labels,
                )
            except Exception as exc:
                self.after(0, lambda exc=exc: self._compare_failed(exc))
                return
            self.after(0, lambda: self._compare_done(
                detections, image_a, image_b, rows, pose_a, pose_b, path_a.name, path_b.name,
            ))

        threading.Thread(target=work, daemon=True).start()

    def _compare_failed(self, exc: Exception) -> None:
        self._busy = False
        self.compare_button.configure(state="normal")
        self.status.set(f"没有对照成：{exc}")

    def _compare_done(self, detections, image_a, image_b, rows, pose_a, pose_b, name_a, name_b) -> None:
        self._busy = False
        self.compare_button.configure(state="normal")
        self._detections = detections
        self.image_a = image_a
        self.image_b = image_b
        self.rows = rows
        self._pose_a, self._pose_b = pose_a, pose_b
        self.title_a.configure(text=f"照片 A　{name_a}　平台 X {pose_a[0]}　Y {pose_a[1]}")
        self.title_b.configure(text=f"照片 B　{name_b}　平台 X {pose_b[0]}　Y {pose_b[1]}")
        theory = sum(1 for row in rows if row["theory"] == "应重叠")
        both = sum(1 for row in rows if row["actual"] == "两张都有框")
        miss = sum(1 for row in rows if row["tag"] == "miss")
        extra = sum(1 for row in rows if row["tag"] == "extra")
        clash = sum(1 for row in rows if row["tag"] == "clash")
        self.status.set(
            f"坐标说应重叠 {theory} 颗，照片里两张都有框 {both} 颗。"
            f"应重叠但缺框 {miss}，不该重叠却都有框 {extra}，编号对不上 {clash}。"
            "点一行，看品红十字是否落在同一颗上。"
        )
        self._fill_tree()

    def _fill_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(self.rows):
            if row["tag"] == "only" and not self.show_only.get():
                continue
            dist = "" if row["dist"] is None else f"{row['dist']:.2f}"
            pixel = "" if row["pixel_err"] is None else f"{row['pixel_err']:.0f}"
            self.tree.insert("", "end", iid=str(index), tags=(row["tag"],), values=(
                row["n"], row["kind"], row["theory"], row["actual"], row["same"], dist, pixel,
            ))
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self.tree.see(children[0])
        else:
            self._paint()

    def _selected_row(self) -> dict | None:
        selected = self.tree.selection()
        if not selected or not self.rows:
            return None
        index = int(selected[0])
        if index < 0 or index >= len(self.rows):
            return None
        return self.rows[index]

    def _show_selected(self) -> None:
        self._paint()
        self._show_crops()

    def _schedule_paint(self, _event=None) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._paint)

    def _paint(self) -> None:
        self._resize_job = None
        if self.image_a is None or self.image_b is None:
            return
        selected = self.tree.selection()
        current = int(selected[0]) if selected else None
        self._click_a = self._draw_photo(self.canvas_a, self.image_a, "a", current)
        self._click_b = self._draw_photo(self.canvas_b, self.image_b, "b", current)

    def _draw_photo(self, canvas: tk.Canvas, image: np.ndarray, side: str, current: int | None) -> list[tuple[float, float, float, float, int]]:
        canvas.delete("all")
        view_w, view_h = canvas.winfo_width(), canvas.winfo_height()
        if view_w < 40 or view_h < 40:
            return []
        height, width = image.shape[:2]
        scale = min(view_w / width, view_h / height)
        new_w, new_h = max(int(width * scale), 1), max(int(height * scale), 1)
        ox = (view_w - new_w) // 2
        oy = (view_h - new_h) // 2
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        plate = np.full((view_h, view_w, 3), 26, np.uint8)
        plate[oy:oy + new_h, ox:ox + new_w] = small

        def pt(x: float, y: float) -> tuple[int, int]:
            return int(ox + x * scale), int(oy + y * scale)

        clicks = []
        for index, row in enumerate(self.rows):
            box = row["box_a"] if side == "a" else row["box_b"]
            if box is None:
                continue
            x0, y0 = pt(box[0], box[1])
            x1, y1 = pt(box[0] + box[2], box[1] + box[3])
            color = (0, 220, 255) if index == current else _BOX_COLOR[row["tag"]]
            thick = 3 if index == current else 1
            cv2.rectangle(plate, (x0, y0), (x1, y1), color, thick)
            clicks.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), index))
        for index, row in enumerate(self.rows):
            show_pred = index == current or row["tag"] in ("miss", "extra", "clash")
            if not show_pred:
                continue
            pred = row["pred_a"] if side == "a" else row["pred_b"]
            if pred is None:
                continue
            if pred[0] > width or pred[1] > height or pred[0] + pred[2] < 0 or pred[1] + pred[3] < 0:
                continue
            x0, y0 = pt(pred[0], pred[1])
            x1, y1 = pt(pred[0] + pred[2], pred[1] + pred[3])
            cv2.rectangle(plate, (x0, y0), (x1, y1), (220, 0, 220), 2)
            cx, cy = pt(pred[0] + pred[2] / 2, pred[1] + pred[3] / 2)
            cv2.drawMarker(plate, (cx, cy), (220, 0, 220), cv2.MARKER_CROSS, 16, 2)
        rgb = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._photos.append(photo)
        if len(self._photos) > 8:
            self._photos = self._photos[-4:]
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas._photo = photo
        return clicks

    def _click_canvas(self, event, side: str) -> None:
        boxes = self._click_a if side == "a" else self._click_b
        hit = None
        best = 1e9
        for x0, y0, x1, y1, index in boxes:
            if x0 - 4 <= event.x <= x1 + 4 and y0 - 4 <= event.y <= y1 + 4:
                gap = (event.x - (x0 + x1) / 2) ** 2 + (event.y - (y0 + y1) / 2) ** 2
                if gap < best:
                    best = gap
                    hit = index
        if hit is None or not self.tree.exists(str(hit)):
            return
        self.tree.selection_set(str(hit))
        self.tree.focus(str(hit))
        self.tree.see(str(hit))

    def _show_crops(self) -> None:
        self._photos_crop = []
        row = self._selected_row()
        if row is None or self.image_a is None or self.image_b is None:
            self.crop_a.configure(image="")
            self.crop_b.configure(image="")
            return
        crop_a = _marked_crop(self.image_a, row["box_a"], row["pred_a"] if row["box_b"] is not None else None, _BOX_COLOR[row["tag"]])
        crop_b = _marked_crop(self.image_b, row["box_b"], row["pred_b"], _BOX_COLOR[row["tag"]])
        self._crop_images = {"a": crop_a, "b": crop_b}
        self._put_crop(self.crop_a, crop_a)
        self._put_crop(self.crop_b, crop_b)
        bits = [f"编号 {row['n']}  {row['kind']}", f"坐标：{row['theory']}", f"照片：{row['actual']}", f"对照：{row['same']}"]
        if row["ax"] is not None:
            bits.append(f"A 板上 X {row['ax']:.2f}  Y {row['ay']:.2f}")
        if row["bx"] is not None:
            bits.append(f"B 板上 X {row['bx']:.2f}  Y {row['by']:.2f}")
        if row["dist"] is not None and row["actual"] == "两张都有框":
            bits.append(f"两框距离 {row['dist']:.2f} mm")
        elif row["dist"] is not None:
            bits.append(f"另一张最近的框 {row['dist']:.2f} mm，大于 2.5 mm，不算同一颗")
        if row["pixel_err"] is not None:
            bits.append(f"十字偏离实际框中心 {row['pixel_err']:.0f} px")
        self.detail.set("　　".join(bits) + "　　点击放大图可再放大。")

    def _put_crop(self, label: ttk.Label, image: np.ndarray | None) -> None:
        if image is None or image.size == 0:
            label.configure(image="")
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        picture.thumbnail((460, 200))
        photo = ImageTk.PhotoImage(picture)
        self._photos.append(photo)
        label.configure(image=photo)
        label._photo = photo

    def _zoom(self, side: str) -> None:
        image = getattr(self, "_crop_images", {}).get(side)
        if image is None:
            return
        if self._zoom_win is not None and self._zoom_win.winfo_exists():
            self._zoom_win.destroy()
        window = tk.Toplevel(self)
        window.title("照片 A 放大" if side == "a" else "照片 B 放大")
        self._zoom_win = window
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        long_side = max(picture.size)
        scale = min(2.0, max(1.0, 900 / long_side))
        if scale > 1.01:
            picture = picture.resize((int(picture.width * scale), int(picture.height * scale)), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(picture)
        ttk.Label(window, image=photo).pack()
        window._photo = photo


def _marked_crop(image: np.ndarray, box, pred, box_color: tuple[int, int, int]) -> np.ndarray | None:
    rects = [rect for rect in (box, pred) if rect is not None]
    if not rects:
        return None
    height, width = image.shape[:2]
    x0 = min(rect[0] for rect in rects)
    y0 = min(rect[1] for rect in rects)
    x1 = max(rect[0] + rect[2] for rect in rects)
    y1 = max(rect[1] + rect[3] for rect in rects)
    pad = 40
    xa = max(0, int(math.floor(x0)) - pad)
    ya = max(0, int(math.floor(y0)) - pad)
    xb = min(width, int(math.ceil(x1)) + pad)
    yb = min(height, int(math.ceil(y1)) + pad)
    if xb <= xa or yb <= ya:
        return None
    crop = image[ya:yb, xa:xb].copy()

    def local(rect) -> tuple[int, int, int, int]:
        return (
            int(round(rect[0] - xa)), int(round(rect[1] - ya)),
            int(round(rect[0] + rect[2] - xa)), int(round(rect[1] + rect[3] - ya)),
        )

    if pred is not None:
        left, top, right, bottom = local(pred)
        cv2.rectangle(crop, (left, top), (right, bottom), (220, 0, 220), 2)
        cv2.drawMarker(crop, ((left + right) // 2, (top + bottom) // 2), (220, 0, 220), cv2.MARKER_CROSS, 28, 2)
    if box is not None:
        left, top, right, bottom = local(box)
        cv2.rectangle(crop, (left, top), (right, bottom), box_color, 2)
    return crop
