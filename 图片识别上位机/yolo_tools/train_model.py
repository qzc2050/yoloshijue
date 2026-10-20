# -*- coding: utf-8 -*-
"""One window: label a scan by chamber number, then train from scratch or from the last model."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import board_views as bv
import chamber_data as cd
import export_chamber_crops as ec
import hole_assist as ha
import project_config

HERE = Path(__file__).resolve().parent
DATA = cd.YOLO_ROOT / "datasets" / "chamber_train"
LABEL_FILE = "chamber_labels.json"


def _append(text: tk.Text, line: str) -> None:
    text.configure(state="normal")
    text.insert("end", line + "\n")
    text.see("end")
    text.configure(state="disabled")


def _view_key(view: dict) -> str:
    return f"{view['file']}|{int(view['x'])}|{int(view['y'])}"


def _load_labels(folder: Path, chambers: list[dict]) -> dict[str, str]:
    """Photo key -> class. Older files stored one class for a whole chamber number."""
    path = folder / LABEL_FILE
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("views"), dict):
        return {key: value for key, value in raw["views"].items() if value in cd.CLASSES}
    labels: dict[str, str] = {}
    id_labels: dict[int, str] = {}
    for key, value in raw.items():
        if value in cd.CLASSES:
            try:
                id_labels[int(key)] = value
            except ValueError:
                continue
    for chamber in chambers:
        label = id_labels.get(chamber.get("id"))
        if not label:
            continue
        for view in chamber.get("views") or []:
            labels[_view_key(view)] = label
    return labels


def _save_labels(folder: Path, labels: dict[str, str]) -> None:
    path = folder / LABEL_FILE
    path.write_text(json.dumps({"views": dict(sorted(labels.items()))}, ensure_ascii=False, indent=2), encoding="utf-8")


def export_dataset(folder: Path, chambers: list[dict], labels: dict[str, str], out: Path = DATA) -> dict:
    """One label per photo. Photos of the same chamber number stay together when they can."""
    image_counts: dict[str, int] = defaultdict(int)
    by_id: dict[int, list[tuple[dict, str, int]]] = defaultdict(list)
    for chamber in chambers:
        for view in chamber.get("views") or []:
            label = labels.get(_view_key(view))
            if label not in cd.CLASSES:
                continue
            image_counts[label] += 1
            by_id[chamber["id"]].append((view, label, int(chamber["n"])))
    usable = [name for name in cd.CLASSES if image_counts.get(name, 0) >= cd.MIN_PER_CLASS]
    if len(usable) < 2:
        detail = "，".join(f"{cd.CLASSES[name]} {image_counts.get(name, 0)} 张" for name in cd.CLASSES)
        raise ValueError(f"样本不够：至少要有两类、每类至少 {cd.MIN_PER_CLASS} 张。\n现在：{detail}")

    rng = random.Random(0)
    order = [ident for ident, rows in by_id.items() if any(label in usable for _view, label, _number in rows)]
    rng.shuffle(order)
    order.sort(key=lambda ident: len(by_id[ident]))
    total = sum(len(by_id[ident]) for ident in order)
    target = max(1, int(round(cd.VAL_FRACTION * total)))
    val_ids: set[int] = set()
    held = 0
    for ident in order:
        count = len(by_id[ident])
        if held >= target:
            break
        if held and held + count >= total:
            continue
        val_ids.add(ident)
        held += count
    records = []
    for ident in order:
        side = "val" if ident in val_ids else "train"
        for view, label, number in by_id[ident]:
            if label in usable:
                records.append({"side": side, "view": view, "label": label, "number": number, "id": ident})
    summary = {"train": {}, "val": {}, "skipped": {}, "shared": []}
    for name in cd.CLASSES:
        if name not in usable:
            summary["skipped"][name] = image_counts.get(name, 0)
            continue
        if any(row["label"] == name and row["side"] == "val" for row in records):
            continue
        for row in records:
            if row["label"] == name and row["side"] == "train":
                row["side"] = "val"
                summary["shared"].append(row["number"])
                break
    if out.exists():
        shutil.rmtree(out)
    photos: dict[str, object] = {}
    for row in records:
        view = row["view"]
        image = photos.get(view["file"])
        if image is None and view["file"] not in photos:
            image = ec._read(folder / view["file"])
            photos[view["file"]] = image
        if image is None:
            continue
        hit = ec.Hit(view["score"], view["vertical"], view["x"], view["y"], view["w"], view["h"])
        crop = cd.square(ec.crop_chamber(image, hit))
        filename = f"n{row['number']:03d}_{Path(view['file']).stem}_{int(view['x'])}_{int(view['y'])}.png"
        cd.write_image(out / row["side"] / row["label"] / filename, crop)
        summary[row["side"]][row["label"]] = summary[row["side"]].get(row["label"], 0) + 1
    return summary


class TrainModelWindow:
    def __init__(self, app) -> None:
        self.app = app
        self.folder: Path | None = None
        self.labels: dict[str, str] = {}
        self.picked_key: str | None = None
        self._stopped = False
        self._assist_token = 0
        self._assist_running = False
        self._assist_key: str | None = None
        self._assist_label: str | None = None
        self._assist_vis = None
        self._assist_busy = False
        self.show_frame = tk.BooleanVar(value=True)
        self.win = tk.Toplevel(app.root)
        self.win.title("训练模型")
        self.win.geometry("1460x900")
        self.win.protocol("WM_DELETE_WINDOW", self._close)

        bar = ttk.Frame(self.win)
        bar.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Button(bar, text="最近一次扫描", command=self._use_latest).pack(side="left")
        ttk.Button(bar, text="选择扫描文件夹", command=self._pick_folder).pack(side="left", padx=6)
        self.path_label = ttk.Label(bar, text="还没有选择扫描")
        self.path_label.pack(side="left", padx=10)

        self.panel = bv.GerberBoardPanel(self.win, embedded=True)
        self.panel.pack(fill="both", expand=True)
        self.panel.view_name = lambda view: cd.CLASSES.get(self.labels.get(_view_key(view)), "还没标")
        self.panel.view_picked = lambda view: _view_key(view) == self.picked_key
        self.panel.pick_view = self._pick_view
        self.panel.zoom_image = self._zoom_source
        self.panel.on_thumbs = self._start_assist
        self.panel.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select(), add="+")

        assist = ttk.LabelFrame(self.win, text="圆孔画框")
        assist.pack(fill="x", padx=8, pady=(4, 0))
        head = ttk.Frame(assist)
        head.pack(fill="x", padx=6, pady=(4, 0))
        self.assist_title = tk.Label(head, text="点一张照片后，这里给出描边和各项参数。", anchor="w", justify="left",
                                     font=("Microsoft YaHei UI", 11, "bold"), wraplength=1100, fg="#1a3d6d")
        self.assist_title.pack(side="left", fill="x", expand=True)
        tk.Checkbutton(head, text="显示描线", variable=self.show_frame, command=self._toggle_assist_view,
                       font=("Microsoft YaHei UI", 12, "bold"), fg="#1a3d6d",
                       activeforeground="#1a3d6d").pack(side="right")
        self.assist_text = tk.Text(assist, height=4, wrap="word", font=("Microsoft YaHei UI", 10), state="disabled")
        self.assist_text.pack(fill="x", padx=6, pady=4)
        arow = ttk.Frame(assist)
        arow.pack(fill="x", padx=6, pady=(0, 6))
        self.assist_apply = ttk.Button(arow, text="采用圆孔结果", command=self._apply_assist, state="disabled")
        self.assist_apply.pack(side="left")
        self.assist_batch = ttk.Button(arow, text="未标注的都按建议先标上", command=self._batch_assist)
        self.assist_batch.pack(side="left", padx=8)

        labels = ttk.Frame(self.win)
        labels.pack(fill="x", padx=8, pady=4)
        self.chamber_label = ttk.Label(labels, text="先点下面的一张照片，再标这一张", font=("Microsoft YaHei UI", 12, "bold"))
        self.chamber_label.pack(side="left")
        for key, title in (("hege", "1  合格"), ("shenmo", "2  渗墨"), ("quemo", "3  缺墨"), ("kanbuqing", "4  看不清")):
            ttk.Button(labels, text=title, command=lambda key=key: self._set_label(key)).pack(side="left", padx=(12, 0))
        ttk.Button(labels, text="清除", command=lambda: self._set_label(None)).pack(side="left", padx=(12, 0))

        train = ttk.Frame(self.win)
        train.pack(fill="x", padx=8, pady=(0, 4))
        self.counts = ttk.Label(train, text="")
        self.counts.pack(side="left")
        self.full_button = ttk.Button(train, text="全量训练", style="Big.TButton", command=lambda: self._start(False))
        self.full_button.pack(side="right")
        self.inc_button = ttk.Button(train, text="增量训练", style="Big.TButton", command=lambda: self._start(True))
        self.inc_button.pack(side="right", padx=(0, 8))
        self.stop_button = ttk.Button(train, text="停止", command=self._stop, state="disabled")
        self.stop_button.pack(side="right", padx=(0, 8))
        self.stage = ttk.Label(train, text="")
        self.stage.pack(side="right", padx=(0, 12))

        self.log = tk.Text(self.win, height=6, wrap="word", font=("Microsoft YaHei UI", 10), state="disabled")
        self.log.pack(fill="x", padx=8, pady=(0, 8))
        self.win.bind("<KeyPress>", self._key)
        self._use_latest()

    def _close(self) -> None:
        proc = self.app.train_proc
        if proc is not None and proc.poll() is None:
            if not messagebox.askyesno("正在训练", "训练还没结束。关闭这个窗口会停掉训练。", parent=self.win):
                return
            self._stop()
        self.win.destroy()

    def _use_latest(self) -> None:
        found = bv.latest_grouped_scan()
        if found is None:
            self.path_label.configure(text="还没有带编号的扫描")
            return
        self._open_folder(found[0])

    def _pick_folder(self) -> None:
        name = filedialog.askdirectory(title="选择一次扫描的文件夹", parent=self.win)
        if name:
            self._open_folder(Path(name))

    def _open_folder(self, folder: Path) -> None:
        groups = folder / "groups.json"
        if not groups.is_file() or not any(folder.glob("line_*.jpg")):
            messagebox.showinfo("不能对照", "这个文件夹里没有整板编号和扫描照片，坐标图出不来。", parent=self.win)
            return
        payload = json.loads(groups.read_text(encoding="utf-8"))
        self.folder = folder
        self.picked_key = None
        self.labels = _load_labels(folder, payload.get("chambers") or [])
        self.panel.data = payload
        self._sync_marks()
        self.panel.show(folder, payload)
        self.path_label.configure(text=str(folder))
        self._refresh_counts()

    def _selected(self) -> dict | None:
        selected = self.panel.tree.selection()
        if not selected or not self.panel.data:
            return None
        number = int(selected[0])
        return next((item for item in self.panel.data["chambers"] if item.get("n") == number), None)

    def _sync_marks(self) -> None:
        marks: dict[int, str] = {}
        texts: dict[int, str] = {}
        if self.panel.data:
            for chamber in self.panel.data["chambers"]:
                found = [self.labels[key] for view in chamber.get("views") or [] if (key := _view_key(view)) in self.labels]
                if not found:
                    continue
                names = list(dict.fromkeys(cd.CLASSES[label] for label in found))
                texts[chamber["id"]] = names[0] if len(names) == 1 else "+".join(names)
                marks[chamber["id"]] = found[0] if len(names) == 1 and len(found) == len(chamber.get("views") or []) else "mixed"
        self.panel.labels = marks
        self.panel.label_text = texts

    def _show_thumbs(self) -> None:
        self._on_select(redraw=False, assist=False)
        self.panel._show_selected()
        self._start_assist()

    def _pick_view(self, view: dict) -> None:
        self.picked_key = _view_key(view)
        self._on_select(redraw=False, assist=False)
        self.panel.refresh_pick()
        self._start_assist()

    def _on_select(self, redraw: bool = True, assist: bool = True) -> None:
        chamber = self._selected()
        if chamber is None:
            return
        views = chamber.get("views") or []
        keys = [_view_key(view) for view in views]
        if self.picked_key not in keys:
            pending = [key for key in keys if key not in self.labels]
            self.picked_key = pending[0] if pending else (keys[0] if keys else None)
            redraw = True
        view = next((item for item in views if _view_key(item) == self.picked_key), None)
        if view is None:
            self.chamber_label.configure(text=f"编号 {chamber['n']}　没有照片")
            return
        label = cd.CLASSES.get(self.labels.get(self.picked_key), "还没标")
        index = keys.index(self.picked_key) + 1
        self.chamber_label.configure(
            text=f"编号 {chamber['n']}　第 {index} 张 / 共 {len(views)} 张　电机 {view['mx']}, {view['my']}　{label}"
        )
        if redraw:
            self.panel._show_selected()
        if assist:
            self._start_assist()

    def _set_label(self, key: str | None) -> None:
        if self.folder is None or self.picked_key is None:
            return
        if key is None:
            self.labels.pop(self.picked_key, None)
        else:
            self.labels[self.picked_key] = key
        _save_labels(self.folder, self.labels)
        self._sync_marks()
        self.panel.note_labels()
        self._refresh_counts()
        self._advance()

    def _advance(self) -> None:
        chamber = self._selected()
        if chamber is not None:
            for view in chamber.get("views") or []:
                key = _view_key(view)
                if key not in self.labels:
                    self.picked_key = key
                    self._show_thumbs()
                    return
        if not self.panel.data:
            self._show_thumbs()
            return
        numbers = [int(iid) for iid in self.panel.tree.get_children()]
        selected = self.panel.tree.selection()
        start = int(selected[0]) if selected else 0
        ordered = numbers[numbers.index(start) + 1:] + numbers[: numbers.index(start) + 1] if start in numbers else numbers
        by_number = {item.get("n"): item for item in self.panel.data["chambers"]}
        for number in ordered:
            item = by_number.get(number)
            if item and any(_view_key(view) not in self.labels for view in item.get("views") or []):
                self.picked_key = None
                self.panel.tree.selection_set(str(number))
                self.panel.tree.focus(str(number))
                self.panel.tree.see(str(number))
                return
        self._show_thumbs()

    def _refresh_counts(self) -> None:
        counts = {name: 0 for name in cd.CLASSES}
        for label in self.labels.values():
            if label in counts:
                counts[label] += 1
        text = "已标注：" + "，".join(f"{cd.CLASSES[name]} {counts[name]} 张" for name in cd.CLASSES)
        ready = sum(counts[name] >= cd.MIN_PER_CLASS for name in cd.CLASSES) >= 2
        self.counts.configure(text=text + ("，可以训练" if ready else f"，至少两类并且每类 {cd.MIN_PER_CLASS} 张"))

    def _view_now(self) -> dict | None:
        chamber = self._selected()
        if chamber is None or self.picked_key is None:
            return None
        return next((view for view in chamber.get("views") or [] if _view_key(view) == self.picked_key), None)

    def _zoom_source(self, view: dict, crop):
        if self.show_frame.get() and self._assist_key == _view_key(view) and self._assist_vis is not None:
            return self._assist_vis
        return crop

    def _set_assist(self, title: str, detail: str, label: str | None) -> None:
        colors = {"hege": "#2e7d32", "shenmo": "#c62828", "quemo": "#b28704", "kanbuqing": "#e65100"}
        self.assist_title.configure(text=title, fg=colors.get(label, "#1a3d6d"))
        self.assist_text.configure(state="normal")
        self.assist_text.delete("1.0", "end")
        if detail:
            self.assist_text.insert("1.0", detail)
        self.assist_text.configure(state="disabled")

    def _start_assist(self) -> None:
        view = self._view_now()
        if view is None:
            return
        self.panel.follow_zoom()
        key = _view_key(view)
        if key == self._assist_key and self._assist_vis is not None:
            self._paint_assist()
            return
        crop = self.panel.crop_of(view)
        if crop is None:
            return
        self._assist_token += 1
        token = self._assist_token
        self.win.after(250, lambda token=token, key=key, crop=crop: self._assist_after(token, key, crop))

    def _assist_after(self, token: int, key: str, crop) -> None:
        if token != self._assist_token or not self.win.winfo_exists():
            return
        if self._assist_running:
            return
        self._assist_running = True
        self._assist_key = key
        self._assist_label = None
        self._assist_vis = None
        if not self._assist_busy:
            self.assist_apply.configure(state="disabled")
        self._set_assist("圆孔画框正在分析这张…", "", None)

        def done(result: ha.Assist) -> None:
            self._assist_running = False
            if not self.win.winfo_exists():
                return
            if token != self._assist_token or self.picked_key != key:
                self._start_assist()
                return
            self._assist_label = result.label
            self._assist_vis = result.overlay
            self._set_assist(result.title, result.detail, result.label)
            if result.label and not self._assist_busy:
                self.assist_apply.configure(state="normal")
            self._paint_assist()

        def failed(_exc: Exception, _detail: str) -> None:
            self._assist_running = False
            if self.win.winfo_exists() and token != self._assist_token:
                self._start_assist()

        self.app.run_bg(lambda: ha.analyse(crop), done, failed)

    def _paint_assist(self) -> None:
        view = self._view_now()
        if view is None or self._assist_key != _view_key(view):
            return
        image = self._assist_vis if self.show_frame.get() and self._assist_vis is not None else self.panel.crop_of(view)
        if image is not None:
            self.panel.set_card_image(view, image)
            self.panel.follow_zoom()

    def _toggle_assist_view(self) -> None:
        self._paint_assist()

    def _apply_assist(self) -> None:
        if self._assist_label:
            self._set_label(self._assist_label)

    def _batch_assist(self) -> None:
        if self._assist_busy or self.folder is None or self.panel.data is None:
            return
        pending = []
        for chamber in self.panel.data["chambers"]:
            for view in chamber.get("views") or []:
                if _view_key(view) not in self.labels:
                    pending.append(view)
        if not pending:
            messagebox.showinfo("圆孔画框", "没有未标注的照片。", parent=self.win)
            return
        if not messagebox.askyesno(
            "按圆孔建议先标",
            f"用圆孔画框给 {len(pending)} 张还没标注的照片先写上建议。\n框没套准的会跳过。写上之后还可以改。继续吗？",
            parent=self.win,
        ):
            return
        self._assist_busy = True
        self._assist_token += 1
        self.assist_apply.configure(state="disabled")
        self.assist_batch.configure(state="disabled")
        folder = self.folder
        total = len(pending)

        def work():
            photos: dict[str, object] = {}
            updates: dict[str, str] = {}
            counts: dict[str, int] = {}
            skipped = 0
            for index, view in enumerate(pending, 1):
                image = photos.get(view["file"])
                if view["file"] not in photos:
                    image = ec._read(folder / view["file"])
                    photos[view["file"]] = image
                crop = None
                if image is not None:
                    hit = ec.Hit(view["score"], view["vertical"], view["x"], view["y"], view["w"], view["h"])
                    crop = ec.crop_chamber(image, hit)
                result = ha.analyse(crop) if crop is not None else None
                if result is None or result.label is None:
                    skipped += 1
                else:
                    updates[_view_key(view)] = result.label
                    counts[result.label] = counts.get(result.label, 0) + 1
                if index == 1 or index % 10 == 0 or index == total:
                    self.app.later(self._assist_progress, index, total)
            return updates, counts, skipped

        def done(pack) -> None:
            updates, counts, skipped = pack
            applied = 0
            for key, label in updates.items():
                if key in self.labels:
                    continue
                self.labels[key] = label
                applied += 1
            if applied and self.folder is not None:
                _save_labels(self.folder, self.labels)
            self._assist_busy = False
            self.assist_batch.configure(state="normal")
            self._sync_marks()
            self.panel.note_labels()
            self._refresh_counts()
            self._show_thumbs()
            summary = "，".join(f"{cd.CLASSES[name]} {counts.get(name, 0)}" for name in cd.CLASSES if counts.get(name))
            messagebox.showinfo("圆孔画框", f"写上了 {applied} 张（{summary or '没有'}）。跳过 {skipped} 张。", parent=self.win)

        def failed(exc: Exception, detail: str) -> None:
            self._assist_busy = False
            self.assist_batch.configure(state="normal")
            messagebox.showerror("出错了", f"{exc}\n\n{detail[-800:]}", parent=self.win)

        self.app.run_bg(work, done, failed)

    def _assist_progress(self, done: int, total: int) -> None:
        if not self.win.winfo_exists() or not self._assist_busy:
            return
        self._set_assist(f"圆孔画框正在先标未标注的：{done}/{total}", "标完后可以逐张改。框没套准的会留空。", None)

    def _key(self, event) -> None:
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            return
        keys = {"1": "hege", "2": "shenmo", "3": "quemo", "4": "kanbuqing"}
        if event.char in keys:
            self._set_label(keys[event.char])

    def _lock(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.full_button.configure(state=state)
        self.inc_button.configure(state=state)
        self.stop_button.configure(state="normal" if busy else "disabled")

    def _start(self, incremental: bool) -> None:
        proc = self.app.train_proc
        if proc is not None and proc.poll() is None:
            return
        if self.folder is None or self.panel.data is None:
            messagebox.showinfo("还不能训练", "先打开一次扫描。", parent=self.win)
            return
        weights = None
        if incremental:
            info = cd.read_latest()
            raw = info.get("pt") if info else None
            weights = Path(raw) if raw else None
            if weights is None or not weights.is_file():
                messagebox.showinfo("还不能增量训练", "还没有上次训好的模型。先做一次全量训练。", parent=self.win)
                return
        self._lock(True)
        self.stage.configure(text="整理已标注的截图…")
        _append(self.log, "—— 增量训练 ——" if incremental else "—— 全量训练 ——")
        folder, chambers, labels = self.folder, list(self.panel.data["chambers"]), dict(self.labels)

        def work() -> dict:
            return export_dataset(folder, chambers, labels)

        def done(summary: dict) -> None:
            for side, title in (("train", "给它学"), ("val", "留着考它")):
                _append(self.log, f"{title}：" + "，".join(f"{cd.CLASSES[key]} {value} 张" for key, value in summary[side].items()))
            if summary["skipped"]:
                _append(self.log, "张数不够、这次不学：" + "，".join(f"{cd.CLASSES[key]} {value} 张" for key, value in summary["skipped"].items()))
            if summary["shared"]:
                _append(self.log, "这些编号只有一颗，考试用了它的一张图：" + "，".join(str(number) for number in summary["shared"]))
            self._launch(weights)

        def failed(exc: Exception, _detail: str) -> None:
            self._lock(False)
            self.stage.configure(text="")
            messagebox.showwarning("还不能训练", str(exc), parent=self.win)

        self.app.run_bg(work, done, failed)

    def _launch(self, weights: Path | None) -> None:
        args = [sys.executable, str(HERE / "train_classifier.py"), "--data", str(DATA), "--model", "n", "--epochs", "30", "--imgsz", "512"]
        self._stopped = False
        if weights is not None:
            args += ["--weights", str(weights)]
        temp_dir = project_config.machine().temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
               "TEMP": str(temp_dir), "TMP": str(temp_dir)}
        self.app.train_proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=str(HERE.parent), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        threading.Thread(target=self._read, args=(self.app.train_proc,), daemon=True).start()

    def _read(self, proc: subprocess.Popen) -> None:
        cd.RUNS.mkdir(parents=True, exist_ok=True)
        tail: list[str] = []
        with (cd.RUNS / "train_log.txt").open("w", encoding="utf-8") as log:
            for line in proc.stdout:
                line = line.rstrip()
                log.write(line + "\n")
                if line.startswith("@@"):
                    tag, _, payload = line[2:].partition(" ")
                    self.app.later(self._message, tag, json.loads(payload) if payload else None)
                elif line.strip():
                    tail = (tail + [line])[-12:]
        code = proc.wait()
        self.app.later(self._finished, code, tail, proc)

    def _message(self, tag: str, data) -> None:
        if not self.win.winfo_exists():
            return
        if tag == "STAGE":
            self.stage.configure(text=data)
            _append(self.log, data)
        elif tag == "EPOCH" and data["epoch"] <= data["epochs"]:
            self.stage.configure(text=f"正在学第 {data['epoch']} / {data['epochs']} 遍")
            _append(self.log, f"第 {data['epoch']:>3} 遍：误差 {data['loss']:.3f}　考试准确率 {data['acc'] * 100:.1f}%")
        elif tag == "DONE":
            val = data.get("val") or {}
            _append(self.log, f"训练完成。考试 {val.get('right', 0)}/{val.get('total', 0)}，漏检 {val.get('missed', 0)}，误报 {val.get('false_alarm', 0)}")

    def _finished(self, code: int, tail: list[str], proc: subprocess.Popen) -> None:
        if self.app.train_proc is proc:
            self.app.train_proc = None
        if not self.win.winfo_exists():
            return
        if self._stopped:
            self._stopped = False
            self._lock(False)
            return
        self._lock(False)
        if code != 0:
            self.stage.configure(text="训练出错了")
            _append(self.log, "训练出错，最后几行：\n" + "\n".join(tail))

    def _stop(self) -> None:
        proc = self.app.train_proc
        if proc is not None and proc.poll() is None:
            self._stopped = True
            self.app.train_proc = None
            proc.terminate()
            self.stage.configure(text="已停止")
            _append(self.log, "已手动停止。")
            self._lock(False)


def open_train_window(app) -> tk.Toplevel:
    return TrainModelWindow(app).win
