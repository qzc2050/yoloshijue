# -*- coding: utf-8 -*-
"""One-click window for the OpenCV step previews."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from qc_host.vision.lab import OUT, run

LABELS = {
    "01_gray.png": "1  灰度",
    "02_light.png": "2  环形光",
    "03_flat.png": "3  去掉环形光",
    "04_clahe.png": "4  局部均衡",
    "05_gold.png": "5  金色分割",
    "06_ridge.png": "6  亮纹",
    "07_copper.png": "7  开闭运算后的铜",
    "08_rings.png": "8  认出来的环",
}


class LabView:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("OpenCV 步骤")
        self.root.geometry("1180x760")
        self._photo: ImageTk.PhotoImage | None = None
        self._paths: list = []
        self._busy = False

        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill=tk.X)
        self.rerun_btn = ttk.Button(bar, text="重新生成", command=self.rerun)
        self.rerun_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="用系统看图打开", command=self.open_external).pack(side=tk.LEFT, padx=(8, 0))
        self.status = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.status).pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        self.list = tk.Listbox(body, width=24, font=("Microsoft YaHei", 12), activestyle="none")
        self.list.pack(side=tk.LEFT, fill=tk.Y)
        self.list.bind("<<ListboxSelect>>", self._on_select)
        self.canvas = tk.Canvas(body, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.canvas.bind("<Configure>", lambda _e: self._show_current())

        self.refresh()
        if not self._paths:
            self.rerun()

    def refresh(self) -> None:
        self._paths = sorted(OUT.glob("*.png")) if OUT.is_dir() else []
        self.list.delete(0, tk.END)
        for path in self._paths:
            self.list.insert(tk.END, LABELS.get(path.name, path.stem))
        if self._paths:
            self.list.selection_set(0)
            self.list.activate(0)
            self._show_current()
            self.status.set(str(OUT))
        else:
            self.canvas.delete("all")
            self.status.set("还没有步骤图")

    def _selected(self):
        if not self.list.curselection():
            return None
        index = int(self.list.curselection()[0])
        if index < 0 or index >= len(self._paths):
            return None
        return self._paths[index]

    def _on_select(self, _event: object) -> None:
        self._show_current()

    def _show_current(self) -> None:
        path = self._selected()
        if path is None:
            return
        image = Image.open(path).convert("RGB")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self._photo)

    def open_external(self) -> None:
        path = self._selected()
        if path is not None:
            import os

            os.startfile(path)  # noqa: S606

    def rerun(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.rerun_btn.state(["disabled"])
        self.status.set("正在从原图生成各步骤…")

        def work() -> None:
            try:
                run()
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: self._finish_error(exc))
                return
            self.root.after(0, self._finish_ok)

        threading.Thread(target=work, daemon=True).start()

    def _finish_ok(self) -> None:
        self._busy = False
        self.rerun_btn.state(["!disabled"])
        self.refresh()

    def _finish_error(self, exc: Exception) -> None:
        self._busy = False
        self.rerun_btn.state(["!disabled"])
        self.status.set("生成失败")
        messagebox.showerror("生成失败", str(exc))

    def mainloop(self) -> None:
        self.root.mainloop()


def main() -> None:
    LabView().mainloop()


if __name__ == "__main__":
    main()
