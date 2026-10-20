# -*- coding: utf-8 -*-
"""XY / gantry stage. Stub until the factory controller is wired."""

from __future__ import annotations


class MotionStage:
    def __init__(self) -> None:
        self._connected = False
        self.x_mm = 0.0
        self.y_mm = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def status_text(self) -> str:
        return "未接入（预留）"

    def connect(self, port: str = "") -> str:
        raise RuntimeError("移动平台尚未接入。")

    def disconnect(self) -> None:
        self._connected = False

    def home(self) -> None:
        raise RuntimeError("移动平台尚未接入。")

    def move_xy(self, x_mm: float, y_mm: float) -> None:
        raise RuntimeError("移动平台尚未接入。")
