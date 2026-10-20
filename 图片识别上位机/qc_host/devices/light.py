# -*- coding: utf-8 -*-
"""Coaxial / ring light supply. Stub until the lamp controller is wired."""

from __future__ import annotations


class CoaxialLight:
    def __init__(self) -> None:
        self._connected = False
        self.intensity = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def status_text(self) -> str:
        return "未接入（预留）"

    def connect(self, port: str = "") -> str:
        raise RuntimeError("同轴电源尚未接入。")

    def disconnect(self) -> None:
        self._connected = False

    def set_enabled(self, on: bool) -> None:
        raise RuntimeError("同轴电源尚未接入。")

    def set_intensity(self, value: int) -> None:
        raise RuntimeError("同轴电源尚未接入。")
