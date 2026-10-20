# -*- coding: utf-8 -*-
"""Camera device. Wraps the existing Hikrobot grabber."""

from __future__ import annotations

import sys
from ctypes import byref, memset, sizeof
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hikrobot_camera import HikrobotCamera  # noqa: E402


class Camera:
    DEFAULT_IP = "192.168.0.100"

    def __init__(self) -> None:
        self._hik = HikrobotCamera()

    @property
    def connected(self) -> bool:
        return self._hik.cam is not None

    @property
    def info(self) -> str:
        return self._hik.info or ""

    def enumerate(self) -> List[Tuple[str, str, str]]:
        return self._hik.enumerate()

    def connect(self, ip: str) -> str:
        return self._hik.connect(ip.strip())

    def disconnect(self) -> None:
        self._hik.disconnect()

    def grab_bgr(self, timeout_ms: int = 2500):
        return self._hik.grab_bgr(timeout_ms=timeout_ms)

    def get_float(self, name: str) -> Optional[float]:
        cam = self._hik.cam
        if cam is None:
            return None
        try:
            from CameraParams_header import MVCC_FLOATVALUE
        except ImportError:
            return None
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        if cam.MV_CC_GetFloatValue(name, value) != 0:
            return None
        return float(value.fCurValue)

    def exposure_us(self) -> Optional[float]:
        return self.get_float("ExposureTime")

    def gain_db(self) -> Optional[float]:
        return self.get_float("Gain")
