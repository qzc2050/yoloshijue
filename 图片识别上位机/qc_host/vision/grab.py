# -*- coding: utf-8 -*-
"""Take one frame from the Hikrobot camera with whatever exposure, gain and white balance it has now.

HikrobotCamera.connect forces Gain 6.0 and ExposureTime 40000. While lighting is being tuned the
camera settings belong to the operator, so those two writes are skipped here and the camera is
closed again right after the frame.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from ctypes import byref, memset, sizeof

import cv2
import numpy as np

KEEP_KEYS = ("Gain", "ExposureTime")
WARMUP_FRAMES = 3


@dataclass
class CameraSettings:
    exposure_us: float | None
    gain_db: float | None
    white_balance_auto: str | None


def _read_settings(cam) -> CameraSettings:
    from MvCameraControl_class import MVCC_ENUMVALUE, MVCC_FLOATVALUE

    def read_float(key: str) -> float | None:
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        return float(value.fCurValue) if cam.MV_CC_GetFloatValue(key, value) == 0 else None

    balance = MVCC_ENUMVALUE()
    memset(byref(balance), 0, sizeof(balance))
    names = {0: "关", 1: "连续自动", 2: "单次自动"}
    white = names.get(int(balance.nCurValue), str(balance.nCurValue)) if cam.MV_CC_GetEnumValue("BalanceWhiteAuto", balance) == 0 else None
    return CameraSettings(read_float("ExposureTime"), read_float("Gain"), white)


def grab_frame() -> tuple[np.ndarray, CameraSettings]:
    import hikrobot_camera as hc

    original = hc.MvCamera.MV_CC_SetFloatValue

    def keep_operator_settings(self, key, value):
        if key in KEEP_KEYS:
            return 0
        return original(self, key, value)

    hc.MvCamera.MV_CC_SetFloatValue = keep_operator_settings
    camera = hc.HikrobotCamera()
    try:
        devices = camera.enumerate()
        if not devices:
            raise RuntimeError("没有找到相机。")
        camera.connect(devices[0][0])
        image = None
        for _ in range(WARMUP_FRAMES):
            image = camera.grab_bgr()
        settings = _read_settings(camera.cam)
    finally:
        camera.disconnect()
        hc.MvCamera.MV_CC_SetFloatValue = original
    return image, settings


def _downscale(frame: np.ndarray, max_side: int = 1400) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(width, height))
    if scale >= 0.999:
        return frame
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


class CameraSession:
    """Keep the camera open so the operator can preview and save many frames."""

    def __init__(self) -> None:
        self.info = ""
        self._camera = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._serial = 0
        self._full: np.ndarray | None = None
        self._center = 0.0
        self._overall = 0.0
        self._error: str | None = None

    @property
    def connected(self) -> bool:
        return self._running

    def start(self) -> str:
        if self._running:
            return self.info
        self._stop.clear()
        with self._lock:
            self._error = None
            self._serial = 0
            self._full = None
        ready = threading.Event()
        box: dict = {}

        def body() -> None:
            try:
                box["info"] = self._open()
                if self._stop.is_set():
                    ready.set()
                    return
                self._running = True
                ready.set()
                self._loop()
            except Exception as exc:  # reported by start()
                box["error"] = exc
                ready.set()
            finally:
                self._running = False
                camera = self._camera
                self._camera = None
                if camera is not None:
                    try:
                        camera.disconnect()
                    except Exception:
                        pass

        self._thread = threading.Thread(target=body, name="yolo-capture", daemon=True)
        self._thread.start()
        if not ready.wait(25):
            self.stop()
            raise TimeoutError("连接相机超过 25 秒，没有连上。")
        if "error" in box:
            self.stop()
            raise box["error"]
        self.info = str(box.get("info", ""))
        return self.info

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        thread = self._thread
        if wait and thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)
        if not wait:
            return
        self._thread = None
        self._running = False

    def peek(self) -> tuple[int, np.ndarray | None, float, float]:
        """Latest full-resolution frame, so the preview can zoom into real camera pixels."""
        with self._lock:
            return self._serial, self._full, self._center, self._overall

    def consume_error(self) -> str | None:
        with self._lock:
            error, self._error = self._error, None
            return error

    def take(self) -> tuple[np.ndarray, float, float] | None:
        """Latest full-resolution frame. The array is not copied; do not modify it."""
        with self._lock:
            if self._full is None:
                return None
            return self._full, self._center, self._overall

    def _open(self) -> str:
        import hikrobot_camera as hc

        original = hc.MvCamera.MV_CC_SetFloatValue

        def keep_operator_settings(cam, key, value):
            if key in KEEP_KEYS:
                return 0
            return original(cam, key, value)

        hc.MvCamera.MV_CC_SetFloatValue = keep_operator_settings
        camera = hc.HikrobotCamera()
        try:
            devices = camera.enumerate()
            if not devices:
                raise RuntimeError("没有找到相机。检查网线，并确认 MVS 已经关闭。")
            info = camera.connect(devices[0][0])
        finally:
            hc.MvCamera.MV_CC_SetFloatValue = original
        self._camera = camera
        return info

    def _loop(self) -> None:
        camera = self._camera
        if camera is None:
            return
        while not self._stop.is_set():
            try:
                frame = camera.grab_bgr(timeout_ms=2000)
            except Exception as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._error = str(exc)
                return
            if self._stop.is_set():
                return
            small = _downscale(frame, 800)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            y0, y1 = int(height * 0.3), int(height * 0.7)
            x0, x1 = int(width * 0.3), int(width * 0.7)
            with self._lock:
                self._serial += 1
                self._full = frame
                self._center = float(gray[y0:y1, x0:x1].mean())
                self._overall = float(gray.mean())
