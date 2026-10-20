# -*- coding: utf-8 -*-
"""Keep one COM port open for small stage nudges. No console window."""

from __future__ import annotations

import ctypes
import re
import threading
import time
from ctypes import wintypes

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
INVALID_HANDLE = ctypes.c_void_p(-1).value
CLRDTR = 6
CLRRTS = 4
PURGE_RXCLEAR = 0x0008
PURGE_TXCLEAR = 0x0004
POS = re.compile(r"x=(-?\d+\.\d+)\s+y=(-?\d+\.\d+)")
DONE_POS = re.compile(r"(?m)^pos\s+x=(-?\d+\.\d+)\s+y=(-?\d+\.\d+)")

kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
    wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
]
kernel32.CreateFileW.restype = ctypes.c_void_p
kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.ReadFile.argtypes = [
    ctypes.c_void_p, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
]
kernel32.ReadFile.restype = wintypes.BOOL
kernel32.WriteFile.argtypes = [
    ctypes.c_void_p, wintypes.LPCVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID,
]
kernel32.WriteFile.restype = wintypes.BOOL
kernel32.GetCommState.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
kernel32.GetCommState.restype = wintypes.BOOL
kernel32.SetCommState.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
kernel32.SetCommState.restype = wintypes.BOOL
kernel32.SetCommTimeouts.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
kernel32.SetCommTimeouts.restype = wintypes.BOOL
kernel32.EscapeCommFunction.argtypes = [ctypes.c_void_p, wintypes.DWORD]
kernel32.EscapeCommFunction.restype = wintypes.BOOL
kernel32.PurgeComm.argtypes = [ctypes.c_void_p, wintypes.DWORD]
kernel32.PurgeComm.restype = wintypes.BOOL


class _DCB(ctypes.Structure):
    _fields_ = [
        ("DCBlength", wintypes.DWORD),
        ("BaudRate", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("wReserved", wintypes.WORD),
        ("XonLim", wintypes.WORD),
        ("XoffLim", wintypes.WORD),
        ("ByteSize", ctypes.c_byte),
        ("Parity", ctypes.c_byte),
        ("StopBits", ctypes.c_byte),
        ("XonChar", ctypes.c_char),
        ("XoffChar", ctypes.c_char),
        ("ErrorChar", ctypes.c_char),
        ("EofChar", ctypes.c_char),
        ("EvtChar", ctypes.c_char),
        ("wReserved1", wintypes.WORD),
    ]


class _Timeouts(ctypes.Structure):
    _fields_ = [
        ("ReadIntervalTimeout", wintypes.DWORD),
        ("ReadTotalTimeoutMultiplier", wintypes.DWORD),
        ("ReadTotalTimeoutConstant", wintypes.DWORD),
        ("WriteTotalTimeoutMultiplier", wintypes.DWORD),
        ("WriteTotalTimeoutConstant", wintypes.DWORD),
    ]


def list_ports() -> list[str]:
    """COM ports currently registered by Windows."""
    import winreg

    found: list[str] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
    except OSError:
        return []
    index = 0
    while True:
        try:
            _name, value, _kind = winreg.EnumValue(key, index)
        except OSError:
            break
        found.append(str(value))
        index += 1
    winreg.CloseKey(key)

    def sort_key(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(set(found), key=sort_key)


class MotionStopped(OSError):
    """The operator pressed the emergency stop while the stage was moving."""


class MotionLink:
    """COM7 at 115200. DTR and RTS stay off so opening the port does not reset the board."""

    def __init__(self, port: str = "COM7") -> None:
        self.port = port
        self._handle: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    def connect(self) -> str:
        with self._lock:
            if self._handle is not None:
                return self._where()
            handle = kernel32.CreateFileW(
                "\\\\.\\" + self.port, GENERIC_READ | GENERIC_WRITE, 0, None, OPEN_EXISTING, 0, None,
            )
            if not handle or handle == INVALID_HANDLE:
                code = ctypes.get_last_error()
                raise OSError(f"打不开 {self.port}（错误 {code}）。先关掉正在占用串口的程序。")
            self._handle = handle
            try:
                dcb = _DCB()
                dcb.DCBlength = ctypes.sizeof(_DCB)
                if not kernel32.GetCommState(handle, ctypes.byref(dcb)):
                    raise OSError("读串口参数失败")
                dcb.BaudRate = 115200
                dcb.ByteSize = 8
                dcb.Parity = 0
                dcb.StopBits = 0
                dcb.flags = 1
                if not kernel32.SetCommState(handle, ctypes.byref(dcb)):
                    raise OSError("设置串口失败")
                kernel32.EscapeCommFunction(handle, CLRDTR)
                kernel32.EscapeCommFunction(handle, CLRRTS)
                timeouts = _Timeouts(20, 0, 200, 0, 2000)
                kernel32.SetCommTimeouts(handle, ctypes.byref(timeouts))
                time.sleep(0.4)
                boot = self._read_for(0.3)
                if "qc_motion boot" in boot:
                    raise OSError("打开串口时单片机复位了，没有移动")
                kernel32.PurgeComm(handle, PURGE_RXCLEAR | PURGE_TXCLEAR)
                x_text, y_text = self._where()
                return f"x={x_text} y={y_text}"
            except Exception:
                self._close_unlocked()
                raise

    def set_port(self, port: str) -> None:
        port = port.strip().upper()
        if not port:
            raise OSError("先选择串口")
        if self._handle is not None:
            raise OSError("先关闭串口再换端口")
        self.port = port

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def transact(self, line: str) -> str:
        """Send one command line and return the board's reply text."""
        line = line.strip()
        if not line:
            raise OSError("命令是空的")
        try:
            line.encode("ascii")
        except UnicodeEncodeError:
            raise OSError("命令只能是英文和数字")
        low = line.lower()
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            self._stop_event.clear()
            kernel32.PurgeComm(self._handle, PURGE_RXCLEAR | PURGE_TXCLEAR)
            self._write(line + "\n")
            if low.startswith("m ") or low.startswith("m\t"):
                reply, stopped = self._read_until_done(90)
                if stopped:
                    found = DONE_POS.findall(reply)
                    where = f"，停在 x={found[-1][0]} y={found[-1][1]}" if found else ""
                    raise MotionStopped(f"已急停{where}")
                return reply.strip()
            if low[:2] in ("x+", "x-", "y+", "y-", "z+", "z-"):
                return self._read_until_text("jog  done", 8).strip()
            return self._read_reply(1.2).strip()

    def read_xy(self) -> tuple[float, float]:
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            x_text, y_text = self._where()
            return float(x_text), float(y_text)

    def move_abs(self, x: float, y: float, expect_x: float, expect_y: float, timeout: float = 50.0) -> tuple[float, float]:
        """Move only when the live reading still matches expect. Otherwise send nothing."""
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            x_text, y_text = self._where()
            got_x, got_y = float(x_text), float(y_text)
            if abs(got_x - expect_x) > 0.02 or abs(got_y - expect_y) > 0.02:
                raise OSError(
                    f"当前位置 x={got_x:.3f} y={got_y:.3f}，不是 x={expect_x:.3f} y={expect_y:.3f}，没有移动"
                )
            return self._move_unlocked(x, y, timeout)

    def go_to(self, x: float, y: float) -> tuple[float, float]:
        """Absolute move from wherever the stage is now."""
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            self._stop_event.clear()
            x_text, y_text = self._where()
            travel = abs(float(x_text) - x) + abs(float(y_text) - y)
            return self._move_unlocked(x, y, 10 + travel / 7.0 * 1.3)

    def nudge(self, axis: str, delta: float) -> tuple[float, float]:
        if axis not in ("x", "y") or abs(delta) < 0.001 or abs(delta) > 50:
            raise ValueError("每次只能走 0.05 到 50 mm")
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            self._stop_event.clear()
            x_text, y_text = self._where()
            x_value, y_value = float(x_text), float(y_text)
            if axis == "x":
                x_value = round(x_value + delta, 3)
            else:
                y_value = round(y_value + delta, 3)
            return self._move_unlocked(x_value, y_value, 8 + abs(delta) * 0.3)

    def zero(self) -> tuple[float, float]:
        """Call the current pose x=0 y=0. The stage does not move."""
        with self._lock:
            if self._handle is None:
                raise OSError("请先打开串口")
            kernel32.PurgeComm(self._handle, PURGE_RXCLEAR | PURGE_TXCLEAR)
            self._write("zero\n")
            self._read_for(0.4)
            x_text, y_text = self._where()
            got_x, got_y = float(x_text), float(y_text)
            if abs(got_x) > 0.0005 or abs(got_y) > 0.0005:
                raise OSError(f"设原点后读到 x={got_x:.3f} y={got_y:.3f}，不是 0")
            return got_x, got_y

    def request_stop(self) -> None:
        """Emergency stop. Safe to call from any thread, moving or not."""
        self._stop_event.set()
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._handle is not None:
                self._write("s\n")
                self._read_for(0.3)
        finally:
            self._stop_event.clear()
            self._lock.release()

    def _move_unlocked(self, x: float, y: float, timeout: float) -> tuple[float, float]:
        kernel32.PurgeComm(self._handle, PURGE_RXCLEAR | PURGE_TXCLEAR)
        self._write(f"m {x:.3f} {y:.3f}\n")
        reply, stopped_by_operator = self._read_until_done(timeout)
        found = DONE_POS.findall(reply)
        if stopped_by_operator:
            where = f"，停在 x={found[-1][0]} y={found[-1][1]}" if found else ""
            raise MotionStopped(f"已急停{where}")
        if "stopped" in reply or "err" in reply:
            raise OSError(f"没有走到 {x:.3f} {y:.3f}：{reply.strip()}")
        if not found:
            raise OSError(f"没有读到到位：{reply.strip()}")
        arrived_x, arrived_y = float(found[-1][0]), float(found[-1][1])
        if abs(arrived_x - x) > 0.02 or abs(arrived_y - y) > 0.02:
            raise OSError(f"没有走到 x={x:.3f} y={y:.3f}，读到 x={arrived_x:.3f} y={arrived_y:.3f}")
        return arrived_x, arrived_y

    def _close_unlocked(self) -> None:
        if self._handle is not None:
            kernel32.CloseHandle(self._handle)
            self._handle = None

    def _write(self, text: str) -> None:
        data = text.encode("ascii")
        written = wintypes.DWORD()
        ok = kernel32.WriteFile(self._handle, data, len(data), ctypes.byref(written), None)
        if not ok or written.value != len(data):
            raise OSError("串口写入失败")

    def _read_until_text(self, marker: str, seconds: float) -> str:
        deadline = time.perf_counter() + seconds
        text = ""
        while time.perf_counter() < deadline:
            text += self._read_for(0.12)
            if marker in text or "err" in text or "stopped" in text:
                time.sleep(0.15)
                text += self._read_for(0.1)
                break
        return text

    def _read_reply(self, seconds: float) -> str:
        deadline = time.perf_counter() + seconds
        text = ""
        saw_line = False
        while time.perf_counter() < deadline:
            chunk = self._read_for(0.1)
            if chunk:
                text += chunk
                if "\n" in text:
                    saw_line = True
            elif saw_line:
                break
        return text

    def _read_for(self, seconds: float) -> str:
        deadline = time.perf_counter() + seconds
        chunks: list[str] = []
        buf = ctypes.create_string_buffer(256)
        nread = wintypes.DWORD()
        while time.perf_counter() < deadline:
            ok = kernel32.ReadFile(self._handle, buf, 256, ctypes.byref(nread), None)
            if ok and nread.value:
                chunks.append(buf.raw[:nread.value].decode("ascii", errors="replace"))
            else:
                time.sleep(0.02)
        return "".join(chunks)

    def _read_until_done(self, seconds: float) -> tuple[str, bool]:
        """Wait for the position printed after the move, not the target printed before it starts."""
        deadline = time.perf_counter() + seconds
        text = ""
        stopped_by_operator = False
        while time.perf_counter() < deadline:
            if self._stop_event.is_set() and not stopped_by_operator:
                self._write("s\n")
                stopped_by_operator = True
                self._stop_event.clear()
                deadline = time.perf_counter() + 3.0
            text += self._read_for(0.15)
            if "stopped" in text or "err" in text or DONE_POS.search(text):
                time.sleep(0.2)
                text += self._read_for(0.15)
                break
        return text, stopped_by_operator

    def _read_until_pos(self, seconds: float) -> str:
        deadline = time.perf_counter() + seconds
        text = ""
        while time.perf_counter() < deadline:
            text += self._read_for(0.15)
            if POS.search(text):
                time.sleep(0.2)
                text += self._read_for(0.15)
                break
        return text

    def _where(self) -> tuple[str, str]:
        kernel32.PurgeComm(self._handle, PURGE_RXCLEAR | PURGE_TXCLEAR)
        self._write("pos\n")
        reply = self._read_until_pos(5)
        found = POS.findall(reply)
        if not found:
            raise OSError(f"没有读到位置：{reply.strip()}")
        return found[-1]
