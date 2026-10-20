# -*- coding: utf-8 -*-
"""Hikrobot GigE camera grabber (MVS SDK)."""

from __future__ import annotations

import os
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _sdk_runtime_dir() -> str:
    bit32 = sys.maxsize <= 2**32
    root = r"C:\Program Files (x86)\Common Files\MVS\Runtime"
    return os.path.join(root, "Win32_i86" if bit32 else "Win64_x64")


def _sdk_python_dir() -> str:
    runenv = os.environ.get("MVCAM_COMMON_RUNENV") or r"C:\Program Files (x86)\MVS\Development"
    return os.path.join(runenv, "Samples", "Python", "MvImport")


def _prepare_sdk() -> None:
    runtime = _sdk_runtime_dir()
    if not os.path.isdir(runtime):
        raise RuntimeError(f"未找到海康 MVS 运行库：{runtime}")
    os.environ["PATH"] = runtime + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(runtime)
    py_dir = _sdk_python_dir()
    if py_dir not in sys.path:
        sys.path.insert(0, py_dir)


_prepare_sdk()

from MvCameraControl_class import (  # noqa: E402
    MV_ACCESS_Control,
    MV_ACCESS_Exclusive,
    MV_ACCESS_ExclusiveWithSwitch,
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_HB_DECODE_PARAM,
    MV_CC_PIXEL_CONVERT_PARAM_EX,
    MV_FRAME_OUT,
    MV_GENTL_CXP_DEVICE,
    MV_GENTL_CAMERALINK_DEVICE,
    MV_GENTL_GIGE_DEVICE,
    MV_GENTL_XOF_DEVICE,
    MV_GIGE_DEVICE,
    MV_TRIGGER_MODE_OFF,
    MV_USB_DEVICE,
    MvCamera,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_HB_BayerBG8,
    PixelType_Gvsp_HB_BayerGB8,
    PixelType_Gvsp_HB_BayerGR8,
    PixelType_Gvsp_HB_BayerRG8,
    PixelType_Gvsp_HB_Mono8,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
    PixelType_Gvsp_Mono14,
    PixelType_Gvsp_Mono16,
)


_SDK_READY = False


def _ensure_sdk() -> None:
    global _SDK_READY
    if not _SDK_READY:
        MvCamera.MV_CC_Initialize()
        _SDK_READY = True


def _decode_chars(array) -> str:
    raw = memoryview(array).tobytes()
    raw = raw.split(b"\x00", 1)[0]
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _ip_from_int(value: int) -> str:
    return ".".join(str((int(value) >> shift) & 0xFF) for shift in (24, 16, 8, 0))


def _prefer_bayer_color(cam) -> str:
    """10GC is a color sensor; Mono8 was used for metrology and looks gray."""
    for name in ("BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8"):
        if cam.MV_CC_SetEnumValueByString("PixelFormat", name) == 0:
            cam.MV_CC_SetEnumValueByString("BalanceWhiteAuto", "Once")
            return name
    return ""


def _is_hb(pixel_type: int) -> bool:
    return pixel_type in (
        PixelType_Gvsp_HB_Mono8,
        PixelType_Gvsp_HB_BayerGR8,
        PixelType_Gvsp_HB_BayerRG8,
        PixelType_Gvsp_HB_BayerGB8,
        PixelType_Gvsp_HB_BayerBG8,
    )


def _is_mono(pixel_type: int) -> bool:
    return pixel_type in (
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
        PixelType_Gvsp_Mono14,
        PixelType_Gvsp_Mono16,
        PixelType_Gvsp_HB_Mono8,
    )


class HikrobotCamera:
    def __init__(self) -> None:
        self.cam: Optional[MvCamera] = None
        self.info: Optional[str] = None
        self._grabbing = False

    def enumerate(self) -> List[Tuple[str, str, str]]:
        _ensure_sdk()
        devices = MV_CC_DEVICE_INFO_LIST()
        layer = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = MvCamera.MV_CC_EnumDevices(layer, devices)
        if ret != 0:
            raise RuntimeError(f"枚举相机失败 0x{ret:x}")
        found: List[Tuple[str, str, str]] = []
        for index in range(devices.nDeviceNum):
            info = cast(devices.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                gige = info.SpecialInfo.stGigEInfo
                found.append(
                    (
                        _ip_from_int(gige.nCurrentIp),
                        _decode_chars(gige.chModelName),
                        _decode_chars(gige.chSerialNumber),
                    )
                )
        return found

    def connect(self, ip: str) -> str:
        self.disconnect()
        _ensure_sdk()
        devices = MV_CC_DEVICE_INFO_LIST()
        layer = MV_GIGE_DEVICE | MV_GENTL_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(layer, devices)
        if ret != 0:
            raise RuntimeError(f"枚举相机失败 0x{ret:x}")
        target = None
        model = serial = ip
        for index in range(devices.nDeviceNum):
            info = cast(devices.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
            if info.nTLayerType not in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
                continue
            gige = info.SpecialInfo.stGigEInfo
            current_ip = _ip_from_int(gige.nCurrentIp)
            if current_ip == ip:
                target = info
                model = _decode_chars(gige.chModelName)
                serial = _decode_chars(gige.chSerialNumber)
                break
        if target is None:
            names = ", ".join(item[0] for item in self.enumerate()) or "无"
            raise RuntimeError(f"未找到 {ip}。当前可见：{names}")

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(target)
        if ret != 0:
            raise RuntimeError(f"创建句柄失败 0x{ret:x}")
        opened = False
        last_error = 0
        for mode in (MV_ACCESS_Exclusive, MV_ACCESS_ExclusiveWithSwitch, MV_ACCESS_Control):
            ret = cam.MV_CC_OpenDevice(mode, 0)
            if ret == 0:
                opened = True
                break
            last_error = ret
        if not opened:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"打开相机失败 0x{last_error:x}。请先在 MVS 里停止采集并关闭该设备。"
            )

        packet = cam.MV_CC_GetOptimalPacketSize()
        if int(packet) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet))
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        pixel = _prefer_bayer_color(cam)
        cam.MV_CC_SetEnumValueByString("ExposureAuto", "Off")
        cam.MV_CC_SetEnumValueByString("GainAuto", "Off")
        cam.MV_CC_SetFloatValue("Gain", 6.0)
        cam.MV_CC_SetFloatValue("ExposureTime", 40000.0)
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"开始取流失败 0x{ret:x}")

        self.cam = cam
        self._grabbing = True
        extra = f"  {pixel}" if pixel else ""
        self.info = f"{model}  {serial}  {ip}{extra}"
        return self.info

    def grab_bgr(self, timeout_ms: int = 3000) -> np.ndarray:
        if self.cam is None:
            raise RuntimeError("相机未连接。")
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = self.cam.MV_CC_GetImageBuffer(frame, timeout_ms)
        if ret != 0 or not frame.pBufAddr:
            raise RuntimeError(f"取图失败 0x{ret:x}（超时或丢包）。")
        try:
            return self._frame_to_bgr(frame)
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def _frame_to_bgr(self, frame: MV_FRAME_OUT) -> np.ndarray:
        width = int(frame.stFrameInfo.nWidth)
        height = int(frame.stFrameInfo.nHeight)
        src_type = int(frame.stFrameInfo.enPixelType)
        src_buf = frame.pBufAddr
        src_len = int(frame.stFrameInfo.nFrameLen)

        if _is_hb(src_type):
            decode_len = width * height * 3
            decode_buf = (c_ubyte * decode_len)()
            decode = MV_CC_HB_DECODE_PARAM()
            decode.pSrcBuf = src_buf
            decode.nSrcLen = src_len
            decode.pDstBuf = decode_buf
            decode.nDstBufSize = decode_len
            ret = self.cam.MV_CC_HBDecode(decode)
            if ret != 0:
                raise RuntimeError(f"HB 解码失败 0x{ret:x}")
            src_buf = decode.pDstBuf
            src_len = decode.nDstBufLen
            src_type = decode.enDstPixelType

        convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(convert), 0, sizeof(convert))
        convert.nWidth = width
        convert.nHeight = height
        convert.pSrcData = src_buf
        convert.nSrcDataLen = src_len
        convert.enSrcPixelType = src_type
        convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
        dst_len = width * height * 3
        dst_buf = (c_ubyte * dst_len)()
        convert.pDstBuffer = dst_buf
        convert.nDstBufferSize = dst_len
        ret = self.cam.MV_CC_ConvertPixelTypeEx(convert)
        if ret != 0:
            if _is_mono(src_type):
                gray = np.frombuffer(src_buf, dtype=np.uint8, count=width * height).reshape(height, width).copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            raise RuntimeError(f"像素转换失败 0x{ret:x}")
        return (
            np.frombuffer(dst_buf, dtype=np.uint8, count=dst_len)
            .reshape(height, width, 3)
            .copy()
        )

    def disconnect(self) -> None:
        cam = self.cam
        self.cam = None
        self.info = None
        grabbing = self._grabbing
        self._grabbing = False
        if cam is None:
            return
        if grabbing:
            cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
