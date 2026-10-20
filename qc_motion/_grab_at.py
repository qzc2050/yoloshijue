# -*- coding: utf-8 -*-
import sys
from pathlib import Path

root = Path(r"E:\xiangmu\shibie")
host = next(p for p in root.iterdir() if (p / "hikrobot_camera.py").is_file())
sys.path.insert(0, str(host))
sys.path.insert(0, str(host / "qc_host" / "vision"))

import cv2
import hikrobot_camera as hc
from grab import _read_settings

original = hc.MvCamera.MV_CC_SetFloatValue


def keep_until_we_set(self, key, value):
    if key in ("Gain", "ExposureTime"):
        return 0
    return original(self, key, value)


hc.MvCamera.MV_CC_SetFloatValue = keep_until_we_set
camera = hc.HikrobotCamera()
try:
    devices = camera.enumerate()
    if not devices:
        raise SystemExit("no camera")
    print(camera.connect(devices[0][0]), flush=True)
    original(camera.cam, "Gain", 0.0)
    original(camera.cam, "ExposureTime", 40000.0)
    image = None
    for _ in range(4):
        image = camera.grab_bgr()
    settings = _read_settings(camera.cam)
finally:
    camera.disconnect()
    hc.MvCamera.MV_CC_SetFloatValue = original

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
print(
    f"saved {out.name} {image.shape[1]}x{image.shape[0]} exp {settings.exposure_us} gain {settings.gain_db} mean {float(image.mean()):.1f}",
    flush=True,
)
