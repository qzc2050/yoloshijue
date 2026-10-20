# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import cv2

from grab import grab_frame

out = Path(sys.argv[1])
image, settings = grab_frame()
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
print(f"saved {out.name} {image.shape[1]}x{image.shape[0]} exp {settings.exposure_us}")
