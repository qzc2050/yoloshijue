# -*- coding: utf-8 -*-
import sys

import cv2
import numpy as np

before = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
after = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
before = cv2.resize(before, (480, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
after = cv2.resize(after, (480, 320), interpolation=cv2.INTER_AREA).astype(np.float32)
before = (before - before.mean()) / (before.std() + 1e-6)
after = (after - after.mean()) / (after.std() + 1e-6)
corr = float((before * after).mean())
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.82
print(f"corr {corr:.3f} stop-above {limit:.2f}", flush=True)
if corr > limit:
    raise SystemExit(2)
