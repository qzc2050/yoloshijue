# -*- coding: utf-8 -*-
"""Draw one scan_line frame with detections and expected Gerber centers."""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\xiangmu\shibie")
HOST = next(p for p in ROOT.iterdir() if (p / "yolo_tools" / "export_chamber_crops.py").is_file())
sys.path.insert(0, str(HOST / "yolo_tools"))
import export_chamber_crops as ec  # noqa: E402

SCAN = ROOT / "qc_motion" / "scan_line"
name = sys.argv[1] if len(sys.argv) > 1 else "line_15_x160_y135.jpg"
path = SCAN / name
mx = int(name.split("_x")[1].split("_")[0])
my = int(name.split("_y")[1].split(".")[0])
dets = [item for item in json.loads((SCAN / "detections.json").read_text(encoding="utf-8")) if item["file"] == name]
groups = json.loads((SCAN / "groups.json").read_text(encoding="utf-8"))
image = ec._read(path)
height, width = image.shape[:2]
scale = 1400 / max(height, width)
vis = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
px_per_mm = 44.2

def to_px(bx, by):
    py = (bx - mx) * px_per_mm + height / 2
    px = (by - my) * px_per_mm + width / 2
    return int(px * scale), int(py * scale)

for chamber in groups["chambers"]:
    px, py = to_px(chamber["x"], chamber["y"])
    if px < -20 or py < -20 or px > vis.shape[1] + 20 or py > vis.shape[0] + 20:
        continue
    color = {"horizontal": (80, 80, 255), "vertical": (255, 180, 40), "short": (80, 220, 80)}[chamber["kind"]]
    cv2.circle(vis, (px, py), 6, color, 1, cv2.LINE_AA)

for hit in dets:
    p0 = (int(hit["x"] * scale), int(hit["y"] * scale))
    p1 = (int((hit["x"] + hit["w"]) * scale), int((hit["y"] + hit["h"]) * scale))
    color = (255, 80, 255) if hit["vertical"] else (0, 255, 255)
    cv2.rectangle(vis, p0, p1, color, 1)
    cv2.putText(vis, f"{hit['score']:.2f}", (p0[0], max(12, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

scores = [hit["score"] for hit in dets if hit["vertical"]]
others = [hit["score"] for hit in dets if not hit["vertical"]]
print(name, "vert scores", sorted(scores), "horiz scores", sorted(others))
out = SCAN / "_preview.jpg"
ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
buf.tofile(str(out))
print("wrote", out, vis.shape)
