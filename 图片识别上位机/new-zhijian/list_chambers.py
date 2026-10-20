# -*- coding: utf-8 -*-
"""Count ionization chambers (inner+outer ring) and list each ring width."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_three_ways import preview, save_image, to_gray  # noqa: E402
from ring_detector import _measure_dual_ring_widths, detect_rings  # noqa: E402

SRC = Path(__file__).resolve().parent / "captured" / "width_now" / "full.jpg"
OUT = Path(__file__).resolve().parent / "captured" / "width_now"


def nms(boxes: list[tuple], iou_thr: float = 0.35) -> list[int]:
    if not boxes:
        return []
    areas = np.array([b[2] * b[3] for b in boxes], dtype=np.float32)
    order = np.argsort(-areas)
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i][0], np.array([boxes[j][0] for j in rest]))
        yy1 = np.maximum(boxes[i][1], np.array([boxes[j][1] for j in rest]))
        xx2 = np.minimum(boxes[i][0] + boxes[i][2], np.array([boxes[j][0] + boxes[j][2] for j in rest]))
        yy2 = np.minimum(boxes[i][1] + boxes[i][3], np.array([boxes[j][1] + boxes[j][3] for j in rest]))
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] + 1e-6 - inter
        iou = inter / union
        order = rest[iou < iou_thr]
    return keep


def center_nms(rings: list, min_dist_frac: float = 0.55) -> list:
    """Drop duplicate boxes on the same stadium (packed neighbors stay)."""
    if not rings:
        return []
    minors = np.array([min(r.w, r.h) for r in rings], dtype=np.float32)
    min_dist = float(np.median(minors) * min_dist_frac)
    order = np.argsort([-r.w * r.h for r in rings])
    kept: list = []
    for idx in order:
        r = rings[int(idx)]
        cx, cy = r.x + r.w / 2.0, r.y + r.h / 2.0
        if any((cx - (k.x + k.w / 2.0)) ** 2 + (cy - (k.y + k.h / 2.0)) ** 2 < min_dist**2 for k in kept):
            continue
        kept.append(r)
    return kept


def main() -> None:
    img = cv2.imdecode(np.fromfile(str(SRC), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {SRC}")
    gray = to_gray(img)
    h, w = gray.shape

    det = detect_rings(img, source_dpi=1200, threshold_width_px=3, sr_scale=1, allow_small=True)
    raw = []
    for r in det.results:
        aspect = max(r.w, r.h) / max(1.0, min(r.w, r.h))
        if min(r.w, r.h) < 28 or max(r.w, r.h) < 55:
            continue
        if aspect < 1.55:
            continue
        if r.inner_line_width_px < 0.8 and r.outer_line_width_px < 0.8:
            continue
        raw.append(r)

    boxes = [(r.x, r.y, r.w, r.h) for r in raw]
    keep = nms(boxes, 0.35)
    chambers = center_nms([raw[i] for i in keep])
    chambers.sort(key=lambda r: (r.y // 50, r.x))

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # scale: elongated blob major ~ pad 9.2 mm
    thr = int(np.clip(np.percentile(gray, 88), 70, 230))
    _, bw = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    majors = []
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        ww, hh = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        if area < 500:
            continue
        aspect = max(ww, hh) / max(1.0, min(ww, hh))
        if 2.2 <= aspect <= 8:
            majors.append(max(ww, hh))
    um_px = 9200.0 / float(np.median(majors)) if majors else 28.2

    vis = img.copy()
    rows = []
    for i, r in enumerate(chambers, 1):
        inner_min, outer_min, inner_med, outer_med, *_rest = _measure_dual_ring_widths(
            gray, binary, r.x, r.y, r.w, r.h, sr_scale=1
        )
        inner_px = float(inner_med if inner_med >= 0.8 else r.inner_line_width_px)
        outer_px = float(outer_med if outer_med >= 0.8 else r.outer_line_width_px)
        inner_mm = inner_px * um_px / 1000.0
        outer_mm = outer_px * um_px / 1000.0
        rows.append(
            {
                "id": i,
                "x": r.x,
                "y": r.y,
                "w": r.w,
                "h": r.h,
                "inner_px": round(inner_px, 2),
                "outer_px": round(outer_px, 2),
                "inner_mm": round(inner_mm, 3),
                "outer_mm": round(outer_mm, 3),
                "inner_min_px": round(float(inner_min), 2),
                "outer_min_px": round(float(outer_min), 2),
            }
        )
        color = (0, 220, 255)
        cv2.rectangle(vis, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)
        cv2.putText(
            vis,
            f"{i}",
            (r.x, max(16, r.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    csv_path = OUT / "ionization_chambers.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    save_image(OUT / "chambers_annotated.jpg", vis)
    save_image(OUT / "chambers_preview.jpg", preview(vis, 1800))

    print(f"FOV_scale {um_px:.2f} um/px  (pad length 9.2mm assumption)")
    print(f"CHAMBER_COUNT {len(chambers)}")
    print(f"raw_detect {len(det.results)} filtered {len(raw)} after_nms {len(chambers)}")
    if chambers:
        minors = [min(r.w, r.h) for r in chambers]
        majors = [max(r.w, r.h) for r in chambers]
        print(
            f"box_minor median {np.median(minors):.0f}px  "
            f"box_major median {np.median(majors):.0f}px"
        )
    if rows:
        inn = np.array([r["inner_px"] for r in rows])
        out = np.array([r["outer_px"] for r in rows])
        print(
            f"inner_px median {np.median(inn):.2f}  outer_px median {np.median(out):.2f}"
        )
        print(
            f"inner_mm median {np.median(inn)*um_px/1000:.3f}  "
            f"outer_mm median {np.median(out)*um_px/1000:.3f}"
        )
    print("CSV", csv_path)
    for r in rows:
        print(
            f"{r['id']:3d}  inner {r['inner_px']:5.2f}px / {r['inner_mm']:.3f}mm  "
            f"outer {r['outer_px']:5.2f}px / {r['outer_mm']:.3f}mm  xy=({r['x']},{r['y']})"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        SRC = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        OUT = Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    main()
