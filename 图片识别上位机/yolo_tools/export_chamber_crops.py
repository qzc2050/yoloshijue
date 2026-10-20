# -*- coding: utf-8 -*-
"""Cut every chamber out of full-resolution front-light frames for YOLO classification.

Chambers are found by matching one known-good chamber (chamber_template.png) in both
orientations, then turned long-side horizontal and saved at camera resolution. The host's rule
check (qc_host.vision.frontlight) runs on the same frame; each crop goes into the folder of the
rule verdict so labelling only means moving the wrong ones. Chambers the rule code did not find
go to "weidingwei". An overview per frame shows what was found.

    python yolo_tools/export_chamber_crops.py samples/grabs samples/board2.png
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for folder in (str(ROOT), str(HERE)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from qc_host.vision.frontlight import _analyse_chamber, _locate_chambers, _px  # noqa: E402
import project_config  # noqa: E402

DEFAULT_OUT = project_config.machine().yolo_root / "datasets" / "chamber_raw"
TEMPLATE = Path(__file__).with_name("chamber_template.png")
FOLDERS = {"合格": "hege", "渗墨": "shenmo", "缺墨": "quemo", "不能识别": "buneng", None: "weidingwei"}
COLORS = {"hege": (80, 210, 80), "shenmo": (40, 40, 220), "quemo": (0, 210, 255), "buneng": (0, 165, 255),
          "weidingwei": (255, 160, 0)}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# Frames smaller than this are previews or zooms, not camera frames.
MIN_SIDE = 3000
# Coarse search runs on a quarter-size frame; the hit is then re-matched at full size.
COARSE = 0.25
REFINE_REACH = 12
# Matching runs on locally normalised edge strength: the gap reads dark under one lamp setting and
# olive under another, but the ring outlines stay put. Blur is in coarse pixels.
EDGE_NORM_SIGMA = 8.0


@dataclass
class Hit:
    score: float
    vertical: bool
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


def _read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def _edges(gray: np.ndarray, sigma: float) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return mag / (cv2.GaussianBlur(mag, (0, 0), sigma) + 0.5 * float(mag.mean()) + 1e-6)


def find_chambers(image: np.ndarray, template: np.ndarray, threshold: float) -> list[Hit]:
    """Every full chamber in the frame, as boxes the size of the template."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = _edges(cv2.resize(gray, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA), EDGE_NORM_SIGMA)
    small_tpl = _edges(cv2.resize(template, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA), EDGE_NORM_SIGMA)
    full_tpl = _edges(template, EDGE_NORM_SIGMA / COARSE)
    height, width = gray.shape
    hits: list[Hit] = []
    for vertical in (False, True):
        tpl = cv2.rotate(full_tpl, cv2.ROTATE_90_CLOCKWISE) if vertical else full_tpl
        stpl = cv2.rotate(small_tpl, cv2.ROTATE_90_CLOCKWISE) if vertical else small_tpl
        th, tw = tpl.shape
        response = cv2.matchTemplate(small, stpl, cv2.TM_CCOEFF_NORMED)
        k = (min(stpl.shape) // 2) | 1
        peaks = (response == cv2.dilate(response, np.ones((k, k), np.uint8))) & (response >= threshold)
        for sy, sx in zip(*np.nonzero(peaks)):
            x0 = int(round(sx / COARSE)) - REFINE_REACH
            y0 = int(round(sy / COARSE)) - REFINE_REACH
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(width, x0 + tw + 2 * REFINE_REACH), min(height, y0 + th + 2 * REFINE_REACH)
            window = gray[y0c:y1c, x0c:x1c]
            if window.shape[0] < th or window.shape[1] < tw:
                continue
            fine = cv2.matchTemplate(_edges(window, EDGE_NORM_SIGMA / COARSE), tpl, cv2.TM_CCOEFF_NORMED)
            _lo, _hi, _lo_at, (fx, fy) = cv2.minMaxLoc(fine)
            hits.append(Hit(float(response[sy, sx]), vertical, x0c + fx, y0c + fy, tw, th))
    hits.sort(key=lambda h: -h.score)
    kept: list[Hit] = []
    for hit in hits:
        cx, cy = hit.center
        if all(abs(cx - k.center[0]) > 0.6 * min(k.w, k.h) or abs(cy - k.center[1]) > 0.6 * min(k.w, k.h) for k in kept):
            kept.append(hit)
    row = min(template.shape)
    kept.sort(key=lambda h: (round(h.center[1] / row), h.center[0]))
    return kept


def rule_verdicts(image: np.ndarray, hits: list[Hit]) -> list[str | None]:
    """Rule verdict for each hit, None where the rule code did not frame that chamber."""
    px = _px(image)
    found = []
    for i, loc in enumerate(_locate_chambers(image, px), start=1):
        m = cv2.moments(loc.outer)
        if m["m00"] > 0:
            found.append((m["m10"] / m["m00"], m["m01"] / m["m00"], _analyse_chamber(image, loc, i, px).verdict))
    out: list[str | None] = []
    for hit in hits:
        cx, cy = hit.center
        reach = 0.4 * min(hit.w, hit.h)
        near = [v for fx, fy, v in found if abs(fx - cx) < reach and abs(fy - cy) < reach]
        out.append(near[0] if near else None)
    return out


def _overview(image: np.ndarray, hits: list[Hit], folders: list[str]) -> np.ndarray:
    scale = 1600.0 / max(image.shape[:2])
    vis = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, (hit, folder) in enumerate(zip(hits, folders), start=1):
        p0 = (int(hit.x * scale), int(hit.y * scale))
        p1 = (int((hit.x + hit.w) * scale), int((hit.y + hit.h) * scale))
        cv2.rectangle(vis, p0, p1, COLORS[folder], 2)
        cv2.putText(vis, str(i), (p0[0] + 2, p0[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    y = 20
    for folder, color in COLORS.items():
        cv2.putText(vis, folder, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 22
    return vis


def collect_images(items: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT))
        elif p.suffix.lower() in IMAGE_EXT:
            files.append(p)
    return files


def load_template(path: Path = TEMPLATE) -> np.ndarray:
    template = _read(path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"template not found: {path}")
    return template


def crop_chamber(image: np.ndarray, hit: Hit) -> np.ndarray:
    crop = image[hit.y:hit.y + hit.h, hit.x:hit.x + hit.w]
    return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE) if hit.vertical else crop


def export_files(files: list[Path], out: Path, template: np.ndarray, threshold: float = 0.5,
                 use_rule: bool = True, log=print) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"
    new_index = not index_path.exists()
    counts = {name: 0 for name in COLORS}
    with index_path.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        if new_index:
            writer.writerow(["crop", "source", "chamber", "rule_verdict", "score", "vertical", "x", "y", "w", "h"])
        for path in files:
            image = _read(path)
            if image is None or max(image.shape[:2]) < MIN_SIDE:
                log(f"skip  {path}  (not a full-resolution frame)")
                continue
            started = time.perf_counter()
            hits = find_chambers(image, template, threshold)
            verdicts = rule_verdicts(image, hits) if use_rule else [None] * len(hits)
            folders = [FOLDERS.get(v, "buneng") for v in verdicts]
            tag = f"{path.parent.name}_{path.stem}"
            for i, (hit, verdict, folder) in enumerate(zip(hits, verdicts, folders), start=1):
                name = f"{tag}_c{i:03d}.png"
                _write(out / folder / name, crop_chamber(image, hit))
                counts[folder] += 1
                writer.writerow([f"{folder}/{name}", str(path.resolve()), i, verdict or "", f"{hit.score:.2f}",
                                 int(hit.vertical), hit.x, hit.y, hit.w, hit.h])
                fh.flush()
            _write(out / "_overview" / f"{tag}.jpg", _overview(image, hits, folders))
            log(f"done  {path}  chambers {len(hits)}  {time.perf_counter() - started:.1f}s")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files or folders (searched recursively)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--threshold", type=float, default=0.5, help="match score to accept a chamber")
    parser.add_argument("--no-rule", action="store_true", help="skip the rule check, put every crop in weidingwei")
    args = parser.parse_args()

    files = collect_images(args.inputs)
    if not files:
        sys.exit("no images found")
    counts = export_files(files, args.out, load_template(args.template), args.threshold, not args.no_rule)
    print(f"\nsaved to {args.out}")
    print("  " + "  ".join(f"{k}: {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
