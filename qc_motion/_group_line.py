# -*- coding: utf-8 -*-
"""Find chambers in scan_line and group every view of the same physical chamber."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HOST = next(p for p in ROOT.iterdir() if (p / "yolo_tools" / "export_chamber_crops.py").is_file())
sys.path.insert(0, str(HOST / "yolo_tools"))

import export_chamber_crops as ec  # noqa: E402
import project_config as cfg  # noqa: E402

CJ_MM = 25.4 / 100000.0
SCAN = ROOT / "qc_motion" / "scan_line"
POSE = re.compile(r"_x(-?\d+(?:\.\d+)?)_y(-?\d+(?:\.\d+)?)$")
FLASH = re.compile(r"(?:X(-?\d+))?(?:Y(-?\d+))?D03\*")
APERTURE = re.compile(r"G54D(\d+)\*")


def read_board(folder: Path) -> cfg.Board:
    path = Path(folder) / "board.txt"
    if path.is_file():
        return cfg.board(path.read_text(encoding="utf-8").strip().lower())
    return cfg.board()


def read_side(folder: Path, board: cfg.Board) -> str:
    path = Path(folder) / "side.txt"
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip().lower()
        if text in board.sides:
            return text
    return next(iter(board.sides))


def _cj_flashes(path: Path) -> dict[str, list[tuple[float, float]]]:
    current = None
    x = y = None
    flashes: dict[str, list[tuple[float, float]]] = {"D49": [], "D50": []}
    for raw in path.read_text(encoding="latin1").splitlines():
        line = raw.strip()
        if line.startswith("G36"):
            continue
        aperture = APERTURE.search(line)
        if aperture:
            current = "D" + aperture.group(1)
        if current not in flashes:
            continue
        for match in FLASH.finditer(line):
            if match.group(1) is not None:
                x = int(match.group(1)) * CJ_MM
            if match.group(2) is not None:
                y = int(match.group(2)) * CJ_MM
            if x is None or y is None:
                continue
            flashes[current].append((x, y))
    return flashes


def _dedupe(points: list[tuple[float, float]], tol: float = 0.3) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for point in sorted(points):
        if kept and abs(point[0] - kept[-1][0]) <= tol and abs(point[1] - kept[-1][1]) <= tol:
            continue
        kept.append(point)
    return kept


def _to_machine(x: float, y: float, affine) -> tuple[float, float]:
    ax, bx, cx = affine[0]
    ay, by, cy = affine[1]
    return ax * x + bx * y + cx, ay * x + by * y + cy


def gerber_chambers(side: str, board: cfg.Board) -> list[dict]:
    spec = board.side(side)
    if not spec.gerber.is_file():
        raise RuntimeError(f"缺少光绘文件 {spec.gerber}")
    flashes = _cj_flashes(spec.gerber)
    horizontal = _dedupe(flashes["D49"])
    vertical_ap = _dedupe(flashes["D50"])
    if not vertical_ap:
        raise RuntimeError(f"{spec.gerber.name} 没有竖放电离室")
    bar_y = float(np.median([point[1] for point in vertical_ap]))
    vertical = [point for point in vertical_ap if abs(point[1] - bar_y) <= 2.0]
    short = [point for point in vertical_ap if abs(point[1] - bar_y) > 2.0]
    items = (
        [("horizontal", point) for point in horizontal]
        + [("vertical", point) for point in vertical]
        + [("short", point) for point in short]
    )
    chambers = []
    for kind, (gx, gy) in items:
        mx, my = _to_machine(gx, gy, board.gerber_to_machine)
        chambers.append({"kind": kind, "gx": round(gx, 3), "gy": round(gy, 3), "x": mx, "y": my})
    if spec.mirror_y:
        verticals = [chamber["y"] for chamber in chambers if chamber["kind"] == "vertical"]
        if not verticals:
            raise RuntimeError("底面没有竖放电离室，无法按这一列镜像")
        axis = float(np.median(verticals))
        for chamber in chambers:
            chamber["y"] = 2 * axis - chamber["y"]
    chambers.sort(key=lambda item: (round(item["x"] / 3.0), item["y"]))
    for number, chamber in enumerate(chambers, start=1):
        chamber["id"] = number
    return chambers


def board_mm(mx: int, my: int, width: int, height: int, px: float, py: float) -> tuple[float, float]:
    px_per_mm = cfg.machine().px_per_mm
    return mx + (py - height / 2) / px_per_mm, my + (px - width / 2) / px_per_mm


def detect(folder: Path, template: np.ndarray) -> list[dict]:
    folder = Path(folder)
    cache = folder / "detections.json"
    if cache.is_file():
        print(f"load {cache.name}", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))
    files = sorted(p for p in folder.glob("line_*.jpg") if POSE.search(p.stem))
    detections = []
    for index, path in enumerate(files, start=1):
        mx, my = (int(float(v)) for v in POSE.search(path.stem).groups())
        image = ec._read(path)
        height, width = image.shape[:2]
        hits = ec.find_chambers(image, template, 0.5)
        print(f"{index}/{len(files)} {path.name} hits {len(hits)}", flush=True)
        for hit in hits:
            cx, cy = hit.center
            bx, by = board_mm(mx, my, width, height, cx, cy)
            detections.append({
                "file": path.name,
                "mx": mx,
                "my": my,
                "x": int(hit.x),
                "y": int(hit.y),
                "w": int(hit.w),
                "h": int(hit.h),
                "vertical": bool(hit.vertical),
                "score": round(float(hit.score), 3),
                "bx": bx,
                "by": by,
            })
    cache.write_text(json.dumps(detections), encoding="utf-8")
    return detections


def cluster(detections: list[dict], merge_mm: float) -> list[dict]:
    detections = sorted(detections, key=lambda item: -item["score"])
    clusters: list[dict] = []
    for det in detections:
        chosen = None
        nearest = merge_mm
        for group in clusters:
            distance = float(np.hypot(det["bx"] - group["x"], det["by"] - group["y"]))
            if distance < nearest:
                nearest = distance
                chosen = group
        if chosen is None:
            clusters.append({"x": det["bx"], "y": det["by"], "views": [det]})
            continue
        chosen["views"].append(det)
        chosen["x"] = sum(view["bx"] for view in chosen["views"]) / len(chosen["views"])
        chosen["y"] = sum(view["by"] for view in chosen["views"]) / len(chosen["views"])
    return clusters


def _column_centers(horizontals: list[dict], rotation_deg: float) -> list[float]:
    theta = math.tan(math.radians(rotation_deg))
    bins: dict[int, list[float]] = {}
    for chamber in horizontals:
        straight = chamber["y"] - chamber["x"] * theta
        key = round(straight / 25) * 25
        bins.setdefault(key, []).append(straight)
    return [float(np.median(bins[key])) for key in sorted(bins)]


def _column_index(x: float, y: float, centers: list[float], rotation_deg: float) -> int:
    theta = math.tan(math.radians(rotation_deg))
    straight = y - x * theta
    return min(range(len(centers)), key=lambda index: abs(straight - centers[index]))


def _split_columns(horizontals: list[dict], clusters: list[dict], centers: list[float], rotation_deg: float):
    gerber_cols: dict[int, list[dict]] = {index: [] for index in range(len(centers))}
    found_cols: dict[int, list[dict]] = {index: [] for index in range(len(centers))}
    for chamber in horizontals:
        gerber_cols[_column_index(chamber["x"], chamber["y"], centers, rotation_deg)].append(chamber)
    for group in clusters:
        votes = sum(1 for view in group["views"] if view["vertical"])
        if votes * 2 >= len(group["views"]):
            continue
        found_cols[_column_index(group["x"], group["y"], centers, rotation_deg)].append(group)
    return gerber_cols, found_cols


def _equal_pairs(gerber_cols: dict[int, list[dict]], found_cols: dict[int, list[dict]]):
    src: list[list[float]] = []
    dst: list[list[float]] = []
    for index, column in gerber_cols.items():
        found = found_cols[index]
        if len(found) != len(column) or not column:
            print(f"skip column {index}: gerber {len(column)} clusters {len(found)}", flush=True)
            continue
        paired = zip(sorted(column, key=lambda item: item["x"]), sorted(found, key=lambda item: item["x"]))
        for chamber, group in paired:
            src.append([group["x"], group["y"]])
            dst.append([chamber["x"], chamber["y"]])
    return src, dst


def _align(chambers: list[dict], clusters: list[dict], board: cfg.Board) -> np.ndarray:
    """Shift by the columns whose counts already agree, then fit rotation. One column cannot see the angle."""
    horizontals = [chamber for chamber in chambers if chamber["kind"] == "horizontal"]
    rotation = board.column_rotation_deg
    centers = _column_centers(horizontals, rotation)
    gerber_cols, found_cols = _split_columns(horizontals, clusters, centers, rotation)
    src, dst = _equal_pairs(gerber_cols, found_cols)
    if len(src) < 3:
        raise RuntimeError("照片和光绘对不上，没有写出编号")
    src_arr = np.array(src, np.float32)
    dst_arr = np.array(dst, np.float32)
    shift = np.median(dst_arr - src_arr, axis=0)
    print(f"coarse shift dx {shift[0]:.2f} dy {shift[1]:.2f}", flush=True)
    for group in clusters:
        group["x"] += float(shift[0])
        group["y"] += float(shift[1])
    gerber_cols, found_cols = _split_columns(horizontals, clusters, centers, rotation)
    src, dst = _equal_pairs(gerber_cols, found_cols)
    matrix, _inliers = cv2.estimateAffinePartial2D(
        np.array(src, np.float32), np.array(dst, np.float32), method=cv2.RANSAC, ransacReprojThreshold=3,
    )
    if matrix is None:
        raise RuntimeError("照片和光绘对不上，没有写出编号")
    return matrix


def _mapped(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = cv2.transform(np.array([[[x, y]]], np.float32), matrix)[0, 0]
    return float(point[0]), float(point[1])


def _match_bar(chambers: list[dict], clusters: list[dict]) -> dict[int, dict]:
    """Walk the cross bar in order. Nearest-neighbour slips by one when the error nears the 4 mm pitch."""
    verticals = sorted((c for c in chambers if c["kind"] == "vertical"), key=lambda c: c["y"])
    ys = np.array([c["y"] for c in verticals])
    xs = np.array([c["x"] for c in verticals])
    bar = []
    for group in clusters:
        line_x = float(np.interp(group["my_gerber"], ys, xs))
        if abs(group["mx_gerber"] - line_x) < 4:
            bar.append(group)
    bar.sort(key=lambda group: group["my_gerber"])
    count_v, count_b = len(verticals), len(bar)
    skip_cost = 4.0
    huge = 1e6
    cost = [[huge] * (count_b + 1) for _ in range(count_v + 1)]
    step = [[None] * (count_b + 1) for _ in range(count_v + 1)]
    cost[0][0] = 0.0
    for i in range(count_v + 1):
        for j in range(count_b + 1):
            if cost[i][j] >= huge:
                continue
            if j < count_b and cost[i][j] + skip_cost < cost[i][j + 1]:
                cost[i][j + 1] = cost[i][j] + skip_cost
                step[i][j + 1] = (i, j, "skip")
            if i < count_v and j < count_b:
                dist = abs(verticals[i]["y"] - bar[j]["my_gerber"])
                pair = dist if dist <= 6 else huge
                if cost[i][j] + pair < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1] = cost[i][j] + pair
                    step[i + 1][j + 1] = (i, j, "pair")
    if cost[count_v][count_b] >= huge:
        raise RuntimeError(f"横梁没有对上，找到 {count_b} 组，光绘 {count_v} 颗")
    assigned: dict[int, dict] = {}
    i, j = count_v, count_b
    while step[i][j] is not None:
        pi, pj, kind = step[i][j]
        if kind == "pair":
            dist = abs(verticals[pi]["y"] - bar[pj]["my_gerber"])
            assigned[verticals[pi]["id"]] = {"distance": round(dist, 3), "group": bar[pj]}
        i, j = pi, pj
    print(f"bar {len(assigned)}/{count_v} from {count_b} clusters", flush=True)
    return assigned


def match(chambers: list[dict], clusters: list[dict], board: cfg.Board) -> tuple[np.ndarray, dict, list[dict]]:
    matrix = _align(chambers, clusters, board)
    for group in clusters:
        gx, gy = _mapped(matrix, group["x"], group["y"])
        group["mx_gerber"] = gx
        group["my_gerber"] = gy
    assigned = _match_bar(chambers, clusters)
    used_groups = {id(item["group"]) for item in assigned.values()}
    pairs = []
    for group in clusters:
        if id(group) in used_groups:
            continue
        for chamber in chambers:
            if chamber["kind"] == "vertical" or chamber["id"] in assigned:
                continue
            distance = math.hypot(group["mx_gerber"] - chamber["x"], group["my_gerber"] - chamber["y"])
            if distance <= board.match_mm:
                pairs.append((distance, chamber["id"], group))
    pairs.sort(key=lambda item: item[0])
    used_chambers = set(assigned)
    for distance, chamber_id, group in pairs:
        key = id(group)
        if chamber_id in used_chambers or key in used_groups:
            continue
        used_chambers.add(chamber_id)
        used_groups.add(key)
        assigned[chamber_id] = {"distance": round(distance, 3), "group": group}
    unmatched = [group for group in clusters if id(group) not in used_groups]
    return matrix, assigned, unmatched


def view_out(view: dict) -> dict:
    return {key: view[key] for key in ("file", "mx", "my", "x", "y", "w", "h", "vertical", "score")}


def group_scan(folder: Path | None = None, side: str | None = None, board_id: str | None = None) -> dict:
    """Match one scan folder to the board's Gerber ids. The motor offset is measured from the photos."""
    scan = Path(folder) if folder is not None else SCAN
    board = cfg.board(board_id) if board_id else (read_board(scan) if folder is not None else cfg.board())
    side = side or read_side(scan, board)
    spec = board.side(side)
    chambers = gerber_chambers(side, board)
    counts = {kind: sum(1 for chamber in chambers if chamber["kind"] == kind) for kind in ("horizontal", "vertical", "short")}
    print(f"gerber {board.id} {side} {len(chambers)} {counts}", flush=True)
    if len(chambers) != spec.expect:
        raise RuntimeError(f"{board.id} {'背面' if side == 'back' else '正面'}应有 {spec.expect} 颗，读到 {len(chambers)}")
    detections = detect(scan, ec.load_template())
    groups = cluster(detections, board.merge_mm)
    matrix, assigned, unmatched = match(chambers, groups, board)
    rows = []
    missing = []
    for chamber in chambers:
        hit = assigned.get(chamber["id"])
        views = sorted(hit["group"]["views"], key=lambda view: -view["score"]) if hit else []
        row = {
            "id": chamber["id"],
            "kind": chamber["kind"],
            "x": round(chamber["x"], 2),
            "y": round(chamber["y"], 2),
            "views": [view_out(view) for view in views],
        }
        if hit:
            row["error_mm"] = hit["distance"]
        else:
            missing.append({"id": chamber["id"], "kind": chamber["kind"], "x": row["x"], "y": row["y"]})
        rows.append(row)
    extras = []
    for group in unmatched:
        extras.append({
            "x": round(group["mx_gerber"], 2),
            "y": round(group["my_gerber"], 2),
            "n": len(group["views"]),
            "best": max(view["score"] for view in group["views"]),
            "files": sorted({view["file"] for view in group["views"]}),
        })
    errors = [row["error_mm"] for row in rows if "error_mm" in row]
    try:
        source = scan.relative_to(ROOT).as_posix()
    except ValueError:
        source = str(scan)
    payload = {
        "source": source,
        "photos": len(list(scan.glob("line_*.jpg"))),
        "detections": len(detections),
        "clusters": len(groups),
        "board": board.id,
        "side": side,
        "gerber": spec.expect,
        "matched": len(assigned),
        "missing": len(missing),
        "extras": len(extras),
        "align": {
            "dx_mm": round(float(matrix[0, 2]), 3),
            "dy_mm": round(float(matrix[1, 2]), 3),
            "rotation_deg": round(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))), 3),
        },
        "median_error_mm": round(float(np.median(errors)), 3) if errors else None,
        "p90_error_mm": round(float(np.percentile(errors, 90)), 3) if errors else None,
        "kind_counts": counts,
        "chambers": rows,
        "missing_list": missing,
        "extra_list": extras,
    }
    out = scan / "groups.json"
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row["kind"], [0, 0])
        by_kind[row["kind"]][1] += 1
        if row["views"]:
            by_kind[row["kind"]][0] += 1
    multi = sum(1 for row in rows if len(row["views"]) > 1)
    print(
        f"detections {len(detections)} clusters {len(groups)} matched {len(assigned)}/{spec.expect} "
        f"missing {len(missing)} extras {len(extras)} multi {multi} "
        f"align {payload['align']} median {payload['median_error_mm']} p90 {payload['p90_error_mm']}",
        flush=True,
    )
    print("by kind found/total", by_kind, flush=True)
    print("wrote", out, flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", help="scan folder; defaults to qc_motion/scan_line")
    parser.add_argument("--board", help="board id; defaults to the folder's board.txt")
    parser.add_argument("--side", choices=("front", "back"), help="board side; defaults to side.txt")
    args = parser.parse_args()
    group_scan(Path(args.folder) if args.folder else None, args.side, args.board)


if __name__ == "__main__":
    main()
