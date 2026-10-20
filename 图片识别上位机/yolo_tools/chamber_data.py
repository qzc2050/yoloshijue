# -*- coding: utf-8 -*-
"""Labels, dataset layout and model bookkeeping shared by the YOLO lab UI and the trainer."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import project_config

YOLO_ROOT = project_config.machine().yolo_root
RAW = YOLO_ROOT / "datasets" / "chamber_raw"
CLS = YOLO_ROOT / "datasets" / "chamber_cls"
RUNS = YOLO_ROOT / "runs"
WEIGHTS = YOLO_ROOT / "weights"
LATEST = RUNS / "latest_chamber_model.json"
LABELS = RAW / "labels.csv"
INDEX = RAW / "index.csv"

CLASSES = {"hege": "合格", "shenmo": "渗墨", "quemo": "缺墨", "kanbuqing": "看不清"}
LABEL_NAMES = dict(CLASSES)
RULE_NAMES = {"hege": "合格", "shenmo": "渗墨", "quemo": "缺墨", "buneng": "不能识别", "weidingwei": "规则没找到"}
MIN_PER_CLASS = 5
VAL_FRACTION = 0.2


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, flags) if data.size else None


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))


def square(image: np.ndarray) -> np.ndarray:
    """Pad to a square so the classifier's centre crop keeps both ends of the chamber."""
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(image, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def read_index(raw: Path = RAW) -> dict[str, dict[str, str]]:
    """Crop path (relative to raw, forward slashes) -> its index row; later exports win."""
    rows: dict[str, dict[str, str]] = {}
    path = raw / "index.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                rows[row["crop"].replace("\\", "/")] = row
    return {k: v for k, v in rows.items() if (raw / k).exists()}


def read_labels(path: Path = LABELS) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                labels[row["crop"]] = row["label"]
    return labels


def write_labels(labels: dict[str, str], path: Path = LABELS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop", "label"])
        for crop, label in sorted(labels.items()):
            writer.writerow([crop, label])
    tmp.replace(path)


def label_counts(labels: dict[str, str]) -> dict[str, int]:
    counts = {k: 0 for k in LABEL_NAMES}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return counts


def build_dataset(raw: Path = RAW, out: Path = CLS, seed: int = 0) -> dict[str, dict[str, int]]:
    """Write train/val folders of squared crops.

    Crops from one source frame all land on the same side, so the validation score is measured on
    frames the model never saw.
    """
    index = read_index(raw)
    labels = {k: v for k, v in read_labels(raw / "labels.csv").items() if v in CLASSES and (raw / k).exists()}
    by_class: dict[str, list[str]] = defaultdict(list)
    for crop, label in labels.items():
        by_class[label].append(crop)
    usable = [c for c in CLASSES if len(by_class.get(c, [])) >= MIN_PER_CLASS]
    if len(usable) < 2:
        raise ValueError(
            f"样本不够：至少要有两类、每类至少 {MIN_PER_CLASS} 张。\n现在：" + "，".join(
                f"{CLASSES[c]} {len(by_class.get(c, []))} 张" for c in CLASSES)
        )

    rng = random.Random(seed)
    if out.exists():
        shutil.rmtree(out)
    summary = {"train": {}, "val": {}, "skipped": {}}
    for cls in CLASSES:
        crops = by_class.get(cls, [])
        if cls not in usable:
            summary["skipped"][cls] = len(crops)
            continue
        by_source: dict[str, list[str]] = defaultdict(list)
        for crop in crops:
            by_source[index.get(crop, {}).get("source", crop)].append(crop)
        sources = list(by_source)
        rng.shuffle(sources)
        # Small sources first, so one frame with many chambers does not swallow the whole val share.
        sources.sort(key=lambda s: len(by_source[s]))
        target = max(1, int(round(VAL_FRACTION * len(crops))))
        val: list[str] = []
        if len(sources) >= 2:
            for src in sources:
                if len(val) >= target:
                    break
                if len(val) + len(by_source[src]) >= len(crops):
                    continue
                val.extend(by_source[src])
        if not val:
            shuffled = crops[:]
            rng.shuffle(shuffled)
            val = shuffled[:target]
        val_set = set(val)
        for crop in crops:
            split = "val" if crop in val_set else "train"
            image = read_image(raw / crop)
            if image is None:
                continue
            write_image(out / split / cls / Path(crop).name, square(image))
            summary[split][cls] = summary[split].get(cls, 0) + 1
    return summary


def read_latest() -> dict | None:
    if LATEST.exists():
        try:
            return json.loads(LATEST.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def write_latest(info: dict) -> None:
    LATEST.parent.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
