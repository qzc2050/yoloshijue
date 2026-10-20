# -*- coding: utf-8 -*-
"""Train the chamber classifier on chamber_cls, export it for OpenVINO and check it on the val split.

Lines starting with @@ are progress messages for the YOLO lab UI; everything else is the
ultralytics log.

    python yolo_tools/train_classifier.py --model n --epochs 30 --imgsz 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chamber_data as cd  # noqa: E402


def say(tag: str, payload) -> None:
    print(f"@@{tag} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def check_val(model, data: Path, imgsz: int) -> dict:
    """Per-image results on the val split, so misses can be shown by name."""
    rows = []
    for cls_dir in sorted((data / "val").iterdir()):
        for img in sorted(cls_dir.glob("*.png")):
            r = model.predict(str(img), imgsz=imgsz, verbose=False)[0]
            pred = r.names[int(r.probs.top1)]
            rows.append({"file": img.name, "truth": cls_dir.name, "pred": pred, "conf": round(float(r.probs.top1conf), 3)})
    right = sum(r["truth"] == r["pred"] for r in rows)
    missed = [r for r in rows if r["truth"] != "hege" and r["pred"] == "hege"]
    false_alarm = [r for r in rows if r["truth"] == "hege" and r["pred"] != "hege"]
    wrong = [r for r in rows if r["truth"] != r["pred"]]
    return {"total": len(rows), "right": right, "missed": len(missed), "false_alarm": len(false_alarm), "wrong": wrong}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=cd.CLS)
    parser.add_argument("--model", choices=["n", "s"], default="n")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    cd.WEIGHTS.mkdir(parents=True, exist_ok=True)
    if args.weights:
        say("STAGE", f"从上次的模型接着训练：{args.weights.name}")
        model = YOLO(str(args.weights))
    else:
        say("STAGE", "加载预训练模型（第一次会联网下载，约 5–20 MB）")
        model = YOLO(str(cd.WEIGHTS / f"yolo11{args.model}-cls.pt"))

    def on_epoch(trainer) -> None:
        metrics = trainer.metrics or {}
        loss = trainer.tloss
        parts = loss.values() if isinstance(loss, dict) else [loss] if loss is not None else []
        say("EPOCH", {
            "epoch": trainer.epoch + 1,
            "epochs": trainer.epochs,
            "loss": round(sum(float(p.sum() if hasattr(p, "sum") else p) for p in parts), 4),
            "acc": round(float(metrics.get("metrics/accuracy_top1", 0.0)), 4),
        })

    model.add_callback("on_fit_epoch_end", on_epoch)
    say("STAGE", "开始训练")
    started = time.time()
    # Random crops and random erasing could cut the defect out while the label still says defect.
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device="cpu", workers=2,
        project=str(cd.RUNS), name="chamber_cls", exist_ok=False, plots=True,
        scale=0.0, erasing=0.0, auto_augment=None, fliplr=0.5, flipud=0.5, patience=max(10, args.epochs // 3),
    )
    best = Path(model.trainer.best)
    say("STAGE", f"训练完成，用时 {(time.time() - started) / 60:.1f} 分钟；导出 OpenVINO 格式")

    trained = YOLO(str(best))
    openvino_dir = Path(trained.export(format="openvino", imgsz=args.imgsz, verbose=False))
    say("STAGE", "用验证集逐张检查")
    val = check_val(trained, args.data, args.imgsz)

    info = {
        "pt": str(best),
        "openvino": str(openvino_dir),
        "imgsz": args.imgsz,
        "model": f"yolo11{args.model}-cls",
        "epochs": args.epochs,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "names": trained.names,
        "val": val,
    }
    cd.write_latest(info)
    say("DONE", info)


if __name__ == "__main__":
    main()
