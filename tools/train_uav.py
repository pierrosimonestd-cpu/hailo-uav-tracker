#!/usr/bin/env python3
"""Fine-tune a YOLO detector for single-class UAV detection on DUT Anti-UAV.

Design choices worth knowing before you change them:

* **yolov8n at 640x640.** The Hailo Model Zoo publishes yolov8n for Hailo-8L at
  202 FPS (batch 1), which leaves the pan/tilt loop an order of magnitude more
  headroom than it needs. A larger backbone buys accuracy the servos cannot use.
* **640 input, not 416.** Over half of the DUT Anti-UAV objects are below 32x32
  pixels (see ``data/dut_antiuav/stats.json``). Dropping the input resolution is
  the fastest way to destroy recall on exactly the targets that matter.
* **Mosaic on, closed near the end.** Mosaic augmentation manufactures extra
  small-object context; leaving it on until the last epochs biases the model
  towards fragmented scenes, so it is disabled for the final ``--close-mosaic``
  epochs.
* **Fixed seed, deterministic ops.** The numbers in docs/benchmarks.md must be
  reproducible from this script alone.
* **``--cache ram`` when you have the memory.** Training here is disk-bound, not
  compute-bound: mosaic augmentation reads four images per sample, so a batch of
  16 is 64 JPEG decodes, and on a machine with real-time virus scanning that
  dominates everything else. Caching the decoded, resized images costs roughly
  0.7 MB per image (about 3.6 GB for this dataset at 640) and removes the
  bottleneck outright.

Usage:
    python tools/train_uav.py --data data/dut_antiuav/dut_antiuav.yaml --epochs 30
    python tools/train_uav.py --resume runs/train/uav_yolov8n/weights/last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/dut_antiuav/dut_antiuav.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="starting checkpoint or model yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', '0,1', ...")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache",
        default=None,
        choices=["ram", "disk"],
        help="cache decoded images; 'ram' needs about 0.7 MB per training image",
    )
    parser.add_argument("--close-mosaic", type=int, default=8)
    parser.add_argument("--patience", type=int, default=100, help="early-stopping patience")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="uav_yolov8n")
    parser.add_argument("--resume", type=Path, default=None, help="resume from a last.pt")
    args = parser.parse_args()

    from ultralytics import YOLO

    if args.resume is not None:
        model = YOLO(str(args.resume))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=args.cache or False,
            close_mosaic=args.close_mosaic,
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            # Resolved to an absolute path: Ultralytics otherwise interprets a
            # relative project under its own settings runs_dir, so the outputs
            # land somewhere other than where the caller asked for.
            project=str(args.project.resolve()),
            name=args.name,
            exist_ok=True,
            plots=True,
            val=True,
        )

    save_dir = Path(results.save_dir)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "cache": args.cache,
        "seed": args.seed,
        "weights": str(save_dir / "weights" / "best.pt"),
    }
    (save_dir / "train_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining finished.")
    print(f"  weights: {save_dir / 'weights' / 'best.pt'}")
    print("  next:    python tools/export_onnx.py --weights", save_dir / "weights" / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
