#!/usr/bin/env python3
"""Render a contact sheet of predictions against ground truth.

Aggregate metrics tell you how well a detector does; they do not tell you *what*
it gets wrong. A grid of real outputs does, in about ten seconds of looking:
whether the misses are the tiny distant targets or the ones against cluttered
rooftops, whether the boxes are loose, whether the false positives are all one
kind of thing.

Green is ground truth, orange is prediction. Images are picked to span the
object-size range rather than taken in file order, so the sheet is not entirely
made of whichever size happens to come first.

Usage:
    python tools/visualise_predictions.py \\
        --model runs/train/uav_yolov8n/weights/best.pt \\
        --out assets/predictions.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uavtrack.detect.factory import build_detector  # noqa: E402

COLOUR_TRUTH = (90, 200, 90)
COLOUR_PREDICTION = (40, 140, 240)
CELL = 480


def select_images(gt: dict, count: int) -> list[dict]:
    """Pick images spanning the object-size range, smallest first."""
    areas: dict[int, float] = {}
    for annotation in gt["annotations"]:
        area = annotation["bbox"][2] * annotation["bbox"][3]
        areas[annotation["image_id"]] = max(areas.get(annotation["image_id"], 0.0), area)

    with_objects = [image for image in gt["images"] if image["id"] in areas]
    with_objects.sort(key=lambda image: areas[image["id"]])
    if len(with_objects) <= count:
        return with_objects

    # Even spread across the sorted-by-size list.
    step = len(with_objects) / count
    return [with_objects[int(i * step)] for i in range(count)]


def draw(frame: np.ndarray, box: tuple[float, ...], colour, label: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def fit_cell(frame: np.ndarray, size: int) -> np.ndarray:
    """Letterbox a frame into a square cell, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cell = np.full((size, size, 3), 18, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return cell


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default=None, choices=["ultralytics", "onnx", "hailo"])
    parser.add_argument("--gt", type=Path, default=Path("data/dut_antiuav/coco/test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/dut_antiuav/test/img"))
    parser.add_argument("--out", type=Path, default=Path("assets/predictions.jpg"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    if not args.gt.exists():
        sys.exit(f"{args.gt} not found; run tools/prepare_dataset.py first")

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    truth: dict[int, list] = {}
    for annotation in gt["annotations"]:
        truth.setdefault(annotation["image_id"], []).append(annotation["bbox"])

    detector = build_detector(args.model, backend=args.backend, conf_threshold=args.conf)
    detector.warmup()

    cells = []
    for image_info in select_images(gt, args.count):
        frame = cv2.imread(str(args.images / image_info["file_name"]))
        if frame is None:
            continue

        for x, y, w, h in truth.get(image_info["id"], []):
            draw(frame, (x, y, x + w, y + h), COLOUR_TRUTH, "truth")
        for det in detector.infer(frame):
            draw(frame, (det.x1, det.y1, det.x2, det.y2), COLOUR_PREDICTION, f"{det.score:.2f}")

        cells.append(fit_cell(frame, CELL))
    detector.close()

    if not cells:
        sys.exit("no images could be read")

    columns = min(args.columns, len(cells))
    rows = (len(cells) + columns - 1) // columns
    while len(cells) < rows * columns:
        cells.append(np.full((CELL, CELL, 3), 18, dtype=np.uint8))

    sheet = np.vstack(
        [np.hstack(cells[r * columns : (r + 1) * columns]) for r in range(rows)]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(cells)} cells)")
    print("green = ground truth, orange = prediction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
