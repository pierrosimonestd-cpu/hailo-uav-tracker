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
COLOUR_INSET = (200, 200, 200)

#: Cells are 16:9 because the benchmark's images are. Fitting a 1920x1080 frame
#: into a square cell letterboxes away 44% of it as black bar.
CELL_W, CELL_H = 480, 270

#: Side of the zoomed inset, and how much context it shows around the target.
INSET = 108
INSET_CONTEXT = 5.0

#: Targets already at least this many pixels across in the rendered cell get no
#: inset. Magnifying something the reader can already see adds a box and a
#: connector line to look at and no information.
INSET_NEEDED_BELOW_PX = 34.0

#: The crop may not exceed this fraction of the frame's shorter side. Without a
#: cap, a large target asks for a crop bigger than the frame, and the outline
#: marking it degenerates into a border around the whole cell.
INSET_MAX_FRAME_FRACTION = 0.55


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
    origin = (x1, max(14, y1 - 6))
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Dark outline first, so the label stays readable against bright sky.
    cv2.putText(frame, label, origin, font, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, origin, font, 0.5, colour, 1, cv2.LINE_AA)


def fit_cell(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a frame into a ``width x height`` cell, preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cell = np.full((height, width, 3), 18, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return cell


def region_of_interest(boxes: list[tuple[float, ...]], shape: tuple[int, ...]) -> tuple[int, ...]:
    """A square crop around ``boxes``, clamped to the frame.

    The point of the sheet is the small targets, and a 30-pixel drone in a
    1920-wide frame survives the downscale to a cell as about two pixels. The
    crop is what makes the sheet worth looking at.
    """
    h, w = shape[:2]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    side = max(x2 - x1, y2 - y1) * INSET_CONTEXT
    side = float(min(max(side, 96.0), min(w, h) * INSET_MAX_FRAME_FRACTION))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    left = int(round(min(max(cx - side / 2, 0), w - side)))
    top = int(round(min(max(cy - side / 2, 0), h - side)))
    return left, top, int(round(side))


def needs_magnifying(boxes: list[tuple[float, ...]], shape: tuple[int, ...]) -> bool:
    """Whether the target is too small to read once the frame is cell-sized."""
    frame_h, frame_w = shape[:2]
    scale = min(CELL_W / frame_w, CELL_H / frame_h)
    largest = max(max(box[2] - box[0], box[3] - box[1]) for box in boxes)
    return largest * scale < INSET_NEEDED_BELOW_PX


def add_inset(cell: np.ndarray, frame: np.ndarray, roi: tuple[int, ...]) -> None:
    """Paste a zoomed crop of ``frame`` into the corner of ``cell``.

    Also outlines where the crop came from, so the inset cannot be mistaken for
    a second image or for a detection the model made somewhere else.
    """
    left, top, side = roi
    crop = frame[top : top + side, left : left + side]
    if crop.size == 0:
        return
    zoom = cv2.resize(crop, (INSET, INSET), interpolation=cv2.INTER_NEAREST)

    cell_h, cell_w = cell.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    scale = min(cell_w / frame_w, cell_h / frame_h)
    offset_x = (cell_w - int(frame_w * scale)) // 2
    offset_y = (cell_h - int(frame_h * scale)) // 2
    cv2.rectangle(
        cell,
        (offset_x + int(left * scale), offset_y + int(top * scale)),
        (offset_x + int((left + side) * scale), offset_y + int((top + side) * scale)),
        COLOUR_INSET,
        1,
    )

    y, x = cell_h - INSET - 6, cell_w - INSET - 6
    cell[y : y + INSET, x : x + INSET] = zoom
    cv2.rectangle(cell, (x - 1, y - 1), (x + INSET, y + INSET), COLOUR_INSET, 1)


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
    parser.add_argument(
        "--no-zoom",
        action="store_true",
        help="omit the zoomed inset and show whole frames only",
    )
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

        boxes = []
        for x, y, w, h in truth.get(image_info["id"], []):
            boxes.append((x, y, x + w, y + h))
            draw(frame, (x, y, x + w, y + h), COLOUR_TRUTH, "truth")
        for det in detector.infer(frame):
            boxes.append((det.x1, det.y1, det.x2, det.y2))
            draw(frame, (det.x1, det.y1, det.x2, det.y2), COLOUR_PREDICTION, f"{det.score:.2f}")

        cell = fit_cell(frame, CELL_W, CELL_H)
        if boxes and not args.no_zoom and needs_magnifying(boxes, frame.shape):
            add_inset(cell, frame, region_of_interest(boxes, frame.shape))
        cells.append(cell)
    detector.close()

    if not cells:
        sys.exit("no images could be read")

    columns = min(args.columns, len(cells))
    rows = (len(cells) + columns - 1) // columns
    while len(cells) < rows * columns:
        cells.append(np.full((CELL_H, CELL_W, 3), 18, dtype=np.uint8))

    sheet = np.vstack([np.hstack(cells[r * columns : (r + 1) * columns]) for r in range(rows)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]}, {len(cells)} cells)")
    print("green = ground truth, orange = prediction")
    if not args.no_zoom:
        print("inset = the outlined region, magnified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
