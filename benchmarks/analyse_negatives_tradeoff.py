#!/usr/bin/env python3
"""Compare two detectors at matched bird false-positive rates.

Training on bird photographs as background images cuts the false-positive rate
on birds from 23% to 7%. On its own that number proves nothing: raising the
confidence threshold cuts it too, for free, and costs drone recall. Any model
can be made to stop firing on birds by making it stop firing.

So the comparison here is not "which model has fewer false positives" but
"at the same false-positive rate on birds, which model finds more drones". That
is the question a deployment actually asks, and it is the only one where a
genuinely better model can be told apart from a more timid one.

Note what is being joined: the bird rate comes from Open Images, the drone
recall from the DUT Anti-UAV test split. Two datasets, deliberately -- the
operating question spans both. DUT Anti-UAV contains no birds, so it cannot
answer the first half, and Open Images contains no labelled UAVs, so it cannot
answer the second.

Usage:
    python benchmarks/analyse_negatives_tradeoff.py
"""

from __future__ import annotations

import argparse
import collections
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (label, detection tag, hard-negative tag) for the two models compared.
BASELINE = ("baseline", "yolov8n-fp32-pytorch", "yolov8n-fp32-pytorch")
WITH_NEGATIVES = ("with-negatives", "yolov8n-negatives", "yolov8n-negatives")

#: Bird false-positive rates to compare at. For each, the highest confidence
#: threshold whose rate is at or below the target is chosen per model, so
#: neither model is ever flattered by being scored at a laxer setting.
TARGET_FP_RATES = [0.10, 0.06, 0.035, 0.02]

IOU_MATCH = 0.5


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def load_truth(gt_path: Path) -> tuple[dict[int, list[list[float]]], int]:
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    truth: dict[int, list[list[float]]] = collections.defaultdict(list)
    for annotation in ground_truth["annotations"]:
        x, y, w, h = annotation["bbox"]
        truth[annotation["image_id"]].append([x, y, x + w, y + h])
    return truth, sum(len(v) for v in truth.values())


def drone_recall(
    detections_path: Path, threshold: float, truth: dict, total: int
) -> tuple[float, float]:
    """Greedy highest-score-first matching at IoU 0.5, as COCO does."""
    detections = json.loads(detections_path.read_text(encoding="utf-8"))

    per_image: dict[int, list[tuple[list[float], float]]] = collections.defaultdict(list)
    for detection in detections:
        if detection["score"] >= threshold:
            x, y, w, h = detection["bbox"]
            per_image[detection["image_id"]].append(([x, y, x + w, y + h], detection["score"]))

    true_positives = false_positives = 0
    for image_id, boxes in per_image.items():
        gts = truth.get(image_id, [])
        claimed = [False] * len(gts)
        for box, _ in sorted(boxes, key=lambda item: -item[1]):
            best_iou, best_index = 0.0, -1
            for index, gt_box in enumerate(gts):
                if claimed[index]:
                    continue
                value = iou(box, gt_box)
                if value > best_iou:
                    best_iou, best_index = value, index
            if best_iou >= IOU_MATCH:
                claimed[best_index] = True
                true_positives += 1
            else:
                false_positives += 1

    recall = true_positives / total if total else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    return recall, precision


def threshold_at_or_below(sweep: list[dict], target: float) -> tuple[float, float] | None:
    """Lowest threshold whose bird FP rate is at or below ``target``.

    Lowest, not nearest: a lower threshold is the harder setting for the model,
    so this never grants a model an easier operating point than the target asks
    for.
    """
    candidates = [row for row in sweep if row["false_positive_rate"] <= target]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: row["threshold"])
    return best["threshold"], best["false_positive_rate"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt", type=Path, default=Path("data/dut_antiuav/coco/test.json"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.gt.exists():
        sys.exit(f"{args.gt} not found; run tools/prepare_dataset.py first")

    models = {}
    for label, detection_tag, negative_tag in (BASELINE, WITH_NEGATIVES):
        detections = args.results / detection_tag / "detections.json"
        negatives = args.results / f"hard_negatives_{negative_tag}.json"
        for path in (detections, negatives):
            if not path.exists():
                sys.exit(
                    f"{path} not found.\nBoth models must have been evaluated first; see "
                    "docs/benchmarks.md section 4."
                )
        models[label] = (detections, json.loads(negatives.read_text(encoding="utf-8")))

    truth, total = load_truth(args.gt)
    print(f"{total} ground-truth drones across {len(truth)} images\n")

    header = f"{'target':>7}{'model':>17}{'thr':>6}{'bird FP':>9}{'recall':>9}{'precision':>11}"
    print(header)
    print("-" * len(header))

    rows = []
    for target in TARGET_FP_RATES:
        entry: dict = {"target_fp_rate": target, "models": {}}
        for label in models:
            detections_path, negatives = models[label]
            found = threshold_at_or_below(negatives["sweep"], target)
            if found is None:
                print(f"{target * 100:6.1f}% {label:>16}   (never reaches this rate)")
                continue
            threshold, actual = found
            recall, precision = drone_recall(detections_path, threshold, truth, total)
            print(
                f"{target * 100:6.1f}% {label:>16} {threshold:6.2f} "
                f"{actual * 100:8.1f}% {recall:9.4f} {precision:11.4f}"
            )
            entry["models"][label] = {
                "threshold": threshold,
                "bird_fp_rate": actual,
                "drone_recall": recall,
                "drone_precision": precision,
            }
        both = entry["models"]
        if len(both) == 2:
            base = both["baseline"]["drone_recall"]
            tuned = both["with-negatives"]["drone_recall"]
            entry["recall_delta"] = tuned - base
            entry["recall_relative"] = (tuned - base) / base if base else None
        rows.append(entry)
        print()

    payload = {
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Bird false-positive rates come from Open Images; drone recall from the DUT "
            "Anti-UAV test split. Each model is scored at the lowest confidence threshold "
            "reaching the target bird rate, so neither is compared at a laxer setting than "
            "the other."
        ),
        "iou_match": IOU_MATCH,
        "ground_truth_objects": total,
        "comparisons": rows,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }

    out = args.out or args.results / "negatives_tradeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
