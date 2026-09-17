#!/usr/bin/env python3
"""COCO-protocol detection accuracy on the DUT Anti-UAV test set.

Runs any backend -- PyTorch, ONNX Runtime, or a Hailo HEF on the Pi -- through
the identical pre/post-processing the live pipeline uses, and scores it with
``pycocotools``. Using the same code path for benchmarking and for deployment is
the point: a benchmark that runs its own special inference wrapper measures the
wrapper, not the system.

Reported metrics are the standard COCO set. The size breakdown matters more
here than in most detection work: over half of the objects in this dataset are
below 32x32 pixels, so ``AP_small`` is close to the whole story and an
aggregate AP can hide a model that has quietly stopped seeing distant targets.

Usage:
    python benchmarks/eval_detection.py \\
        --model runs/train/uav_yolov8n/weights/best.pt \\
        --gt data/dut_antiuav/coco/test.json \\
        --images data/dut_antiuav/test/img \\
        --tag yolov8n-fp32-pytorch
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from uavtrack.detect.factory import build_detector  # noqa: E402

# Scored at a low threshold: COCO AP integrates over the precision-recall curve,
# so suppressing low-confidence detections before scoring truncates the curve
# and *understates* AP. The operating threshold used in the live pipeline is a
# separate decision, made in configs/, not here.
EVAL_CONF_THRESHOLD = 0.001

METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def run_inference(
    detector, images_dir: Path, gt: dict, limit: int | None
) -> tuple[list[dict], list[float], list[int]]:
    """Run the detector over every image in the ground truth.

    Returns:
        ``(coco_detections, per_image_latencies_ms, evaluated_image_ids)``. The
        image ids are returned so the scorer can be restricted to exactly the
        images that were run -- otherwise ``--limit`` scores a subset of
        detections against the full ground truth and every metric collapses in
        proportion to the subset size.
    """
    detections: list[dict] = []
    latencies: list[float] = []
    evaluated_ids: list[int] = []

    images = gt["images"][:limit] if limit else gt["images"]
    total = len(images)

    for index, image_info in enumerate(images):
        path = images_dir / image_info["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[warn] could not read {path}, skipping")
            continue

        start = time.perf_counter()
        results = detector.infer(frame)
        latencies.append((time.perf_counter() - start) * 1000.0)
        evaluated_ids.append(image_info["id"])

        for det in results:
            detections.append(
                {
                    "image_id": image_info["id"],
                    # pycocotools categories are 1-based; the internal class
                    # index is 0-based. The same offset is applied when the
                    # ground truth is built, in uavtrack.data.voc.to_coco.
                    "category_id": det.class_id + 1,
                    "bbox": [
                        round(det.x1, 2),
                        round(det.y1, 2),
                        round(det.width, 2),
                        round(det.height, 2),
                    ],
                    "score": round(det.score, 5),
                }
            )

        if (index + 1) % 250 == 0 or index + 1 == total:
            print(f"  {index + 1}/{total} images", flush=True)

    return detections, latencies, evaluated_ids


def evaluate(
    gt_path: Path, detections: list[dict], work_dir: Path, image_ids: list[int]
) -> dict[str, float]:
    """Score detections with pycocotools and return the named metrics.

    Args:
        gt_path: COCO ground-truth JSON.
        detections: Detections in COCO results format.
        work_dir: Where the results JSON is written.
        image_ids: Exactly the images that were run; the evaluation is
            restricted to these.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit("pycocotools is required: pip install 'uavtrack[bench]'")

    if not detections:
        return dict.fromkeys(METRIC_NAMES, 0.0)

    results_path = work_dir / "detections.json"
    results_path.write_text(json.dumps(detections), encoding="utf-8")

    # pycocotools prints unconditionally; keep the benchmark output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(results_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {name: float(value) for name, value in zip(METRIC_NAMES, evaluator.stats, strict=True)}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help=".pt, .onnx or .hef")
    parser.add_argument("--backend", default=None, choices=["ultralytics", "onnx", "hailo"])
    parser.add_argument("--gt", type=Path, default=Path("data/dut_antiuav/coco/test.json"))
    parser.add_argument("--images", type=Path, default=Path("data/dut_antiuav/test/img"))
    parser.add_argument("--tag", default=None, help="label for the results file")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime intra-op threads")
    args = parser.parse_args()

    if not args.gt.exists():
        sys.exit(f"{args.gt} not found; run tools/prepare_dataset.py first")

    kwargs: dict = {"conf_threshold": EVAL_CONF_THRESHOLD}
    if args.threads is not None:
        kwargs["intra_op_threads"] = args.threads

    detector = build_detector(args.model, backend=args.backend, **kwargs)
    tag = args.tag or Path(args.model).stem
    print(f"model    {args.model}")
    print(f"backend  {detector.name}  input {detector.input_size}")

    detector.warmup()
    gt = json.loads(args.gt.read_text(encoding="utf-8"))

    args.out.mkdir(parents=True, exist_ok=True)
    work_dir = args.out / tag
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    detections, latencies, evaluated_ids = run_inference(detector, args.images, gt, args.limit)
    wall_s = time.perf_counter() - started
    detector.close()

    metrics = evaluate(args.gt, detections, work_dir, evaluated_ids)

    payload = {
        "tag": tag,
        "kind": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": str(args.model),
        "backend": detector.name,
        "input_size": list(detector.input_size),
        "dataset": str(args.gt),
        "images": len(latencies),
        "eval_conf_threshold": EVAL_CONF_THRESHOLD,
        "metrics": {k: round(v, 5) for k, v in metrics.items()},
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "median": round(percentile(latencies, 0.5), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "min": round(min(latencies), 3) if latencies else None,
        },
        "wall_seconds": round(wall_s, 1),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }

    path = args.out / f"detection_{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    for name in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
        print(f"  {name:<10} {metrics[name]:.4f}")
    print(
        f"  {'latency':<10} {payload['latency_ms']['median']:.1f} ms median, "
        f"{payload['latency_ms']['p95']:.1f} ms p95"
    )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
