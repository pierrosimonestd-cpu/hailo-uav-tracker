#!/usr/bin/env python3
"""Run every benchmark that has its inputs, then regenerate the tables.

The individual scripts are the interface; this is a driver so the full sweep is
one command and the order cannot be got wrong. Benchmarks whose inputs are
missing are skipped with a reason rather than failing the run -- most people
will not have a Hailo device, or the tracking subset, or a trained checkpoint,
and the ones they *can* run should still produce a report.

Usage:
    python benchmarks/run_all.py --weights runs/train/uav_yolov8n/weights/best.pt
    python benchmarks/run_all.py --weights ... --skip tracking hard-negatives
    python benchmarks/run_all.py --hef models/uav_yolov8n_640.hef   # on the Pi
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Step:
    """One benchmark invocation, with the inputs it needs to exist first."""

    def __init__(self, name: str, argv: list[str], requires: list[Path]) -> None:
        self.name = name
        self.argv = argv
        self.requires = requires

    def missing(self) -> list[Path]:
        return [p for p in self.requires if not p.exists()]


def run(step: Step, dry_run: bool) -> tuple[str, str]:
    """Execute one step. Returns ``(status, detail)``."""
    missing = step.missing()
    if missing:
        return "skip", f"missing {', '.join(str(p) for p in missing)}"

    printable = " ".join(step.argv)
    print(f"\n{'=' * 78}\n[{step.name}] {printable}\n{'=' * 78}", flush=True)
    if dry_run:
        return "dry", "not executed"

    started = time.perf_counter()
    result = subprocess.run([sys.executable, *step.argv], cwd=ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        return "fail", f"exit {result.returncode} after {elapsed:.0f}s"
    return "ok", f"{elapsed:.0f}s"


def build_steps(args) -> list[Step]:
    weights = Path(args.weights)
    onnx = Path(args.onnx)
    int8 = onnx.with_suffix(".int8.onnx")
    coco_gt = Path("data/dut_antiuav/coco/test.json")
    images = Path("data/dut_antiuav/test/img")
    train_images = Path("data/dut_antiuav/train/img")

    steps = [
        Step(
            "export-onnx",
            ["tools/export_onnx.py", "--weights", str(weights), "--out", str(onnx)],
            [weights],
        ),
        Step(
            "quantize-int8",
            ["tools/quantize_onnx.py", "--model", str(onnx), "--calib-dir", str(train_images)],
            [onnx, train_images],
        ),
        Step(
            "detection-fp32-pytorch",
            [
                "benchmarks/eval_detection.py",
                "--model", str(weights),
                "--gt", str(coco_gt),
                "--images", str(images),
                "--tag", "yolov8n-fp32-pytorch",
            ],
            [weights, coco_gt, images],
        ),
        Step(
            "detection-fp32-onnx",
            [
                "benchmarks/eval_detection.py",
                "--model", str(onnx),
                "--gt", str(coco_gt),
                "--images", str(images),
                "--tag", "yolov8n-fp32-onnx",
            ],
            [onnx, coco_gt, images],
        ),
        Step(
            "detection-int8-onnx",
            [
                "benchmarks/eval_detection.py",
                "--model", str(int8),
                "--gt", str(coco_gt),
                "--images", str(images),
                "--tag", "yolov8n-int8-onnx",
            ],
            [int8, coco_gt, images],
        ),
        Step(
            "latency-onnx",
            [
                "benchmarks/bench_latency.py",
                "--model", str(onnx),
                "--frames", str(args.latency_frames),
                "--tag", "yolov8n-fp32-onnx",
            ],
            [onnx],
        ),
        Step(
            "tracking",
            [
                "benchmarks/eval_tracking.py",
                "--model", str(weights),
                "--root", str(args.tracking_root),
                "--tag", "yolov8n-fp32-pytorch",
            ],
            [weights, Path(args.tracking_root)],
        ),
        Step(
            "hard-negatives",
            [
                "benchmarks/eval_hard_negatives.py",
                "--model", str(weights),
                "--fetch",
                "--tag", "yolov8n-fp32-pytorch",
            ],
            [weights],
        ),
        Step("control-simulation", ["benchmarks/bench_control.py", "--study", "all"], []),
    ]

    if args.hef:
        hef = Path(args.hef)
        steps.extend(
            [
                Step(
                    "detection-hailo",
                    [
                        "benchmarks/eval_detection.py",
                        "--model", str(hef),
                        "--gt", str(coco_gt),
                        "--images", str(images),
                        "--tag", "yolov8n-int8-hailo8l",
                    ],
                    [hef, coco_gt, images],
                ),
                Step(
                    "latency-hailo",
                    [
                        "benchmarks/bench_latency.py",
                        "--model", str(hef),
                        "--frames", str(args.latency_frames),
                        "--tag", "yolov8n-int8-hailo8l",
                    ],
                    [hef],
                ),
            ]
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", default="runs/train/uav_yolov8n/weights/best.pt")
    parser.add_argument("--onnx", default="models/uav_yolov8n_640.onnx")
    parser.add_argument("--hef", default=None, help="compiled HEF, when running on the Pi")
    parser.add_argument("--tracking-root", default="data/dut_antiuav_tracking")
    parser.add_argument("--latency-frames", type=int, default=300)
    parser.add_argument("--skip", nargs="*", default=[], help="step names to skip")
    parser.add_argument("--only", nargs="*", default=None, help="run only these step names")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    args = parser.parse_args()

    steps = build_steps(args)
    if args.only:
        steps = [s for s in steps if s.name in args.only]
    steps = [s for s in steps if s.name not in args.skip]

    results: list[tuple[str, str, str]] = []
    for step in steps:
        status, detail = run(step, args.dry_run)
        results.append((step.name, status, detail))
        if status == "skip":
            print(f"[{step.name}] skipped: {detail}", flush=True)

    if not args.dry_run:
        print(f"\n{'=' * 78}\n[report] regenerating the documentation tables\n{'=' * 78}", flush=True)
        subprocess.run([sys.executable, "benchmarks/make_report.py"], cwd=ROOT)

    print(f"\n{'summary':<26}{'status':<10}{'detail'}")
    print("-" * 78)
    for name, status, detail in results:
        print(f"{name:<26}{status:<10}{detail}")

    failed = [name for name, status, _ in results if status == "fail"]
    if failed:
        print(f"\n{len(failed)} step(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
