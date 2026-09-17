#!/usr/bin/env python3
"""Statically quantise the exported ONNX model to INT8.

Why this exists in a project that deploys on Hailo: the Hailo-8L is an INT8
accelerator, so the model that actually runs on the turret is not the model that
was trained. Somewhere between FP32 and the HEF there is an accuracy cost, and a
project that reports only its FP32 numbers is reporting a model it does not run.

Compiling a HEF needs the Hailo Dataflow Compiler, which needs a developer-zone
login and does not run in CI. Static INT8 quantisation through ONNX Runtime is
the closest thing that *does* run anywhere: same per-tensor affine scheme, same
calibration-set dependence, same failure modes. It is a proxy for the Hailo
quantiser, not a substitute, and docs/benchmarks.md labels it as one.

Calibration images are drawn from the *training* split. Using test images to
calibrate would leak the test set into the model and inflate every number that
follows.

Usage:
    python tools/quantize_onnx.py --model models/uav_yolov8n_640.onnx \\
        --calib-dir data/dut_antiuav/train/img --calib-images 200
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from uavtrack.detect.preprocess import letterbox  # noqa: E402


class LetterboxCalibrationReader:
    """Feeds calibration images through the pipeline's own pre-processing.

    Calibrating on differently pre-processed images than the deployed model
    sees produces activation ranges that do not match reality, and the
    quantised model loses accuracy for reasons that look mysterious later. This
    reader deliberately reuses :func:`uavtrack.detect.preprocess.letterbox`.
    """

    def __init__(self, paths: list[Path], input_name: str, input_size: tuple[int, int]) -> None:
        self.paths = paths
        self.input_name = input_name
        self.input_size = input_size
        self._iterator = iter(paths)

    def get_next(self) -> dict | None:
        """Return the next calibration batch, or ``None`` when exhausted."""
        for path in self._iterator:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            padded, _ = letterbox(frame, self.input_size)
            blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            return {self.input_name: np.ascontiguousarray(blob)}
        return None

    def rewind(self) -> None:
        self._iterator = iter(self.paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=Path("models/uav_yolov8n_640.onnx"))
    parser.add_argument("--out", type=Path, default=None, help="default: <model>.int8.onnx")
    parser.add_argument("--calib-dir", type=Path, default=Path("data/dut_antiuav/train/img"))
    parser.add_argument("--calib-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help=(
            "per-channel weight quantisation; more accurate, and what the Hailo "
            "compiler does by default"
        ),
    )
    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"{args.model} not found; run tools/export_onnx.py first")

    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit("onnxruntime is required: pip install 'uavtrack[onnx]'")

    out = args.out or args.model.with_suffix(".int8.onnx")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    height = input_meta.shape[2] if isinstance(input_meta.shape[2], int) else 640
    width = input_meta.shape[3] if isinstance(input_meta.shape[3], int) else 640
    del session

    images = sorted(args.calib_dir.glob("*.jpg"))
    if not images:
        sys.exit(f"no calibration images in {args.calib_dir}")
    rng = random.Random(args.seed)
    rng.shuffle(images)
    images = images[: args.calib_images]
    print(f"calibrating on {len(images)} images from {args.calib_dir}")

    # Shape inference and constant folding first: without it, quantize_static
    # leaves large parts of the graph in FP32 because it cannot infer shapes.
    prepared = args.model.with_suffix(".prep.onnx")
    quant_pre_process(str(args.model), str(prepared), skip_symbolic_shape=False)

    reader = LetterboxCalibrationReader(images, input_name, (width, height))
    out.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(prepared),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=args.per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepared.unlink(missing_ok=True)

    fp32_mb = args.model.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"wrote {out}")
    print(f"  {fp32_mb:.2f} MB FP32 -> {int8_mb:.2f} MB INT8  ({fp32_mb / int8_mb:.2f}x smaller)")
    print("\nNext: python benchmarks/eval_detection.py --model", out, "--tag yolov8n-int8-onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
