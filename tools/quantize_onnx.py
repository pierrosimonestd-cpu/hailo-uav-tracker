#!/usr/bin/env python3
"""Statically quantise the exported ONNX model to INT8.

Why this exists in a project that deploys on Hailo: the Hailo-8L is an INT8
accelerator, so the model that actually runs on the turret is not the model that
was trained. Somewhere between FP32 and the HEF there is an accuracy cost, and a
project that reports only its FP32 numbers is reporting a model it does not run.

Compiling a HEF needs the Hailo Dataflow Compiler, which needs a developer-zone
login and does not run in CI. Static INT8 quantisation through ONNX Runtime is
the closest thing that *does* run anywhere: same affine scheme, same
calibration-set dependence, same failure modes. It is a proxy for the Hailo
quantiser, not a substitute, and docs/benchmarks.md labels it as one.

Calibration images are drawn from the *training* split. Using test images to
calibrate would leak the test set into the model and inflate every number that
follows.

Two things about quantising a YOLOv8 head, both learned the hard way:

**Quantising the whole graph destroys the model.** Not "loses a point of mAP" --
it produces zero detections at any threshold. The final ``Concat`` in the head
joins decoded box coordinates, which span 0 to 640 in pixel units, with class
scores, which span 0 to 1. One quantisation scale has to cover both, and at
uint8 that scale is about 2.5 units per level, so every class score rounds to
zero. Excluding the head's *decode tail* -- the cheap element-wise ops after the
feature convolutions -- from quantisation fixes it. All the real compute is in
the convolutions, which stay INT8, so the saving is essentially unchanged.
Hailo's compiler does its own mixed-precision analysis for the same reason.

**Per-channel weights need opset 13 or newer.** Opset 11's ``QuantizeLinear``
has no ``axis`` attribute, so a per-channel model is an invalid graph and fails
at load with ``INVALID_GRAPH`` rather than anything that names the real cause.
``tools/export_onnx.py`` exports opset 13, which the Hailo Dataflow Compiler
also accepts.

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

#: Name prefix of the detection head in an Ultralytics YOLOv8 export.
DEFAULT_HEAD_PREFIX = "/model.22/"

#: Feature-convolution sub-blocks inside the head. Everything else under the
#: head prefix is decode arithmetic and is excluded from quantisation.
FEATURE_BLOCKS = ("cv2.", "cv3.")


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


def find_decode_tail(model_path: Path, head_prefix: str = DEFAULT_HEAD_PREFIX) -> list[str]:
    """Names of the head nodes that must stay in floating point.

    The head contains two kinds of node: the ``cv2.*`` / ``cv3.*`` convolution
    blocks that produce box-distribution and class features, and the decode
    arithmetic that turns them into boxes -- reshapes, the DFL softmax and
    convolution, the anchor arithmetic, and the final concatenation. The second
    group is what cannot share a quantisation scale.

    Args:
        model_path: ONNX graph to inspect.
        head_prefix: Node-name prefix of the detection head.

    Returns:
        Node names to pass as ``nodes_to_exclude``. Empty if the prefix matches
        nothing, which means the export is not laid out as expected.
    """
    import onnx

    graph = onnx.load(str(model_path)).graph
    excluded = []
    for node in graph.node:
        if not node.name.startswith(head_prefix):
            continue
        suffix = node.name[len(head_prefix) :]
        if suffix.startswith(FEATURE_BLOCKS):
            continue  # real compute: keep it quantised
        excluded.append(node.name)
    return excluded


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
        "--per-tensor",
        action="store_true",
        help="per-tensor weight quantisation instead of per-channel; smaller and less accurate",
    )
    parser.add_argument(
        "--quantize-head",
        action="store_true",
        help=(
            "also quantise the head's decode tail. Documented because it is the "
            "obvious thing to do and it produces a model that detects nothing; "
            "see this module's docstring"
        ),
    )
    parser.add_argument("--head-prefix", default=DEFAULT_HEAD_PREFIX)
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

    excluded: list[str] = []
    if not args.quantize_head:
        excluded = find_decode_tail(prepared, args.head_prefix)
        if excluded:
            print(f"keeping {len(excluded)} decode nodes in floating point")
        else:
            print(
                f"warning: no nodes matched {args.head_prefix!r}. If this export is not a "
                "stock Ultralytics YOLOv8 graph, pass --head-prefix, or the quantised "
                "model may produce no detections at all."
            )

    reader = LetterboxCalibrationReader(images, input_name, (width, height))
    out.parent.mkdir(parents=True, exist_ok=True)

    quantize_static(
        model_input=str(prepared),
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=not args.per_tensor,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=excluded,
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
