#!/usr/bin/env python3
"""Export a trained YOLO checkpoint to ONNX for ONNX Runtime and for Hailo.

The export settings here are chosen for what happens *downstream*, and the
non-obvious ones are worth stating:

* ``opset=11``. The Hailo Dataflow Compiler parses opsets 11-13 reliably;
  newer opsets introduce operators its parser rejects or lowers inefficiently.
  ONNX Runtime is happy with 11, so one export serves both.
* ``nms=False``. Hailo attaches its own NMS node during compilation, running it
  on the NPU. Exporting with NMS baked in would either fail to compile or push
  NMS onto the Pi's CPU, which is the resource this project is trying to
  protect.
* ``dynamic=False``. A fixed 1x3xHxW input is what the compiler wants, and a
  static shape lets ONNX Runtime fold more of the graph at load time.
* ``simplify=True``. Removes the training-time scaffolding that otherwise
  becomes unsupported operators in the compiler.

Usage:
    python tools/export_onnx.py --weights runs/train/uav_yolov8n/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OPSET = 11


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("models/uav_yolov8n_640.onnx"))
    parser.add_argument("--opset", type=int, default=OPSET)
    args = parser.parse_args()

    if not args.weights.exists():
        sys.exit(f"{args.weights} not found; train a model first with tools/train_uav.py")

    try:
        from ultralytics import YOLO
    except ImportError:  # pragma: no cover - environment dependent
        sys.exit("ultralytics is required: pip install 'uavtrack[train]'")

    model = YOLO(str(args.weights))
    print(f"exporting {args.weights} at {args.imgsz}x{args.imgsz}, opset {args.opset} ...")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        nms=False,
        half=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(args.out))
    print(f"wrote {args.out}")

    try:
        import onnx

        graph = onnx.load(str(args.out))
        onnx.checker.check_model(graph)
        inputs = [
            (i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
            for i in graph.graph.input
        ]
        outputs = [
            (o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
            for o in graph.graph.output
        ]
        print(f"  inputs   {inputs}")
        print(f"  outputs  {outputs}")
    except ImportError:
        print("  (install onnx to validate the exported graph)")

    print("\nNext:")
    print("  ONNX Runtime baseline : python benchmarks/eval_detection.py --model", args.out)
    print("  INT8 study            : python tools/quantize_onnx.py --model", args.out)
    print("  Hailo-8L              : bash tools/compile_hailo.sh", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
