"""ONNX Runtime detection backend.

This is the reference implementation of the decode path. It runs anywhere,
which makes it the backend the unit tests and the accuracy benchmarks use, and
it is the honest CPU baseline the Hailo numbers are compared against.
"""

from __future__ import annotations

import numpy as np

from uavtrack.detect.base import Detection, DetectorBackend
from uavtrack.detect.postprocess import decode_yolo_head
from uavtrack.detect.preprocess import letterbox


class OnnxDetector(DetectorBackend):
    """Run an exported YOLOv8/YOLO11 ONNX graph through ONNX Runtime.

    Args:
        model_path: Path to the ``.onnx`` file.
        conf_threshold: Minimum score for a detection to be reported.
        iou_threshold: NMS IoU threshold.
        labels: Class names, index-aligned with the trained model.
        providers: ONNX Runtime execution providers, highest priority first.
            Defaults to CPU only so benchmark numbers are reproducible.
        intra_op_threads: Thread cap for the CPU provider. Pinning this makes
            latency measurements comparable across machines; ``None`` lets
            ONNX Runtime choose.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        labels: tuple[str, ...] = ("uav",),
        providers: list[str] | None = None,
        intra_op_threads: int | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "onnxruntime is required for OnnxDetector: pip install 'uavtrack[onnx]'"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads

        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=providers or ["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # Exported YOLO graphs are NCHW; a dynamic axis shows up as a string or
        # None, in which case we fall back to the conventional 640x640.
        shape = self._session.get_inputs()[0].shape
        height = shape[2] if isinstance(shape[2], int) else 640
        width = shape[3] if isinstance(shape[3], int) else 640
        self._input_size = (int(width), int(height))

        self._model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = labels

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def name(self) -> str:
        provider = self._session.get_providers()[0].replace("ExecutionProvider", "")
        return f"onnx:{provider.lower()}"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        padded, transform = letterbox(frame, self._input_size)

        # BGR -> RGB, HWC -> NCHW, uint8 -> float32 in [0, 1].
        blob = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        raw = self._session.run(None, {self._input_name: blob})[0]
        boxes, scores, class_ids = decode_yolo_head(raw, self.conf_threshold, self.iou_threshold)
        boxes = transform.to_frame(boxes)
        return [
            Detection.from_xyxy(box, score, int(class_id))
            for box, score, class_id in zip(boxes, scores, class_ids, strict=True)
        ]

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]
