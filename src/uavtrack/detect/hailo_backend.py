"""Hailo-8L detection backend (Raspberry Pi AI HAT+, 13 TOPS).

The HEF produced by ``tools/compile_hailo.sh`` carries the YOLO NMS node on
chip, so the device returns final detections and the Pi CPU stays free for
tracking and the control loop. The raw-head path is kept as a fallback for HEFs
compiled without that node.

On-chip NMS output layout (HailoRT ``HAILO_NMS`` format order): a list with one
entry per class, each an ``(n, 5)`` array of ``[y_min, x_min, y_max, x_max,
score]`` normalised to the *letterboxed network input*, which is why the
letterbox transform is carried through and inverted here.

This module imports ``hailo_platform`` lazily: the rest of the package, its
tests and the accuracy benchmarks all run on machines with no NPU attached.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from typing import Any

import numpy as np

from uavtrack.detect.base import Detection, DetectorBackend
from uavtrack.detect.postprocess import decode_yolo_head
from uavtrack.detect.preprocess import Letterbox, letterbox


class HailoDetector(DetectorBackend):
    """Synchronous wrapper over the HailoRT asynchronous inference API.

    HailoRT only exposes ``run_async``. A tracking pipeline wants a blocking
    call per frame, so each :meth:`infer` posts one job and waits on a queue the
    completion callback writes to. The device context is opened once in the
    constructor: configuring a HEF costs hundreds of milliseconds and must never
    happen inside the control loop.

    Args:
        hef_path: Compiled ``.hef`` model.
        conf_threshold: Minimum score for a detection to be reported.
        iou_threshold: Host-side NMS IoU threshold. Ignored when the HEF carries
            its own NMS node.
        labels: Class names, index-aligned with the trained model.
        timeout_ms: How long to wait for one inference before giving up.
    """

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        labels: tuple[str, ...] = ("uav",),
        timeout_ms: int = 10_000,
    ) -> None:
        try:
            from hailo_platform import HEF, FormatType, HailoSchedulingAlgorithm, VDevice
            from hailo_platform.pyhailort.pyhailort import FormatOrder
        except ImportError as exc:  # pragma: no cover - device-only path
            raise ImportError(
                "hailo_platform not found. Install HailoRT and its Python bindings on the "
                "Raspberry Pi (see docs/hailo-deployment.md); on a desktop use the ONNX backend."
            ) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self._vdevice = VDevice(params)
        self._hef = HEF(hef_path)
        self._infer_model = self._vdevice.create_infer_model(hef_path)
        self._infer_model.set_batch_size(1)

        # Feed the device uint8 straight from the letterbox: quantisation to the
        # model input scale happens on chip, so we skip a float conversion and
        # a 4x larger PCIe transfer per frame.
        self._infer_model.input().set_format_type(FormatType.UINT8)

        output_info = self._hef.get_output_vstream_infos()
        nms_orders = {FormatOrder.HAILO_NMS}
        for extra in ("HAILO_NMS_BY_CLASS", "HAILO_NMS_BY_SCORE"):
            if hasattr(FormatOrder, extra):
                nms_orders.add(getattr(FormatOrder, extra))
        self._nms_on_chip = self._infer_model.outputs[0].format.order in nms_orders

        for info in output_info:
            self._infer_model.output(info.name).set_format_type(FormatType.FLOAT32)
        self._output_names = [info.name for info in output_info]

        self._config_ctx = self._infer_model.configure()
        self._configured = self._config_ctx.__enter__()

        height, width = self._hef.get_input_vstream_infos()[0].shape[:2]
        self._input_size = (int(width), int(height))

        self._timeout_ms = timeout_ms
        self._results: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._closed = False

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.labels = labels
        self._hef_path = hef_path

    @property
    def input_size(self) -> tuple[int, int]:
        return self._input_size

    @property
    def name(self) -> str:
        return "hailo8l"

    @property
    def nms_on_chip(self) -> bool:
        """True when the HEF runs NMS on the NPU rather than on the host CPU."""
        return self._nms_on_chip

    def infer(self, frame: np.ndarray) -> list[Detection]:
        padded, transform = letterbox(frame, self._input_size)
        blob = np.ascontiguousarray(padded[:, :, ::-1])  # BGR -> RGB, stays uint8

        with self._lock:
            output_buffers = {
                name: np.empty(self._infer_model.output(name).shape, dtype=np.float32)
                for name in self._output_names
            }
            bindings = self._configured.create_bindings(output_buffers=output_buffers)
            bindings.input().set_buffer(blob)

            self._configured.wait_for_async_ready(timeout_ms=self._timeout_ms)
            job = self._configured.run_async([bindings], self._on_complete)
            job.wait(self._timeout_ms)

            error = self._results.get(timeout=self._timeout_ms / 1000.0)

        if error is not None:
            raise RuntimeError(f"Hailo inference failed: {error}")

        if self._nms_on_chip:
            return self._parse_nms_output(bindings, transform)
        return self._parse_raw_output(output_buffers[self._output_names[0]], transform)

    def _on_complete(self, completion_info: Any, bindings_list: Any = None) -> None:
        """HailoRT completion callback, invoked on a HailoRT worker thread."""
        error = getattr(completion_info, "exception", None)
        # One job is in flight at a time, so the queue cannot legitimately be
        # full; dropping the duplicate is safer than blocking a HailoRT thread.
        with contextlib.suppress(queue.Full):
            self._results.put_nowait(error)

    def _parse_nms_output(self, bindings: Any, transform: Letterbox) -> list[Detection]:
        """Convert the per-class on-chip NMS output into detections."""
        per_class = bindings.output().get_buffer()

        boxes: list[np.ndarray] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for class_id, class_dets in enumerate(per_class):
            for det in np.asarray(class_dets, dtype=np.float32).reshape(-1, 5):
                score = float(det[4])
                if score < self.conf_threshold:
                    continue
                y_min, x_min, y_max, x_max = (float(v) for v in det[:4])
                boxes.append(np.array([x_min, y_min, x_max, y_max], dtype=np.float32))
                scores.append(score)
                class_ids.append(class_id)

        if not boxes:
            return []

        mapped = transform.normalised_to_frame(np.stack(boxes))
        detections = [
            Detection.from_xyxy(box, score, class_id)
            for box, score, class_id in zip(mapped, scores, class_ids, strict=True)
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def _parse_raw_output(self, raw: np.ndarray, transform: Letterbox) -> list[Detection]:
        """Decode a HEF compiled without the NMS node, on the host CPU."""
        boxes, scores, class_ids = decode_yolo_head(
            raw if raw.ndim == 3 else raw[None], self.conf_threshold, self.iou_threshold
        )
        boxes = transform.to_frame(boxes)
        return [
            Detection.from_xyxy(box, score, int(class_id))
            for box, score, class_id in zip(boxes, scores, class_ids, strict=True)
        ]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._config_ctx.__exit__(None, None, None)
        finally:
            self._vdevice.release()
