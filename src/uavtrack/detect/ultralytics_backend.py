"""PyTorch / Ultralytics detection backend.

Used for training-time validation and as the full-precision reference the
quantised backends are compared against. It is deliberately *not* a dependency
of the Raspberry Pi pipeline.
"""

from __future__ import annotations

import numpy as np

from uavtrack.detect.base import Detection, DetectorBackend


class UltralyticsDetector(DetectorBackend):
    """Wrap an Ultralytics ``YOLO`` model behind the common backend interface.

    Args:
        model_path: A ``.pt`` checkpoint.
        conf_threshold: Minimum score for a detection to be reported.
        iou_threshold: NMS IoU threshold.
        imgsz: Inference resolution. Must match what the model was trained at
            for the accuracy numbers to be comparable.
        device: ``"cpu"``, ``"cuda:0"``, ... Passed straight to Ultralytics.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for UltralyticsDetector: pip install 'uavtrack[train]'"
            ) from exc

        self._model = YOLO(model_path)
        self._imgsz = imgsz
        self._device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        names = self._model.names
        self.labels = tuple(names[i] for i in sorted(names))
        self._model_path = model_path

    @property
    def input_size(self) -> tuple[int, int]:
        return (self._imgsz, self._imgsz)

    @property
    def name(self) -> str:
        return f"ultralytics:{self._device}"

    def infer(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self._device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            detections.extend(
                Detection.from_xyxy(box, float(score), int(class_id))
                for box, score, class_id in zip(xyxy, scores, classes, strict=True)
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections
