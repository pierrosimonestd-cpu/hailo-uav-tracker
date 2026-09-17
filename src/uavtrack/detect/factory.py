"""Backend selection from a config dict, with a deliberate absence of magic.

The pipeline config names a backend; nothing else in the codebase imports a
concrete backend class. That keeps the Raspberry Pi install free of PyTorch and
keeps the desktop install free of HailoRT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uavtrack.detect.base import DetectorBackend

_EXTENSION_HINTS = {".hef": "hailo", ".onnx": "onnx", ".pt": "ultralytics"}


def infer_backend_from_path(model_path: str) -> str:
    """Guess the backend from a model file extension.

    Raises:
        ValueError: If the extension is not one this project knows how to run.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix not in _EXTENSION_HINTS:
        raise ValueError(
            f"cannot infer a backend from {model_path!r}; "
            f"expected one of {sorted(_EXTENSION_HINTS)}"
        )
    return _EXTENSION_HINTS[suffix]


def build_detector(
    model_path: str,
    backend: str | None = None,
    **kwargs: Any,
) -> DetectorBackend:
    """Construct the detection backend for ``model_path``.

    Args:
        model_path: ``.hef``, ``.onnx`` or ``.pt`` model file.
        backend: Force a backend instead of deducing it from the extension.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A ready-to-use detector. Callers should still call
        :meth:`DetectorBackend.warmup` before timing anything.
    """
    backend = backend or infer_backend_from_path(model_path)

    if backend == "hailo":
        from uavtrack.detect.hailo_backend import HailoDetector

        return HailoDetector(model_path, **kwargs)

    if backend == "onnx":
        from uavtrack.detect.onnx_backend import OnnxDetector

        return OnnxDetector(model_path, **kwargs)

    if backend == "ultralytics":
        from uavtrack.detect.ultralytics_backend import UltralyticsDetector

        return UltralyticsDetector(model_path, **kwargs)

    raise ValueError(f"unknown backend {backend!r}; expected hailo, onnx or ultralytics")
