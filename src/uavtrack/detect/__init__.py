"""Detection backends and the shared pre/post-processing used by all of them.

Every backend returns the same :class:`Detection` objects in *original frame*
pixel coordinates, so the tracker and the controller never need to know whether
inference ran on a Hailo-8L, on ONNX Runtime or on PyTorch.
"""

from uavtrack.detect.base import Detection, DetectorBackend
from uavtrack.detect.factory import build_detector

__all__ = ["Detection", "DetectorBackend", "build_detector"]
