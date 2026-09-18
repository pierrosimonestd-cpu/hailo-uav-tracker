"""Backend selection.

The factory is the only place that knows which concrete backend exists, which
is what keeps PyTorch out of the Raspberry Pi install and HailoRT out of the
desktop one. Its error messages matter: a wrong extension or a typo'd backend
name should say what was expected, not fail somewhere deep inside an import.
"""

from __future__ import annotations

import pytest

from uavtrack.detect.factory import build_detector, infer_backend_from_path


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("models/uav.hef", "hailo"),
        ("models/uav.onnx", "onnx"),
        ("runs/train/weights/best.pt", "ultralytics"),
        ("MODELS/UAV.HEF", "hailo"),
        ("/absolute/path/to/model.onnx", "onnx"),
        ("model.int8.onnx", "onnx"),
    ],
)
def test_backend_is_inferred_from_the_extension(path, expected):
    assert infer_backend_from_path(path) == expected


@pytest.mark.parametrize("path", ["model.pth", "model", "model.tflite", "model.engine"])
def test_unknown_extensions_are_rejected_with_the_alternatives(path):
    with pytest.raises(ValueError) as exc_info:
        infer_backend_from_path(path)

    message = str(exc_info.value)
    assert path in message
    for extension in (".hef", ".onnx", ".pt"):
        assert extension in message, "the error should name what it expected"


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="hailo, onnx or ultralytics"):
        build_detector("model.onnx", backend="tensorrt")


def test_an_explicit_backend_overrides_the_extension(monkeypatch):
    """So an ONNX graph can be run through a backend the extension does not imply."""
    built = {}

    class FakeOnnx:
        def __init__(self, path, **kwargs):
            built["path"] = path
            built["kwargs"] = kwargs

    import uavtrack.detect.onnx_backend as module

    monkeypatch.setattr(module, "OnnxDetector", FakeOnnx)
    build_detector("weights.pt", backend="onnx", conf_threshold=0.5)

    assert built["path"] == "weights.pt"
    assert built["kwargs"]["conf_threshold"] == 0.5


def test_keyword_arguments_reach_the_backend(monkeypatch):
    captured = {}

    class FakeOnnx:
        def __init__(self, path, **kwargs):
            captured.update(kwargs)

    import uavtrack.detect.onnx_backend as module

    monkeypatch.setattr(module, "OnnxDetector", FakeOnnx)
    build_detector("m.onnx", conf_threshold=0.4, iou_threshold=0.6, labels=("uav",))

    assert captured == {"conf_threshold": 0.4, "iou_threshold": 0.6, "labels": ("uav",)}


def test_hailo_backend_import_failure_names_the_cause():
    """On a machine with no HailoRT, the message must point at the deployment
    guide rather than surfacing a bare ImportError for a package nobody has
    heard of."""
    pytest.importorskip  # noqa: B018
    try:
        import hailo_platform  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as exc_info:
            build_detector("model.hef")
        message = str(exc_info.value)
        assert "hailo_platform" in message
        assert "docs/hailo-deployment.md" in message
    else:  # pragma: no cover - only on a Pi with the HAT installed
        pytest.skip("hailo_platform is installed; the failure path cannot be exercised")
