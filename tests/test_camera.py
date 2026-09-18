"""Frame sources.

The device-backed sources cannot be tested without a device, but the parts that
actually go wrong can be: which source a specification resolves to, and whether
timestamps are deterministic. A source that timestamps at *delivery* rather than
at capture hides its own buffering delay from the controller, which then
compensates for less lag than really exists -- so the contract is worth pinning
even where the device is not.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from uavtrack.io.camera import ImageSequenceSource, OpenCVCamera, build_source


@pytest.fixture
def frames_dir(tmp_path):
    """A directory of three numbered frames."""
    for index in range(3):
        image = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"{index:05d}.jpg"), image)
    return tmp_path


def test_image_sequence_yields_every_frame_in_order(frames_dir):
    source = ImageSequenceSource(frames_dir, fps=30.0)
    frames = list(source.frames())

    assert len(frames) == 3
    # Frames were written with increasing brightness, so order is checkable.
    means = [float(frame.mean()) for _, frame in frames]
    assert means == sorted(means)


def test_image_sequence_timestamps_are_deterministic(frames_dir):
    """Synthesised from the nominal rate, so downstream timing is reproducible
    rather than dependent on how fast the disk is."""
    timestamps = [t for t, _ in ImageSequenceSource(frames_dir, fps=20.0).frames()]
    assert timestamps == pytest.approx([0.0, 0.05, 0.10])

    faster = [t for t, _ in ImageSequenceSource(frames_dir, fps=50.0).frames()]
    assert faster == pytest.approx([0.0, 0.02, 0.04])


def test_image_sequence_is_replayable(frames_dir):
    source = ImageSequenceSource(frames_dir)
    assert len(list(source.frames())) == len(list(source.frames()))


def test_image_sequence_skips_unreadable_files(frames_dir):
    (frames_dir / "broken.jpg").write_bytes(b"not an image")
    source = ImageSequenceSource(frames_dir)
    assert len(source.paths) == 4
    assert len(list(source.frames())) == 3


def test_image_sequence_rejects_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no images"):
        ImageSequenceSource(tmp_path)


def test_image_sequence_pattern_selects_and_orders(frames_dir):
    (frames_dir / "other.png").write_bytes(b"")
    assert len(ImageSequenceSource(frames_dir, pattern="*.jpg").paths) == 3


def test_build_source_resolves_a_directory_to_an_image_sequence(frames_dir):
    source = build_source(str(frames_dir))
    assert isinstance(source, ImageSequenceSource)
    source.close()


def test_build_source_resolves_a_digit_to_a_camera_index(monkeypatch):
    captured = {}

    class FakeCamera:
        def __init__(self, source, width=None, height=None, fps=None):
            captured["source"] = source

    monkeypatch.setattr("uavtrack.io.camera.OpenCVCamera", FakeCamera)
    build_source("2")
    assert captured["source"] == 2, "a bare digit is a device index, not a filename"


def test_build_source_resolves_a_path_to_opencv(monkeypatch):
    captured = {}

    class FakeCamera:
        def __init__(self, source, width=None, height=None, fps=None):
            captured["source"] = source

    monkeypatch.setattr("uavtrack.io.camera.OpenCVCamera", FakeCamera)
    build_source("rtsp://camera.local/stream")
    assert captured["source"] == "rtsp://camera.local/stream"


def test_build_source_requires_picamera2_for_the_picamera_spec():
    """On a machine without Picamera2 this must say so, not fail obscurely."""
    pytest.importorskip  # noqa: B018 - documents intent when picamera2 IS present
    try:
        import picamera2  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="picamera2"):
            build_source("picamera")
    else:  # pragma: no cover - only on a Raspberry Pi
        pytest.skip("picamera2 is installed; the failure path cannot be exercised")


def test_opencv_camera_raises_on_an_unopenable_source():
    with pytest.raises(RuntimeError, match="could not open"):
        OpenCVCamera("this-file-does-not-exist.mp4")
