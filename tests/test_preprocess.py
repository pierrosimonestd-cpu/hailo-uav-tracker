"""Letterbox geometry and its inverse.

A silent error here shifts every box by a constant, which looks like a badly
trained model and is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from uavtrack.detect.preprocess import letterbox


@pytest.mark.parametrize(
    ("src_w", "src_h", "net"),
    [
        (1280, 720, (640, 640)),
        (720, 1280, (640, 640)),
        (640, 640, (640, 640)),
        (1920, 1080, (416, 416)),
        (333, 777, (640, 640)),
    ],
)
def test_letterbox_output_shape_and_aspect(src_w, src_h, net):
    image = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    padded, transform = letterbox(image, net)

    assert padded.shape == (net[1], net[0], 3)
    assert transform.scale == pytest.approx(min(net[0] / src_w, net[1] / src_h))
    # Padding is applied on exactly one axis: the other is filled by the resize.
    assert transform.pad_x == 0 or transform.pad_y == 0


@pytest.mark.parametrize(("src_w", "src_h"), [(1280, 720), (720, 1280), (640, 640), (333, 777)])
def test_full_frame_box_round_trips(src_w, src_h):
    """The box covering the whole source must map back to the whole source."""
    image = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    _, t = letterbox(image, (640, 640))

    inner = np.array(
        [[t.pad_x, t.pad_y, t.pad_x + src_w * t.scale, t.pad_y + src_h * t.scale]],
        dtype=np.float32,
    )
    mapped = t.to_frame(inner)[0]

    assert mapped[0] == pytest.approx(0.0, abs=1.0)
    assert mapped[1] == pytest.approx(0.0, abs=1.0)
    assert mapped[2] == pytest.approx(src_w - 1, abs=1.0)
    assert mapped[3] == pytest.approx(src_h - 1, abs=1.0)


def test_arbitrary_points_round_trip():
    """Forward-project known points, invert, and land where we started."""
    src_w, src_h = 1280, 720
    image = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    _, t = letterbox(image, (640, 640))

    rng = np.random.default_rng(0)
    xs = rng.uniform(0, src_w - 1, 64)
    ys = rng.uniform(0, src_h - 1, 64)

    net_boxes = np.stack(
        [
            xs * t.scale + t.pad_x,
            ys * t.scale + t.pad_y,
            xs * t.scale + t.pad_x + 10,
            ys * t.scale + t.pad_y + 10,
        ],
        axis=1,
    ).astype(np.float32)

    mapped = t.to_frame(net_boxes)
    assert np.allclose(mapped[:, 0], xs, atol=1e-3)
    assert np.allclose(mapped[:, 1], ys, atol=1e-3)


def test_normalised_mapping_matches_pixel_mapping():
    """Hailo's normalised output and the pixel path must agree."""
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, t = letterbox(image, (640, 640))

    pixel_box = np.array([[100.0, 200.0, 300.0, 400.0]], dtype=np.float32)
    normalised = pixel_box / 640.0

    assert np.allclose(t.to_frame(pixel_box), t.normalised_to_frame(normalised), atol=1e-3)


def test_boxes_are_clipped_to_the_frame():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, t = letterbox(image, (640, 640))

    # A box entirely inside the padded border maps outside the real frame.
    mapped = t.to_frame(np.array([[-500.0, -500.0, 2000.0, 2000.0]], dtype=np.float32))[0]
    assert mapped[0] >= 0 and mapped[1] >= 0
    assert mapped[2] <= 1279 and mapped[3] <= 719


def test_empty_input_is_handled():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, t = letterbox(image, (640, 640))
    assert t.to_frame(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)
    assert t.normalised_to_frame(np.zeros((0, 4), dtype=np.float32)).shape == (0, 4)
