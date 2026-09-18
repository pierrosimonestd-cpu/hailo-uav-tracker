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


# ------------------------------------------------------- the shrinking kernel
#
# The resolutions here are deliberate. At an exact 2:1 downscale OpenCV's
# bilinear filter averages the same 2x2 block that INTER_AREA does, so the two
# kernels are bit-identical and any test comparing them at 1280x720 -> 640
# passes or fails for reasons that have nothing to do with the code. 1920x1080
# gives a 3:1 ratio, where bilinear samples 2 of every 3 pixels per axis and the
# kernels genuinely differ. docs/benchmarks.md has the measured consequence.


def test_the_downscale_kernel_changes_the_pixels_at_a_non_dyadic_ratio():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)

    area, _ = letterbox(image, (640, 640), downscale="area")
    linear, _ = letterbox(image, (640, 640), downscale="linear")

    assert area.shape == linear.shape
    assert not np.array_equal(area, linear)


def test_the_kernels_coincide_at_an_exact_two_to_one_downscale():
    """Not a quirk to work around -- the reason the ablation shows no gain at
    the deployed 1280x720 capture resolution. Pinned so nobody 'fixes' the
    documentation by assuming the kernel always matters."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    area, _ = letterbox(image, (640, 640), downscale="area")
    linear, _ = letterbox(image, (640, 640), downscale="linear")

    assert np.array_equal(area, linear)


def test_the_kernels_agree_on_the_geometry():
    """Only the resampling differs; the transform must not."""
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    _, area = letterbox(image, (640, 640), downscale="area")
    _, linear = letterbox(image, (640, 640), downscale="linear")

    assert (area.scale, area.pad_x, area.pad_y) == (linear.scale, linear.pad_x, linear.pad_y)


def test_bilinear_loses_thin_targets_at_some_sub_pixel_offsets():
    """The actual reason the default is `area`, and it is not "more signal".

    Averaged over sub-pixel offsets the two kernels pass the same total energy.
    The difference is variance. A one-pixel-wide line -- what a distant UAV
    becomes at range -- is rendered by area-averaging at the same intensity
    wherever it falls, while bilinear at a 3:1 ratio samples two columns of
    every three: the line may land on a sample and come through at full
    intensity, or land between samples and disappear entirely.

    A detector does not benefit from the offsets where bilinear over-delivers.
    It fails on the ones where the target is gone.
    """
    area_peaks, linear_peaks = [], []
    for offset in range(24):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[:, 300 + offset] = 255

        area, transform = letterbox(image, (640, 640), downscale="area")
        linear, _ = letterbox(image, (640, 640), downscale="linear")

        # Inside the letterbox border only: the 114 padding is brighter than an
        # area-averaged thin line and would mask what is being measured.
        top = int(transform.pad_y)
        bottom = top + int(round(image.shape[0] * transform.scale))
        area_peaks.append(int(area[top:bottom].max()))
        linear_peaks.append(int(linear[top:bottom].max()))

    assert len(set(area_peaks)) == 1, (
        f"area-averaging should be offset-invariant, saw {sorted(set(area_peaks))}"
    )
    assert min(linear_peaks) == 0, "expected bilinear to drop the target entirely somewhere"
    assert max(linear_peaks) > area_peaks[0], "expected bilinear to overshoot somewhere"

    # The mean is a wash; the distribution is not.
    assert sum(linear_peaks) / len(linear_peaks) == pytest.approx(area_peaks[0], rel=0.05)
    worse = sum(1 for peak in linear_peaks if peak < area_peaks[0])
    assert worse > len(linear_peaks) // 2, (
        f"only {worse}/{len(linear_peaks)} offsets were worse than area-averaging"
    )


def test_an_unknown_kernel_is_rejected_by_name():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown downscale kernel"):
        letterbox(image, (640, 640), downscale="lanczos")
