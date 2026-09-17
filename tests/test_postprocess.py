"""NMS and YOLO head decoding."""

from __future__ import annotations

import numpy as np
import pytest

from uavtrack.detect.postprocess import decode_yolo_head, nms, xywh_to_xyxy


def test_xywh_to_xyxy():
    boxes = np.array([[50.0, 60.0, 20.0, 10.0]], dtype=np.float32)
    assert np.allclose(xywh_to_xyxy(boxes), [[40.0, 55.0, 60.0, 65.0]])


def test_nms_suppresses_overlaps_and_keeps_distinct_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_nms_orders_by_score_not_by_input_order():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.2, 0.9], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 0]


def test_nms_on_empty_input():
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5).shape == (0,)


def test_nms_threshold_is_inclusive_of_lower_overlap():
    """Boxes overlapping at exactly the threshold are kept."""
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    # Intersection 50, union 150 -> IoU = 1/3.
    assert len(nms(boxes, scores, 0.34)) == 2
    assert len(nms(boxes, scores, 0.33)) == 1


def _head(boxes_xywh, scores, num_classes=1, anchors=None):
    """Build a fake YOLO head tensor of shape (1, 4 + nc, anchors)."""
    anchors = anchors or max(len(boxes_xywh), 1)
    raw = np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)
    for i, (box, (cls, score)) in enumerate(zip(boxes_xywh, scores, strict=True)):
        raw[0, :4, i] = box
        raw[0, 4 + cls, i] = score
    return raw


def test_decode_single_detection():
    raw = _head([(50, 60, 20, 10)], [(0, 0.9)])
    boxes, scores, classes = decode_yolo_head(raw, 0.25, 0.5, num_classes=1)
    assert np.allclose(boxes, [[40, 55, 60, 65]])
    assert scores.tolist() == pytest.approx([0.9])
    assert classes.tolist() == [0]


def test_decode_accepts_transposed_layout():
    """Some exports emit (1, anchors, 4 + nc)."""
    raw = _head([(50, 60, 20, 10)], [(0, 0.9)])
    direct = decode_yolo_head(raw, 0.25, 0.5, num_classes=1)
    transposed = decode_yolo_head(raw[0].T[None], 0.25, 0.5, num_classes=1)
    assert np.allclose(direct[0], transposed[0])


def test_decode_respects_confidence_threshold():
    raw = _head([(50, 60, 20, 10)], [(0, 0.2)])
    boxes, _, _ = decode_yolo_head(raw, 0.25, 0.5, num_classes=1)
    assert boxes.shape == (0, 4)


def test_decode_nms_is_class_aware():
    """Identical boxes of different classes must both survive."""
    raw = _head([(50, 60, 20, 10), (50, 60, 20, 10)], [(0, 0.9), (1, 0.8)], num_classes=2)
    boxes, _, classes = decode_yolo_head(raw, 0.25, 0.5, num_classes=2)
    assert len(boxes) == 2
    assert sorted(classes.tolist()) == [0, 1]

    # Same class, same box: one survives.
    raw_same = _head([(50, 60, 20, 10), (50, 60, 20, 10)], [(0, 0.9), (0, 0.8)], num_classes=2)
    boxes_same, _, _ = decode_yolo_head(raw_same, 0.25, 0.5, num_classes=2)
    assert len(boxes_same) == 1


def test_decode_uses_the_highest_scoring_class():
    raw = np.zeros((1, 7, 1), dtype=np.float32)
    raw[0, :4, 0] = [50, 60, 20, 10]
    raw[0, 4:, 0] = [0.1, 0.7, 0.3]
    _, scores, classes = decode_yolo_head(raw, 0.25, 0.5, num_classes=3)
    assert classes.tolist() == [1]
    assert scores.tolist() == pytest.approx([0.7])


def test_decode_caps_detection_count():
    n = 50
    boxes = [(10 + 100 * i, 10, 5, 5) for i in range(n)]
    raw = _head(boxes, [(0, 0.9)] * n)
    result, _, _ = decode_yolo_head(raw, 0.25, 0.5, max_detections=10, num_classes=1)
    assert len(result) == 10


def test_decode_rejects_a_mismatched_num_classes():
    raw = _head([(50, 60, 20, 10)], [(0, 0.9)])
    with pytest.raises(ValueError, match="num_classes"):
        decode_yolo_head(raw, 0.25, 0.5, num_classes=17)


def test_decode_infers_layout_without_a_hint_on_realistic_shapes():
    raw = np.zeros((1, 5, 8400), dtype=np.float32)
    raw[0, :4, 7] = [100, 100, 20, 20]
    raw[0, 4, 7] = 0.8
    boxes, _, _ = decode_yolo_head(raw, 0.25, 0.5)
    assert np.allclose(boxes, [[90, 90, 110, 110]])
    boxes_t, _, _ = decode_yolo_head(raw[0].T[None], 0.25, 0.5)
    assert np.allclose(boxes, boxes_t)
