"""Host-side YOLO head decoding and NMS, in NumPy only.

The Raspberry Pi build of this project must not depend on PyTorch, so the
decode path is written against NumPy. It is exercised by the ONNX backend and
by the Hailo backend whenever the HEF was compiled *without* the on-chip NMS
node.
"""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[cx, cy, w, h]`` rows to ``[x1, y1, x2, y2]``."""
    out = np.empty_like(boxes)
    half_w = boxes[:, 2] * 0.5
    half_h = boxes[:, 3] * 0.5
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression.

    Args:
        boxes: ``(n, 4)`` xyxy.
        scores: ``(n,)`` confidences.
        iou_threshold: Boxes overlapping a kept box by more than this are dropped.

    Returns:
        Indices of kept boxes, ordered by descending score.
    """
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        inter_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def decode_yolo_head(
    raw: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/YOLO11 detection head into boxes, scores and classes.

    The exported head has shape ``(1, 4 + num_classes, num_anchors)``: four box
    regression channels in ``cxcywh`` network-pixel units followed by one
    sigmoid-activated score per class. There is no objectness channel and no
    separate softmax -- classes are independent, so the class score *is* the
    confidence.

    Args:
        raw: Model output, ``(1, 4 + nc, anchors)`` or ``(1, anchors, 4 + nc)``.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: NMS IoU threshold, applied per class.
        max_detections: Hard cap on returned detections.
        num_classes: Number of classes the model predicts. Supplying it removes
            all ambiguity about which axis is which; without it the axis with
            fewer entries is taken to be the channel axis, which is correct for
            any real export (thousands of anchors, a handful of channels) but
            not for a degenerate toy tensor.

    Returns:
        ``(boxes_xyxy, scores, class_ids)`` in network-input pixels.

    Raises:
        ValueError: If no axis of ``raw`` matches ``4 + num_classes``, or if the
            tensor carries box channels but no class channels.
    """
    pred = np.squeeze(raw, axis=0) if raw.ndim == 3 else raw

    if num_classes is not None:
        expected = 4 + num_classes
        if pred.shape[0] == expected:
            pass
        elif pred.shape[1] == expected:
            pred = pred.T
        else:
            raise ValueError(
                f"neither axis of output shape {raw.shape} matches 4 + num_classes = {expected}"
            )
    elif pred.shape[0] > pred.shape[1]:
        # Channel axis is the short one: the anchor axis is in the thousands.
        pred = pred.T

    boxes_xywh = pred[:4].T
    class_scores = pred[4:].T
    if class_scores.size == 0:
        raise ValueError(f"model output has no class channels: shape {raw.shape}")

    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep = scores >= conf_threshold
    if not np.any(keep):
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
        )

    boxes = xywh_to_xyxy(boxes_xywh[keep].astype(np.float32))
    scores = scores[keep].astype(np.float32)
    class_ids = class_ids[keep].astype(np.int64)

    # Class-aware NMS via coordinate offsetting: shifting each class into its
    # own disjoint region of the plane makes one NMS pass equivalent to running
    # NMS per class, without the Python loop over classes.
    if class_ids.max(initial=0) > 0:
        offset = class_ids.astype(np.float32) * (boxes.max() + 1.0)
        shifted = boxes + offset[:, None]
    else:
        shifted = boxes

    kept = nms(shifted, scores, iou_threshold)[:max_detections]
    return boxes[kept], scores[kept], class_ids[kept]
