"""NumPy box helpers shared by the service and the ONNX postprocess."""

from __future__ import annotations

import numpy as np

from vision_detect.constants import INPUT_SIZE


def box_iou(box_a: list[float] | np.ndarray, box_b: list[float] | np.ndarray) -> float:
    """Intersection over union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression. Returns indices into ``boxes`` in score order."""
    if len(scores) == 0:
        return np.empty(0, dtype=np.int64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def scale_boxes(
    boxes: np.ndarray, orig_width: int, orig_height: int, input_size: int = INPUT_SIZE
) -> np.ndarray:
    """Map xyxy boxes from the square network input back to the original image."""
    scaled = np.array(boxes, dtype=np.float32, copy=True)
    if scaled.size == 0:
        return scaled.reshape(0, 4)
    scaled[:, [0, 2]] *= float(orig_width) / float(input_size)
    scaled[:, [1, 3]] *= float(orig_height) / float(input_size)
    return scaled


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp xyxy boxes to the image extent. ``width`` and ``height`` are exclusive maxima."""
    clipped = np.array(boxes, dtype=np.float32, copy=True)
    if clipped.size == 0:
        return clipped.reshape(0, 4)
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0.0, float(width))
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0.0, float(height))
    return clipped


def postprocess(
    boxes: np.ndarray,
    scores: np.ndarray,
    orig_width: int,
    orig_height: int,
    score_threshold: float,
    iou_threshold: float,
    input_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Threshold, suppress, scale, and clip raw network candidates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = scores >= score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    if len(scores) == 0:
        return boxes.reshape(0, 4), scores
    order = nms(boxes, scores, iou_threshold)
    boxes = scale_boxes(boxes[order], orig_width, orig_height, input_size)
    scores = scores[order]
    boxes = clip_boxes(boxes, orig_width, orig_height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid]
